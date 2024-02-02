import wandb
import logging
from dataclasses import dataclass
import simple_parsing

import torch
import torch.nn as nn

from accelerate import Accelerator
from datasets import load_dataset
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama import LlamaConfig, LlamaForCausalLM
from trl import SFTTrainer

from data import create_alpaca_prompt_with_response, create_chatml_fromvalue
from utils import freeze

logging.basicConfig(level=logging.INFO)

WANDB_PROJECT = "shearllama"
ENTITY = "llm_surgery"
# DATASET_NAME = "togethercomputer/RedPajama-Data-1T-Sample"
# DATASET_NAME = "vicgalle/alpaca-gpt4"
DATASET_SIZE = 50 # other options: 100 200
DATASET_NAME = f"typeof/OH-2.5-{DATASET_SIZE}k"
# MODEL_ID = "mistralai/Mistral-7B-v0.1"
# MODEL_ID = "NousResearch/Llama-2-7b-hf"
MODEL_ID = "./models/mistral_7b_12_layers_start"  # only first 12 layers of Mistral 7B
LAST_CHECKPOINT = None

CHATML = True

@dataclass
class Config(simple_parsing.Serializable):
    model_id: str = MODEL_ID
    output_dir: str = None
    resume_from_checkpoint: str = LAST_CHECKPOINT
    torch_compile: bool = False
    batch_size: int = 2
    learning_rate: float = 1e-4
    gradient_accumulation_steps: int = 8
    n_layers: int = 12
    n_freeze: int = None
    test_size: float = 0.05
    seed: int = 42
    max_seq_length: int = 1024
    max_steps: int = -1
    num_train_epochs: int = 3
    save: bool = True
    log_model: bool = True
    eval: bool = False
    tags: str = "alpaca,12layers"


config: Config = simple_parsing.parse(Config)

accelerator = Accelerator()

if accelerator.is_main_process:
    wandb.init(project=WANDB_PROJECT, 
               entity=ENTITY, 
               job_type="train", 
               config={"init_args": config},
               tags=config.tags.split(","))
    # estimation
    effective_batch_size = config.max_seq_length*config.batch_size*config.gradient_accumulation_steps*accelerator.num_processes
    logging.info(f"\nTraining with an effective batch_size of: {effective_batch_size}\n")

ds = load_dataset(DATASET_NAME)["train"].shuffle(seed=config.seed)
if config.test_size>0:
    ds = ds.train_test_split(test_size=config.test_size)
    train_ds = ds["train"]
    test_ds = ds["test"]
else:
    train_ds = ds
    test_ds = None

if not config.output_dir:
    config.output_dir = f"./models/model_{config.n_layers}_layers_ft"

training_args = TrainingArguments(
    output_dir=config.output_dir,
    report_to="wandb",
    per_device_train_batch_size=config.batch_size,
    per_device_eval_batch_size=config.batch_size,
    bf16=True,
    gradient_checkpointing = True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    torch_compile=config.torch_compile,
    learning_rate=config.learning_rate,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    max_steps=config.max_steps,
    num_train_epochs=config.num_train_epochs,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    evaluation_strategy="no",
    # logging strategies
    logging_dir=f"{config.output_dir}/logs",
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="no",
)

model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        use_flash_attention_2=True,
        )

# Chop the model down to n_layers
if config.n_layers: 
    model.model.layers = model.model.layers[:config.n_layers]

# freeze layers
if config.n_freeze:
    freeze(model, freeze_embed=True, n_freeze=config.n_freeze, module_name="layers")

if CHATML:
    # 💭 perhaps we move this config ugliness somewhere else ?
    # TODO: make configuration modular...
    chat_template = (
        "{% if not add_generation_prompt is defined %}"
        "{% set add_generation_prompt = false %}{% endif %}"
        "{% for message in messages %}"
        "{{bos_token + message['from'] + '\n' + message['value'] + eos_token + '\n'}}"
        "{% endfor %}{% if add_generation_prompt %}{{ bos_token + 'assistant\n' }}{% endif %}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        pad_token='<unk>', # change for non llama/mistral tokenizers
        add_bos_token=False,
        chat_template=chat_template,
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    
formatting_func = create_chatml_fromvalue(tokenizer) if CHATML else create_alpaca_prompt_with_response


trainer = SFTTrainer(
    model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    formatting_func=formatting_func,
    max_seq_length=config.max_seq_length,
    packing=True,
    args=training_args,
)

trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

if config.save and accelerator.is_main_process:
    logging.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)

if config.eval:
    logging.info("Evaluating model")
    trainer.evaluate()

if config.log_model and config.save and accelerator.is_main_process:
    logging.info("Saving model as artifact to wandb")
    model_name = config.model_id.split("/")[-1].replace("-", "_")
    model_at = wandb.Artifact(
        name = f"model_{config.n_layers}_layers-{wandb.run.id}", 
        type="model",
        description="Model trained on Alpaca GPT4 dataset",
        metadata=config.to_dict())
    model_at.add_dir(training_args.output_dir)
    wandb.log_artifact(model_at)
