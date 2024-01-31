import wandb
from types import SimpleNamespace
from datasets import load_dataset
from accelerate import Accelerator

import torch
import torch.nn as nn

from transformers import TrainingArguments, AutoModelForCausalLM, TrainerCallback
from transformers.models.llama import LlamaConfig, LlamaForCausalLM

from trl import SFTTrainer
from utils import parse_args
from data import create_alpaca_prompt_with_response


WANDB_PROJECT = "shearllama"
ENTITY = "capecape"
# DATASET_NAME = "togethercomputer/RedPajama-Data-1T-Sample"
DATASET_NAME = "vicgalle/alpaca-gpt4"
# MODEL_ID = "mistralai/Mistral-7B-v0.1"
MODEL_ID = "./mistral_7b_12_layers_start"  # only first 12 layers of Mistral 7B
LAST_CHECKPOINT = None

config = SimpleNamespace(
    resume_from_checkpoint=LAST_CHECKPOINT,  # uses LAST Checkpoint from W&B
    torch_compile=False,
    batch_size=2,
    learning_rate=1e-4,
    gradient_accumulation_steps=8,
    n_layers=12,
    test_size=0.05,
    seed=42,
    max_seq_length=1024,
    max_steps=-1,
    save=True,
    eval=True,
)

parse_args(config)

accelerator = Accelerator()

if accelerator.is_main_process:
    wandb.init(project=WANDB_PROJECT, 
               entity=ENTITY, 
               job_type="train", 
               config={"init_args": config})
    # estimation
    effective_batch_size = config.max_seq_length*config.batch_size*config.gradient_accumulation_steps*accelerator.num_processes
    print(f"\nTraining with an effective batch_size of: {effective_batch_size}\n")

ds = load_dataset(DATASET_NAME)["train"].shuffle(seed=config.seed)
if config.test_size>0:
    ds = ds.train_test_split(test_size=config.test_size)
    train_ds = ds["train"]
    test_ds = ds["test"]
else:
    train_ds = ds
    test_ds = None


output_dir = f"./output/mistral_7b_{config.n_layers}_layers"
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=config.batch_size,
    per_device_eval_batch_size=config.batch_size,
    bf16=True,
    gradient_checkpointing = True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    torch_compile=config.torch_compile,
    learning_rate=config.learning_rate,
    lr_scheduler_type="cosine",
    warmup_steps=0.1,
    max_steps=config.max_steps,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    evaluation_strategy="no",
    # logging strategies
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="no",
)

model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        use_flash_attention_2=True,
        )

# Chop the model down to n_layers
model.model.layers = model.model.layers[:config.n_layers]
# freeze(model, freeze_embed=True, n_freeze=config.n_freeze, module_name="layers")


trainer = SFTTrainer(
    model,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    formatting_func=create_alpaca_prompt_with_response,
    max_seq_length=config.max_seq_length,
    packing=True,
    args=training_args,
)

trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

if config.eval:
    trainer.evaluate()
if config.save:
    trainer.save_model(training_args.output_dir)
