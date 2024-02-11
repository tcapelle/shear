import wandb
import logging
from dataclasses import dataclass
import simple_parsing

import torch
import torch.nn as nn

from accelerate import Accelerator
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer, Trainer, default_data_collator
from transformers.integrations import WandbCallback
from torch.utils.data import DataLoader, SequentialSampler

logging.basicConfig(level=logging.INFO)

WANDB_PROJECT = "memory_footprint"
WANDB_ENTITY = None

# MODEL_ID = "mistralai/Mistral-7B-v0.1"
MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-480k-1T"
DATASET_NAME = "roneneldan/TinyStories"


class MemoryCallback(WandbCallback):
    def __init__(self):
        super().__init__()

    def on_step_end(self, args, state, control, **kwargs):
        memory_stats = torch.cuda.memory_stats()
        wandb.log(memory_stats)
        wandb.log({"max_memory_allocated [GB]": memory_stats["allocated_bytes.all.peak"]/1e9})

@dataclass
class Config(simple_parsing.Serializable):
    model_id: str = MODEL_ID
    dataset: str = DATASET_NAME
    torch_compile: bool = False
    batch_size: int = 1
    learning_rate: float = 1e-4
    optim: str = "adamw_torch"
    gradient_accumulation_steps: int = 1 
    min_seq_length: int = 5
    max_steps: int = -1
    num_train_epochs: int = 1
    use_flash_attention_2: bool = True
    gradient_checkpointing: bool = False
    name: str = None


config: Config = simple_parsing.parse(Config)

accelerator = Accelerator()

if accelerator.is_main_process:
    wandb.init(project=WANDB_PROJECT, 
               entity=WANDB_ENTITY, 
               name=config.name,
               job_type="train", 
               config={"init_args": config})

tokenizer = AutoTokenizer.from_pretrained(config.model_id)

# create a dummy dataset
with open("random_text.txt", "r") as f:
    txt = f.read()

tok_example = tokenizer(txt)
logging.info(f"Len of text file: {len(tok_example['input_ids'])}")

ds = [{"input_ids": tok_example["input_ids"][:counter],
       "labels": tok_example["input_ids"][:counter],
       "attention_mask": tok_example["attention_mask"][:counter]} for counter in range(config.min_seq_length, 10_000)]

training_args = TrainingArguments(
    output_dir="./outputs",
    report_to="wandb",
    per_device_train_batch_size=config.batch_size,
    per_device_eval_batch_size=config.batch_size,
    bf16=True,
    torch_compile=config.torch_compile,
    gradient_checkpointing = config.gradient_checkpointing,
    learning_rate=config.learning_rate,
    lr_scheduler_type="cosine",
    optim=config.optim,
    warmup_ratio=0.1,
    max_steps=config.max_steps,
    num_train_epochs=config.num_train_epochs,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    evaluation_strategy="no",
    # logging strategies
    logging_dir=".outputs/logs",
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
        use_flash_attention_2=config.use_flash_attention_2,
        )


logging.info(f"Model loaded from {config.model_id}")
logging.info(f"Model Parameters: {model.num_parameters() / 1e9:.2f}B")

mem_cb = MemoryCallback()

class CustomTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        # If your dataset is a `torch.utils.data.Dataset`
        train_sampler = SequentialSampler(self.train_dataset)
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        ) 

def collate_fn(examples):
    token_len = len(examples[0]["input_ids"])
    wandb.log({"token_len": token_len})
    logging.info(f"Token len: {token_len}")
    return default_data_collator(examples)

trainer = CustomTrainer(
    model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=ds,
    data_collator=collate_fn,
    callbacks=[mem_cb],
    )


trainer.train()

