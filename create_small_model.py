import wandb
from dataclasses import dataclass
import simple_parsing
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

import logging

logging.basicConfig(level=logging.INFO)

WANDB_PROJECT = "shearllama"
ENTITY = "capecape"

@dataclass
class Config(simple_parsing.Serializable):
    model_id: str = "mistralai/Mistral-7B-v0.1"
    output_name: str = "models/mistral_7b_12_layers_start"
    n_layers: int = 12
    save_tokenizer: bool = True
    device_map: str = "cuda:0"
    random: bool = False
    log: bool = True

config: Config = simple_parsing.parse(Config)

model_config = AutoConfig.from_pretrained(config.model_id)
model_config.num_hidden_layers = config.n_layers
logging.info(model_config)

if not config.random:
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, 
        config=model_config, 
        torch_dtype='auto',
        device_map=config.device_map)
else:
    model = AutoModelForCausalLM.from_config(model_config)
    model.init_weights()

logging.info(f"Total model Parameters: {(model.num_parameters()/1e6):.2f}M")

model.save_pretrained(config.output_name)
logging.info(f"Saved model to {config.output_name}")
if config.save_tokenizer:
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.save_pretrained(config.output_name)

if config.log:
    wandb.init(project=WANDB_PROJECT, 
               entity=ENTITY, 
               job_type="prune_model", 
               config=config.to_dict())
    logging.info("Saving model as artifact to wandb")
    model_at = wandb.Artifact(
        name = config.output_name.split("/")[-1], 
        type="model",
        description=f"Baseline pruned model with {config.n_layers} layers.",
        metadata=config.to_dict())
    model_at.add_dir(config.output_name)
    wandb.log_artifact(model_at)
