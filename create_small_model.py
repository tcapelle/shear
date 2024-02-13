import wandb
from dataclasses import dataclass, field
import simple_parsing
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

import torch
import logging

from utils import map_state_dict

logging.basicConfig(level=logging.INFO)

WANDB_PROJECT = "shearllama"
ENTITY = "llm_surgery"

@dataclass
class Config(simple_parsing.Serializable):
    model_id: str = "mistralai/Mistral-7B-v0.1"
    output_name: str = "models/mistral_7b_12_layers_start"
    layer_ids: list[int] = field(default_factory=lambda: [0,1,2,3,4,5,6,7])
    save_tokenizer: bool = True
    device_map: str = "cuda:0"
    bfloat16: bool = True
    random: bool = False
    log: bool = True

config: Config = simple_parsing.parse(Config)

logging.info(config)

model_config = AutoConfig.from_pretrained(config.model_id)
model_config.num_hidden_layers = len(config.layer_ids)
logging.info(model_config)

original_model = AutoModelForCausalLM.from_pretrained(
    config.model_id, 
    torch_dtype=torch.bfloat16,
    device_map=config.device_map)

new_model = AutoModelForCausalLM.from_config(model_config)

if config.random:
    new_model.init_weights()
else:
    name_mapping = map_state_dict(original_model, config.layer_ids)
    # Manually copy weights and biases
    for old_name, new_name in name_mapping.items():
        # Check if the mapped name exists in the new model's state_dict
        if new_name in new_model.state_dict():
            # Directly load the parameter from the old model to the new model based on the mapping
            new_model.state_dict()[new_name].data.copy_(original_model.state_dict()[old_name].data)
        else:
            print(f"{new_name} not found in the new model's state_dict. Check your mapping dictionary.")



logging.info(f"Total model Parameters: {(new_model.num_parameters()/1e6):.2f}M")

if config.bfloat16:
    new_model.to(torch.bfloat16)

new_model.save_pretrained(config.output_name)
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
        description=f"Baseline pruned model with {config.layer_ids} layers.",
        metadata=config.to_dict())
    model_at.add_dir(config.output_name)
    wandb.log_artifact(model_at)
