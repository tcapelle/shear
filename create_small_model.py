from dataclasses import dataclass
import simple_parsing
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

@dataclass
class Config(simple_parsing.Serializable):
    model_id: str = "mistralai/Mistral-7B-v0.1"
    output_name: str = "models/mistral_7b_12_layers_start"
    n_layers: int = 12
    save_tokenizer: bool = True
    device_map: str = "cuda:0"

config: Config = simple_parsing.parse(Config)

model_config = AutoConfig.from_pretrained(config.model_id)
model_config.num_hidden_layers = config.n_layers
print(model_config)

model = AutoModelForCausalLM.from_pretrained(
    config.model_id, 
    config=model_config, 
    torch_dtype='auto',
    device_map=config.device_map)

print(f"Total model Parameters: {(model.num_parameters()/1e6):.2f}M")

model.save_pretrained(config.output_name)
print(f"Saved model to {config.output_name}")
if config.save_tokenizer:
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.save_pretrained(config.output_name)
