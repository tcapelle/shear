from dataclasses import dataclass
import simple_parsing
from transformers import AutoModelForCausalLM, AutoTokenizer

@dataclass
class Config(simple_parsing.Serializable):
    model_id: str = "mistralai/Mistral-7B-v0.1"
    output_name: str = "models/mistral_7b_12_layers_start"
    n_layers: int = 12
    save_tokenizer: bool = True

config: Config = simple_parsing.parse(Config)

model = AutoModelForCausalLM.from_pretrained(config.model_id)
model.model.layers = model.model.layers[:config.n_layers]
model.save_pretrained(config.output_name)
print(f"Saved model to {config.output_name}")
if config.save_tokenizer:
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.save_pretrained(config.output_name)
