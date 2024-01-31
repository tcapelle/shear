from types import SimpleNamespace
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import parse_args


config = SimpleNamespace(
    model_id = "mistralai/Mistral-7B-v0.1",
    output_name = "models/mistral_7b_12_layers_start",
    n_layers=12,
    save_tokenizer=True,
)

parse_args(config)

model = AutoModelForCausalLM.from_pretrained(config.model_id)
model.model.layers = model.model.layers[:config.n_layers]
model.save_pretrained(config.output_name)
print(f"Saved model to {config.output_name}")
if config.save_tokenizer:
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.save_pretrained(config.output_name)
