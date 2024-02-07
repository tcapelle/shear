def _prompt_no_input(row):
    return ("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n").format_map(row)

def _prompt_input(row):
    return ("Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n").format_map(row)

def create_alpaca_prompt(row):
    return _prompt_no_input(row) if row["input"] == "" else _prompt_input(row)

def create_alpaca_prompt_with_response(row):
    instruct = _prompt_no_input(row) if row["input"] == "" else _prompt_input(row)
    return instruct + row["output"]

def create_chatml_fromvalue(tokenizer, key='conversations'):
    # expects row['conversations'] is of the form [ {'from': '...', 'value': '...'}, ... ]
    def apply_template(x):
        return tokenizer.apply_chat_template(x[key], add_generation_prompt=False, tokenize=False)
    return apply_template

def basic_text_format(tokenizer, key='text'):
    # instead of depending on config.add_bos_token etc..,
    # ensures special tokens are properly set
    return lambda x: f'{tokenizer.bos_token}{x[key]}{tokenizer.eos_token}'
