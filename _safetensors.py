import torch
import re, json, glob
from tqdm.auto import tqdm
from pathlib import Path
from itertools import chain

from safetensors import safe_open
from safetensors.torch import save_file

def natural_sort(xs, key=lambda s:s):
    """
    Sort the list into natural alphanumeric order.
    """
    def get_alphanum_key_func(key):
        convert = lambda text: int(text) if text.isdigit() else text
        return lambda s: [convert(c) for c in re.split('([0-9]+)', key(s))]

    return sorted(list(xs), key=get_alphanum_key_func(key))


def split_safetensors(dir_from, dir_to='layers', device='cuda', dtype=torch.bfloat16):
    """
    Split safetensor(s) file(s) into separate safetensor files per layer-id
    """
    # TODO: might be good practice to require directories to be unique?
    # assert(dir_from != dir_to)
    # TODO: handle safetensors not found... if not tensors: return ?

    count = 0
    order = []
    tensors = list(glob.iglob(f'{dir_from}/*.safetensors'))
    fmt = lambda x : f"{x:05d}" # hard-coded... might be internal to hf...


    # determine # of tensors and key names
    for tensor in tensors:
        with safe_open(tensor, framework="pt", device=device) as f:
            count += len(f.keys())
            order.append(f.keys())

    order = {
        k: f'model-{fmt(i+1)}-of-{fmt(count)}.safetensors' 
        for i, k in enumerate(natural_sort(chain.from_iterable(order)))
    }

    for tensor in tqdm(tensors):
        with safe_open(tensor, framework="pt", device=device) as f:
            for x in tqdm(list(f.keys())):
                res = {x: f.get_tensor(x).to(dtype)}
                save_file(res, Path(dir_to) / order[x], metadata={'format': 'pt'})

    with open(Path(dir_to) / 'model.safetensors.index.json', 'w') as f:
        json.dump({'metadata': {}, 'weight_map': order}, f, indent=2)

    return order
