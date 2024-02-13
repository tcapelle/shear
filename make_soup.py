import torch
import re, json
import subprocess
from glob import glob
from tqdm.auto import tqdm
from pathlib import Path
from itertools import chain
from collections import defaultdict

from safetensors import safe_open
from safetensors.torch import save_file

from huggingface_hub import (
    CommitOperationAdd,
    preupload_lfs_files,
    create_commit,
    create_repo,
    snapshot_download,
)


NON_ID_LAYERS = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
INDEX_FILENAME = "model.safetensors.index.json"


def natural_sort(xs, key=lambda s: s):
    """
    Sort the list into natural alphanumeric order.
    """

    def get_alphanum_key_func(key):
        convert = lambda text: int(text) if text.isdigit() else text
        return lambda s: [convert(c) for c in re.split("([0-9]+)", key(s))]

    return sorted(list(xs), key=get_alphanum_key_func(key))


def make_st_filename(i, count):
    return f"model-{i:05d}-of-{count:05d}.safetensors"


def get_layer_index(x):
    # Note: label names are of the form: "model.layers.N.___.___" where N is layer number
    return None if x in NON_ID_LAYERS else int(x.split(".")[2])


def remap_layer_id(layer, idx):
    if layer in NON_ID_LAYERS:
        return layer

    name = layer.split('.')
    name[2] = str(idx)
    return '.'.join(name)


def dl_snapshot(model_id, out_dir, allow=None, ignore=None):
    # throws:
    # EnvironmentError if token=True and the token cannot be found.
    # OSError if ETag cannot be determined.
    # ValueError if some parameter value is invalid
    return snapshot_download(
        repo_id=model_id,
        allow_patterns=allow,
        ignore_patterns=ignore,
        local_dir_use_symlinks=False,
        local_dir=out_dir,
    )


def construct_layer_map(layer_ids, layer_index):
    layer_ids = ['lm_head.weight', 'model.embed_tokens.weight', *layer_ids, 'model.norm.weight']
    tmp = defaultdict(list)
    layer_map = []
    count = 0

    for k in natural_sort(list(layer_index.keys())):
        key = k if k in NON_ID_LAYERS else get_layer_index(k)
        if key not in layer_ids: continue
        tmp[key].append(k)

    for k in layer_ids:
        layer_map.append([(layer, remap_layer_id(layer, count)) for layer in tmp[k]])
        count += 1 if k not in NON_ID_LAYERS else 0

    layer_map = dict(chain.from_iterable(layer_map))
    sliced_index_map = {k: layer_index[k] for k in layer_map.keys()}

    return layer_map, sliced_index_map


def remap_filenames(slice_map, layer_map, index_path, out_dir):
    index_map = {
        v: make_st_filename(i + 1, len(slice_map)) for i, v in enumerate(slice_map.values())
    }
    # first one below separated incase curious to verify....
    remapped_index = {v: slice_map[k] for k, v in layer_map.items()}
    remapped_index = {k: index_map[v] for k, v in remapped_index.items()}

    return index_map, remapped_index


def update_config(out_dir, num_layers):
    config_path = Path(out_dir) / "config.json"
    temp_config = None

    with open(config_path, "r") as f:
        temp_config = json.load(f)

    with open(config_path, "w") as f:
        temp_config["num_hidden_layers"] = num_layers
        json.dump(temp_config, f, indent=2)

    return temp_config


def write_metadata(model_id, num_layers, layer_ids, maps, out_dir):
    layer_map, slice_map, index_map, fname_map = maps
    metadata = {
        'base_model': model_id, 
        'layer_ids': layer_ids,
        'layer_map': layer_map,
        'slice_map': slice_map,
        'index_map': index_map,
        'fname_map': fname_map,
    }

    with open(out_dir / '.metadata', 'w') as f:
        json.dump(metadata, f, indent=2)


def rename_files(index_map, remapped_index, index_path, out_dir):
    for old_filename, new_filename in index_map.items():
        (out_dir / old_filename).rename(out_dir / new_filename)

    with open(index_path, "w") as f:
        json.dump({"metadata": {}, "weight_map": remapped_index}, f, indent=2)


def get_index(index_path):
    index = None

    if not index_path.exists(): raise FileNotFoundError(f"{index_path}")

    with open(index_path, "r") as f:
        index = json.load(f)["weight_map"]

    return index


def build_model_soup(
    model_id: str = "typeof/mistral-7b",
    layer_ids=None,
    out_dir: str = ".",
) -> None:
    """
    Derive model soup from arbitrary layer_ids in any arbitrary order.
    Usage: 
    ```
    build_model_soup(layer_ids=[0,2,8,6,4], out_dir)
    ```
    """
    # TODO: Check each layer_id exists and num_layers does not exceed number of model layers

    layer_ids = layer_ids if layer_ids is not None else [*range(8), *range(28, 32)] 
    num_layers = len(layer_ids)

    out_dir = Path(out_dir)
    index_path = out_dir / INDEX_FILENAME

    if out_dir.exists(): raise FileExistsError(f'"{out_dir}" exsists. please provide new path.')
    out_dir.mkdir()

    allow, ignore = ["*.json", "*.model"], ["pytorch_model.*"]
    err = dl_snapshot(model_id, out_dir, allow=allow, ignore=ignore)

    index = get_index(index_path)
    layer_map, slice_map = construct_layer_map(layer_ids, index)

    err = dl_snapshot(model_id, out_dir, allow=list(slice_map.values()))
    index_map, remapped_index = remap_filenames(slice_map, layer_map, index_path, out_dir)

    rename_files(index_map, remapped_index, index_path, out_dir)
    updated_config = update_config(out_dir, num_layers)

    maps = [layer_map, slice_map, index_map, remapped_index]
    write_metadata(model_id, num_layers, layer_ids, maps, out_dir)

    return remapped_index
