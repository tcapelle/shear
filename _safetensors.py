import torch
import re, json
import subprocess
from glob import glob
from tqdm.auto import tqdm
from pathlib import Path
from itertools import chain

from safetensors import safe_open
from safetensors.torch import save_file

from huggingface_hub import (
    CommitOperationAdd,
    preupload_lfs_files,
    create_commit,
    create_repo,
    snapshot_download
)


def natural_sort(xs, key=lambda s:s):
    """
    Sort the list into natural alphanumeric order.
    """
    def get_alphanum_key_func(key):
        convert = lambda text: int(text) if text.isdigit() else text
        return lambda s: [convert(c) for c in re.split('([0-9]+)', key(s))]

    return sorted(list(xs), key=get_alphanum_key_func(key))


def make_st_filename(i, count):
    return f'model-{i:05d}-of-{count:05d}.safetensors'


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

    # determine # of tensors and key names
    for tensor in tensors:
        with safe_open(tensor, framework="pt", device=device) as f:
            count += len(f.keys())
            order.append(f.keys())

    order = {
        k: make_st_filename(i+1, count)
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


def from_index(index_filename: str='model.safetensors.index.json'):
    index = json.load(open(index_filename, 'r'))
    wm = index.get('weight_map')

    in_map = {k: wm.get(k) for k in natural_sort(wm.keys())}
    out_map = {k: make_st_filename(i+1, len(in_map)) for i, k in enumerate(in_map.keys())}

    return in_map, out_map


def dl_file_from_repo(repo_id: str, repo_file: str, base_dir: str='.'):
    name = repo_id.split('/')[-1]
    path = Path(base_dir)
    if not path.exists():
        path.mkdir()
    return subprocess.getstatusoutput(
        f'huggingface-cli download {repo_id} {repo_file} --local-dir {path} '
        '--local-dir-use-symlinks False'
    )


def split_repo(from_repo: str, to_repo: str):
    """
    usage: split_repo('152334H/miqu-1-70b-sf', 'typeof/miqu-70b-split')
    will split from_repo and place into to_repo.
    assumes `from_repo` is rather large so processes accordingly...
    """

    # this can take an absurd amount of time? naive? latency? idk...

    base_dir = Path(from_repo.split('/')[-1])
    out_dir = Path(to_repo.split('/')[-1])
    index_name = 'model.safetensors.index.json'

    clone_res = subprocess.getstatusoutput(
        f'GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/{to_repo}'
    )

    if not base_dir.exists(): base_dir.mkdir()

    snapshot_download(
        repo_id=from_repo,
        allow_patterns=["*.json", "*.model"],  # maybe add *.bin ?
        ignore_patterns=["pytorch_model.*"],
        local_dir_use_symlinks=False,
        local_dir=out_dir,
    )

    in_map, out_map = from_index(out_dir / index_name)
    with open(out_dir / index_name, 'w') as f:
        json.dump({'metadata': '', 'weight_map': out_map}, f, indent=2)

    for tensor_filename in tqdm(natural_sort(set(in_map.values()))):

        dl_file_from_repo(from_repo, tensor_filename, base_dir)

        path_to_tensor = base_dir / tensor_filename

        with safe_open(path_to_tensor, framework="pt", device='cpu') as f:
            for x in tqdm(list(f.keys())):
                res = {x: f.get_tensor(x).to(torch.bfloat16)}
                save_file(res, out_dir / out_map[x], metadata={'format': 'pt'})

        status = subprocess.getstatusoutput(
            f'cd {out_dir} && huggingface-cli lfs-enable-largefiles . && '
            'git add . && git commit -m "init" && git push'
        )

        path_to_tensor.unlink(missing_ok=False)
        shutil.rmtree(out_dir)
        clone_res = subprocess.getstatusoutput(
            f'GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/{to_repo}'
        )

        # check `clone_res[0]` == 0 ... ie. no errors?


def build_model_from(model_id: str='typeof/mistral-7b-sharded', num_layers: int=16, out_dir: str='.') -> None:
    """
    Derive model from first num_layers. ... exercise caution.
    """
    # The true aim of this function is to be efficient with space client side
    #
    # TODO: add meaningful error handling? tOooooOOooOoOoo many failure modes...
    #
    # Assumptions:
    # provided model is composed of maximally split safetensors...
    # provided out_dir already exists?
    #
    # NOTE: not all proposed configurations are canonically valid...

    index = None
    keep = ['model.embed_tokens.weight', 'model.norm.weight', 'lm_head.weight']
    index_filename = 'model.safetensors.index.json'

    out_dir = Path(out_dir)
    index_path = out_dir / index_filename

    # fetch saftensor index and other config files
    # TODO: handle failure modes!? ie. resource not available, no internet, etc...

    snapshot_download(
        repo_id=model_id,
        allow_patterns=["*.json", "*.model"],  # maybe add *.bin ?
        ignore_patterns=["pytorch_model.*"],
        local_dir_use_symlinks=False,
        local_dir=out_dir,
    )

    with open(index_path, 'r') as f:
        index = json.load(f)['weight_map']

    # Note: label names are of the form: "model.layers.N.___.___" where N is layer number
    sliced_index = {
        k: v for k, v in index.items()
        if (k.startswith('model.layers.') and int(k.split('.')[2]) < num_layers) or k in keep  # yuck
    }

    # * Now that we know the mapping of which layers resolve to which files...

    snapshot_download(
        repo_id=model_id,
        allow_patterns=list(sliced_index.values()),
        local_dir_use_symlinks=False,
        local_dir=out_dir,
    )

    # * Remap filenames...

    index_map = {
        v: make_st_filename(i+1, len(sliced_index))
        for i, v in enumerate(sliced_index.values())
    }

    for tensor_filename in sliced_index.values():
        (out_dir / tensor_filename).rename(out_dir / index_map[tensor_filename])

    remapped_index = {k: index_map[v] for k, v in sliced_index.items()}
    with open(index_path, 'w') as f:
        json.dump(remapped_index, f, indent=2)

    config_path = out_dir / 'config.json'
    temp_config = json.load(open(config_path, 'r'))
    temp_config['num_hidden_layers'] = num_layers
    json.dump(temp_config, open(config_path, 'w'), indent=2)

    # for the un-initiated, exceptions will bubble so...
    # return something? failure? remapped_index? ... idk? why?
    # *... we need to write unit tests :(


def push_model_to_hub(model_id: str, pathname: str, commit_message: str='init', private: bool=True, preupload: bool=False) -> None:
    # consider using run_as_future and return futures?
    # at least parallelize... can be quite time consuming...

    repo_id = create_repo(model_id, exist_ok=True, private=private).repo_id
    operations = []
    for filename in tqdm(list(glob(f'{pathname}/*'))):
        addition = CommitOperationAdd(path_in_repo=Path(filename).name, path_or_fileobj=filename)
        if preupload: preupload_lfs_files(repo_id, additions=[addition])  # upload + free memory
        operations.append(addition)

    create_commit(repo_id, operations=operations, commit_message=commit_message)
