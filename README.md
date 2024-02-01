# shear

Create the small starting model:

```bash
python create_small_model.py
```

train an alpaca baseline model

```bash
accelerate launch train.py
```

## Docker

Build:
```bash
docker build -t shear-dev .
```

You can run with docker:

```bash
docker run \
  --privileged \
  --gpus '"all"' \
  --shm-size=10g \
  --rm \
  -it \
  --name=shear-dev \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --mount type=bind,src="${PWD}",target=/workspace/shear \
  -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
  -e $WANDB_API_KEY \
  shear-dev
```