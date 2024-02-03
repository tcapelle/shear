# shear

This repo is a set of experiments to pruned versions of 7B models. The goal is to create a small model that can be used for fine-tuning and deployment. 

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

## Pruning

Create the small starting model, this will load a full model, select an amount of `n_layers` and save the new model to W&B. This is important to do before training the model, as the model is too large to fit in memory.

For instance, we can create a 12 layer model mistral with the following command:

```bash
python create_small_model.py --model_id "mistralai/Mistral-7B-v0.1" --n_layers 12 --output_name "models/mistral_7b_12_layers_start" --log True
```

you can also create a randomly initialized model by passing the `--random` flag. Useful for baeline models.

## Training

To train a model, you can use the `train.py` script. This script will load a model from HF, folder or W&B artifact, and train it on Open Hermer or Alpaca.

For example, this will train a chopped 8 layer mistral model, passing the -n_layers flag dees nothing if the model already is a 8 layer model. We can pass some comma separated tags to help us filter the runs in W&B. For some reason the iterations hang if we go to the end of the dataset (809 steps).

> We are keeping the effective batch size (batch_size x n_gpus x gradient_accumulation_steps) equal to 64, as it is the batch size used in the original training of the model. Adjust accordingly if you are using a different batch size.

```bash
accelerate launch train.py --model_id ./models/mistral_7b_8_layers_start \
  --n_layers 8 \
  --tags=oh,8layers \
  --max_steps 800
```

