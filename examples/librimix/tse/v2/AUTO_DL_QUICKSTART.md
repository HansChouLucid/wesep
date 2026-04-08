# WESEP TSE v2 on AUTO DL

This note is a minimal, single-GPU path for first-time setup on AUTO DL.
It keeps the original recipe unchanged and uses separate config files:
`confs/bsrnn_autodl_single_gpu.yaml` and `confs/spexplus_autodl_single_gpu.yaml`.

## 1. Recommended machine

- GPU: 24 GB or above is recommended, such as 3090, 4090, A5000, A40, V100-32G.
- Image: choose a PyTorch image with CUDA 11.3+ and Python 3.9.
- Disk: 150 GB or above is safer if you will keep Libri2Mix, shard data, and checkpoints on the same instance.

## 2. Create the environment

```bash
conda create -n wesep python=3.9 -y
conda activate wesep
conda install pytorch=1.12.1 torchaudio=0.12.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y
cd /root/wesep
pip install -r requirements.txt
pip install -e .
```

## 3. Upload code and data

Put the repo on the instance, for example:

```bash
cd /root
git clone <your-wesep-repo-url> wesep
```

Put Libri2Mix on the instance and make sure the folder layout matches:

```bash
/root/autodl-tmp/Libri2Mix/wav16k/min/train-100
/root/autodl-tmp/Libri2Mix/wav16k/min/dev
/root/autodl-tmp/Libri2Mix/wav16k/min/test
```

Then set:

```bash
export LIBRI2MIX_DIR=/root/autodl-tmp/Libri2Mix
```

## 4. Use tmux before training

```bash
tmux new -s wesep
conda activate wesep
cd /root/wesep/examples/librimix/tse/v2
```

## 5. Run stage by stage

Prepare metadata:

```bash
bash run.sh --stage 1 --stop_stage 1
```

Build shards:

```bash
bash run.sh --stage 2 --stop_stage 2
```

Start BSRNN training on one GPU:

```bash
bash run.sh --stage 3 --stop_stage 3 --gpus "[0]" --config confs/bsrnn_autodl_single_gpu.yaml
```

Start SpEx+ training on one GPU:

```bash
bash run.sh --stage 3 --stop_stage 3 --gpus "[0]" --config confs/spexplus_autodl_single_gpu.yaml
```

Average checkpoints:

```bash
bash run.sh --stage 4 --stop_stage 4 --gpus "[0]" --config confs/spexplus_autodl_single_gpu.yaml
```

Inference:

```bash
bash run.sh --stage 5 --stop_stage 5 --gpus "[0]" --config confs/spexplus_autodl_single_gpu.yaml
```

Scoring:

```bash
bash run.sh --stage 6 --stop_stage 6 --gpus "[0]" --config confs/spexplus_autodl_single_gpu.yaml
```

## 6. First things to check if training fails

- If you get CUDA OOM with BSRNN, lower `batch_size` from `2` to `1` in `confs/bsrnn_autodl_single_gpu.yaml`.
- If you get CUDA OOM with SpEx+, lower `batch_size` from `4` to `2`, then `1`, in `confs/spexplus_autodl_single_gpu.yaml`.
- If data preparation fails, check `LIBRI2MIX_DIR` and whether `wav16k/min` exists.
- If dataloader workers are unstable, reduce `num_workers` from `4` to `2`.
- If stage 6 fails because PESQ or DNSMOS is not available, you can still keep the model and skip scoring first.

## 7. Useful checks

Check GPU:

```bash
nvidia-smi
```

Check training log:

```bash
tail -f exp/SpExplus_autodl_single_gpu/train.log
```

Check checkpoints:

```bash
ls exp/SpExplus_autodl_single_gpu/models
```
