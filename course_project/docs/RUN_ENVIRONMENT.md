# Formal Experiment Runtime Configuration

## Local status on 2026-07-10

The desktop machine has an RTX 4060 Laptop GPU with 8 GB VRAM, but the available
Conda environments (`bigdata` and `pack`) contain CPU-only PyTorch. The
`bigdata` environment also lacks the repository dependency `nltk`, confirming
that it was not provisioned from `requirements.txt`. The local Hugging Face
cache and the repository do not contain the `facebook/opt-2.7b` checkpoint.
Therefore this machine must not be used for the OPT-2.7B FP32 formal generation
or model-assisted copy-paste scoring.

`run_opt27b_pareto_500_fp32.sh` loads the model in FP32. OPT-2.7B weights alone
need about 10.8 GB, so use a remote NVIDIA GPU with at least 16 GB VRAM; 24 GB
or more is recommended for stable generation, PPL scoring, and detector passes.

## Remote environment

```bash
conda create -n watermark python=3.10 -y
conda activate watermark
pip install -r requirements.txt
# Install a CUDA-enabled PyTorch build matching the remote CUDA driver.
# Use the command from https://pytorch.org/get-started/locally/.
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import transformers; print(transformers.__version__)"
```

The last two commands must report `True` for CUDA before starting an experiment.

## Verified h102 deployment

`h102` provides RTX 6000D GPUs with compute capability `sm_120` and a CUDA
13.3 driver. The server's Python 3.12 installation has neither `venv` nor
`pip` enabled, so the verified user-scoped setup is:

```bash
python3 /home/zyb/get-pip.py --user --break-system-packages
/home/zyb/.local/bin/pip3 install --user --break-system-packages virtualenv
cd /home/zyb/inform
/home/zyb/.local/bin/virtualenv .venv
.venv/bin/python -m pip install --force-reinstall torch \
  --index-url https://download.pytorch.org/whl/cu130
.venv/bin/python -m pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -c \
  'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

Use the CUDA 13.0 (`cu130`) PyTorch wheel, not `cu126`: the latter lacks
`sm_120` kernels and fails with `cudaErrorNoKernelImageForDevice` on RTX 6000D.
The deployed model path is `/home/zyb/inform/models/opt-2.7b`.

For the legacy `c4/realnewslike` loading script used by the paper-style prompt
export, pin `datasets==2.19.2`. Newer `datasets 5.x` removes dataset-script
support and fails before the fixed prompt subset can be exported.

## Model layout

Set `MODEL_DIR` to a local Hugging Face checkpoint directory containing at least
`config.json` plus `pytorch_model.bin` or `model.safetensors`:

```text
information_security_homework/
  models/
    opt-2.7b/
      config.json
      pytorch_model.bin  # or model.safetensors
      tokenizer_config.json
      ...
```

For a remote cache, either download the checkpoint to this location or point the
scripts directly to its snapshot directory:

```bash
export MODEL_DIR=/path/to/models/opt-2.7b
export GPU_ID=0
export PYTHON_BIN=/path/to/conda/envs/watermark/bin/python
```

## Formal commands

```bash
bash course_project/scripts/run_opt27b_pareto_500_fp32.sh
bash course_project/scripts/run_copy_paste_attack.sh
```

The Pareto script creates new files under
`course_project/outputs/additional/pareto/raw/`, then computes PPL and
five-fold calibration. The copy-paste script reuses the stored generation CSVs
and supports restart with `--resume`; it writes to
`course_project/outputs/additional/copy_paste/`.

Do not overwrite the existing `opt27b_fp32_c4_500_*` files. To use a different
remote location, override `OUTPUT_DIR`, `MODEL_DIR`, `GPU_ID`, and `PYTHON_BIN`
as environment variables when running either shell script.
