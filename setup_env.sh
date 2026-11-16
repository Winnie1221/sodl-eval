#!/bin/bash
# One-time environment setup. Run in a CPU JupyterLab terminal on Cloud Bursting (no GPU needed).
#   bash setup_env.sh
set -euo pipefail

OVERLAY="$SCRATCH/containers/overlay-25GB-500K.ext3"
IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif

mkdir -p "$SCRATCH/containers" "$SCRATCH/hf_cache"
if [ ! -f "$OVERLAY" ]; then
  cp /share/apps/overlay-fs-ext3/overlay-25GB-500K.ext3.gz "$SCRATCH/containers/"
  gunzip "$SCRATCH/containers/overlay-25GB-500K.ext3.gz"
fi

singularity exec --overlay "$OVERLAY":rw "$IMAGE" /bin/bash -c '
set -euo pipefail
export UV_INSTALL_DIR=/ext3/.uv UV_NO_MODIFY_PATH=1 UV_PYTHON_INSTALL_DIR=/ext3/.uv/python
export UV_CACHE_DIR=/ext3/.cache/uv UV_LINK_MODE=copy
curl --fail --location --silent --show-error https://astral.sh/uv/install.sh -o /ext3/install-uv.sh
sh /ext3/install-uv.sh
/ext3/.uv/uv venv /ext3/.venv --python 3.12
/ext3/.uv/uv pip install --python /ext3/.venv/bin/python vllm math-verify datasets pandas pyarrow matplotlib huggingface_hub
/ext3/.venv/bin/python -c "import vllm, math_verify, datasets; print(\"SODL_ENV_OK vllm\", vllm.__version__)"
'
