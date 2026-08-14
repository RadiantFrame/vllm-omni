#!/bin/bash

# ######################################## SET CUDA ENV ########################################
echo "=== CUDA environment set ==="

CUDA_DIR=/usr/local/cuda-13.0

export LD_LIBRARY_PATH=${CUDA_DIR}/lib64:${LD_LIBRARY_PATH}

export CUDA_HOME=${CUDA_DIR}

export PATH=${CUDA_DIR}/bin:${PATH}

echo $(nvcc -V)

# ######################################## CREATE UV ENV ########################################
echo "=== CREATE UV Environment ==="

# install uv environment
# curl -LsSf https://astral.sh/uv/install.sh | sh

# way 1: pro6000
#-------------------------------------------------------------------------------
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
# Do NOT add pypi.org as an extra index: uv's unsafe-best-match strategy queries
# every index per-package, so an extra pypi.org makes each resolve hit the
# (slow/blocked from CN) official PyPI and time out. Use aliyun only; install any
# truly-missing package from pypi manually instead of opening a global extra index.
# export UV_EXTRA_INDEX_URL=https://pypi.org/simple

# create a virtual environment
uv venv --python 3.12 --seed
# activate the virtual environment
source .venv/bin/activate

# install RUST (官方源，不走镜像)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
cargo --version      # 能打印版本号就成了

# cargo 直接使用 crates.io 官方源（无需镜像）
cat > ~/.cargo/config.toml <<'EOF'
[net]
git-fetch-with-cli = true
EOF

uv pip install vllm==0.26.0

# from source
# change [[tool.uv.index]] from default to "https://pypi.tuna.tsinghua.edu.cn/simple"?
# MiniMax H3 support ships in vLLM-Omni, not the vllm wheel, so install it from a checkout.
# NOTE: do NOT add the [fa4] extra on pro6000. FA4 (CuTe-DSL FlashAttention-4) is
# Blackwell-only; pro6000 (Hopper, SM90) uses the FLASH_ATTN backend, which selects
# FA2/FA3 kernels that ship with the base install. Adding [fa4] would pull
# Blackwell-only wheels that cannot run on pro6000.
uv pip install -e .

sudo apt-get install -y ffmpeg