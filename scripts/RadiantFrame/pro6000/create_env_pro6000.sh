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

# way 1: H800
#-------------------------------------------------------------------------------
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
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

# with pip
# uv pip install "sglang[diffusion]" --prerelease=allow
# from source
uv pip install -e "python[diffusion]" --prerelease=allow
# # change [[tool.uv.index]] from default to "https://mirrors.aliyun.com/pypi/simple"

sudo apt-get install -y ffmpeg