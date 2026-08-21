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

# install RUST
export RUSTUP_DIST_SERVER="https://rsproxy.cn"
export RUSTUP_UPDATE_ROOT="https://rsproxy.cn/rustup"

curl -f https://rsproxy.cn/rustup/dist/x86_64-unknown-linux-gnu/rustup-init -o rustup-init
chmod +x rustup-init
./rustup-init -y
source "$HOME/.cargo/env"
cargo --version      # 能打印版本号就成了

# configure cargo to use the rsproxy mirror for crate downloads
cat > ~/.cargo/config.toml <<'EOF'
[source.crates-io]
replace-with = "rsproxy-sparse"

[source.rsproxy-sparse]
registry = "sparse+https://rsproxy.cn/index/"

[net]
git-fetch-with-cli = true
EOF

uv pip install vllm==0.26.0

# from source
# change [[tool.uv.index]] from default to "https://pypi.tuna.tsinghua.edu.cn/simple"?
# MiniMax H3 support ships in vLLM-Omni, not the vllm wheel, so install it from a checkout.
# NOTE: do NOT add the [fa4] extra on H800. FA4 (CuTe-DSL FlashAttention-4) is
# Blackwell-only; H800 (Hopper, SM90) uses the FLASH_ATTN backend, which selects
# FA2/FA3 kernels that ship with the base install. Adding [fa4] would pull
# Blackwell-only wheels that cannot run on H800.
uv pip install -e .

uv pip install git+https://github.com/thu-ml/SageAttention.git --no-build-isolation

apt-get install -y ffmpeg