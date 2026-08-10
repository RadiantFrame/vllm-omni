#!/bin/bash

# cuda environment settings
#------------------------------------------------------------------------
CUDA_DIR=/usr/local/cuda-13.0

export LD_LIBRARY_PATH=${CUDA_DIR}/lib64:${LD_LIBRARY_PATH}

export CUDA_HOME=${CUDA_DIR}

export PATH=${CUDA_DIR}/bin:${PATH}


# activate anaconda environment
#---------------------------------------------------------------------------
source .venv/bin/activate

# huggingface environment
#--------------------------------------------------------------------------
export HF_ENDPOINT=https://hf-mirror.com

# HF Token: pass via environment variable (e.g. export HF_TOKEN=your_token)
# or uncomment the line below with your actual token
# export HF_TOKEN=your_hf_token_here