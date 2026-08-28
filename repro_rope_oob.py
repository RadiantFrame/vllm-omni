#!/usr/bin/env python3
# Debug repro: _rms_norm_rope_kernel illegal memory access (ref2va 768p/15s, 0828).
# Recreates the exact input structure the MiniMax-H3 attention produces:
#   q/k are NON-CONTIGUOUS views of the fused qkv projection output
#   (row stride == full qkv width), bf16, on CUDA.
# Run:  CUDA_VISIBLE_DEVICES=6 compute-sanitizer --tool memcheck python repro_rope_oob.py

import torch

from vllm_omni.diffusion.layers.fused_qk_norm_rope import (
    _apply_rope_table,
    _launch_fused_rms_norm_rope,
)

HEAD_DIM = 128
ROTARY_DIM = 96
EPS = 1e-6


def run_case(tokens: int, num_heads: int, num_kv_heads: int) -> None:
    qkv_width = (num_heads + 2 * num_kv_heads) * HEAD_DIM
    qkv = torch.randn(tokens, qkv_width, dtype=torch.bfloat16, device="cuda")
    q_size = num_heads * HEAD_DIM
    kv_size = num_kv_heads * HEAD_DIM
    q, k, _ = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q = q.view(tokens, num_heads, HEAD_DIM)
    k = k.view(tokens, num_kv_heads, HEAD_DIM)

    weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    rope_table = torch.randn(tokens, ROTARY_DIM, dtype=torch.bfloat16, device="cuda")

    out_q = _launch_fused_rms_norm_rope(q, weight, rope_table, EPS)
    out_k = _launch_fused_rms_norm_rope(k, weight, rope_table, EPS)
    torch.cuda.synchronize()

    # Reference (eager) on a small slice for correctness spot-check.
    ref_q = _apply_rope_table(
        torch.nn.functional.rms_norm(q[:64].float(), (HEAD_DIM,), weight.float(), EPS),
        rope_table[:64].float(),
        ROTARY_DIM,
    ).to(torch.bfloat16)
    diff = (out_q[:64].float() - ref_q.float()).abs().max().item()
    print(f"tokens={tokens} heads={num_heads} kv={num_kv_heads} qkv_width={qkv_width} OK max_diff={diff:.4f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    # Per-rank shapes for TP2: 28 q heads; GQA kv sweep; token counts around
    # the real packed ref2va sequence (~32.8k after 64-alignment padding).
    for kv in (28, 14, 8, 4, 2):
        for tokens in (1024, 32768, 32832, 33024):
            run_case(tokens, 28, kv)
    # Single-GPU full-head shape as well.
    for tokens in (32832,):
        run_case(tokens, 56, 28)
    print("all cases completed")
