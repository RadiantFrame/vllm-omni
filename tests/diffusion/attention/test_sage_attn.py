# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from typing import Any

import pytest
import torch

import vllm_omni.diffusion.attention.backends.sage_attn as sage_backend
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.sage_attn import SageAttentionBackend, SageAttentionImpl

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_sage_backend_declares_prefix_kv_slicing():
    """The mask-free contract: prefix padding is honored by slicing K/V."""
    assert SageAttentionBackend.supports_prefix_kv_slicing is True
    # Sage kernels still cannot consume a materialized attn_mask.
    assert SageAttentionBackend.supports_attention_mask() is False


def test_sage_slices_valid_kv_prefix_without_padding_mask(monkeypatch):
    observed: dict[str, Any] = {}

    def fake_sageattn(query, key, value, **kwargs):
        observed.update(query=query, key=key, value=value, kwargs=kwargs)
        return query

    monkeypatch.setattr(sage_backend, "sageattn", fake_sageattn)
    impl = SageAttentionImpl(num_heads=2, head_size=4, softmax_scale=0.5)
    query = torch.randn(1, 8, 2, 4)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    output = impl.forward_cuda(
        query,
        key,
        value,
        AttentionMetadata(extra={"valid_kv_length": 5}),
    )

    assert output.shape == query.shape
    assert observed["query"].shape == (1, 8, 2, 4)
    assert observed["key"].shape == (1, 5, 2, 4)
    assert observed["value"].shape == (1, 5, 2, 4)


def test_sage_skips_slicing_when_no_padding(monkeypatch):
    observed: dict[str, Any] = {}

    def fake_sageattn(query, key, value, **kwargs):
        observed.update(key=key)
        return query

    monkeypatch.setattr(sage_backend, "sageattn", fake_sageattn)
    impl = SageAttentionImpl(num_heads=2, head_size=4, softmax_scale=0.5)
    tensors = torch.randn(1, 8, 2, 4)

    impl.forward_cuda(tensors, tensors, tensors, AttentionMetadata(extra={"valid_kv_length": 8}))

    assert observed["key"] is tensors


def test_sage_ignores_valid_kv_length_on_non_packed_layout(monkeypatch):
    observed: dict[str, Any] = {}

    def fake_sageattn(query, key, value, **kwargs):
        observed.update(key=key)
        return query

    monkeypatch.setattr(sage_backend, "sageattn", fake_sageattn)
    impl = SageAttentionImpl(num_heads=2, head_size=4, softmax_scale=0.5)
    tensors = torch.randn(8, 2, 4)

    impl.forward_cuda(tensors, tensors, tensors, AttentionMetadata(extra={"valid_kv_length": 5}))

    assert observed["key"] is tensors


def test_sage_rejects_invalid_valid_kv_length(monkeypatch):
    monkeypatch.setattr(sage_backend, "sageattn", lambda q, k, v, **kwargs: q)
    impl = SageAttentionImpl(num_heads=2, head_size=4, softmax_scale=0.5)
    tensors = torch.randn(1, 8, 2, 4)

    with pytest.raises(ValueError, match="valid_kv_length"):
        impl.forward_cuda(tensors, tensors, tensors, AttentionMetadata(extra={"valid_kv_length": 9}))


def test_sage_rejects_attn_mask():
    impl = SageAttentionImpl(num_heads=2, head_size=4, softmax_scale=0.5)
    tensors = torch.randn(1, 8, 2, 4)

    with pytest.raises(ValueError, match="does not support attn_mask"):
        impl.forward_cuda(tensors, tensors, tensors, AttentionMetadata(attn_mask=torch.zeros(1, 8)))
