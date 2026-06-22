"""
IREEAttentionBackend — vLLM attention plugin for IREE execution.

Phase 1: PyTorch SDPA with Flash Attention (prefill) + paged KV gather (decode).
Phase 2: replace forward() body with IREE-dispatched attention kernel.

Registration: @register_backend(AttentionBackendEnum.CUSTOM) stores
  the fully-qualified class path in _ATTN_OVERRIDES so vLLM can
  import it when IREEPlatform.get_attn_backend_cls() returns our string.
"""

from dataclasses import dataclass
import torch
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum,
    register_backend,
)
from vllm.v1.kv_cache_interface import AttentionSpec


@dataclass
class IREEAttentionMetadata:
    """Per-step attention metadata passed to IREEAttentionBackendImpl.forward."""
    num_actual_tokens: int = 0
    seq_lens: torch.Tensor = None       # [num_reqs] on GPU
    slot_mapping: torch.Tensor = None   # [num_actual_tokens] flat slot indices
    block_tables: torch.Tensor = None   # [num_reqs, max_blocks_per_req]
    is_prefill: bool = True


@register_backend(AttentionBackendEnum.CUSTOM)
class IREEAttentionBackend(AttentionBackend):
    """
    Registered under AttentionBackendEnum.CUSTOM.
    IREEPlatform.get_attn_backend_cls() returns the dotted path to this class,
    which vLLM resolves and instantiates per attention layer.
    """

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["IREEAttentionBackendImpl"]:
        return IREEAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> type["IREEAttentionMetadataBuilder"]:
        return IREEAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_type: str = "",
        cache_dtype_str: str = "",
    ) -> tuple[int, ...]:
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: list[torch.Tensor],
        dst_kv_cache: list[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        src_idx, dst_idx = src_to_dst[:, 0], src_to_dst[:, 1]
        dst_kv_cache[0][dst_idx] = src_kv_cache[0][src_idx]
        dst_kv_cache[1][dst_idx] = src_kv_cache[1][src_idx]

    @staticmethod
    def copy_blocks(
        kv_caches: list[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        src_idx, dst_idx = src_to_dists[:, 0], src_to_dists[:, 1]
        for kv_cache in kv_caches:
            kv_cache[0][dst_idx] = kv_cache[0][src_idx]
            kv_cache[1][dst_idx] = kv_cache[1][src_idx]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [16]


class IREEAttentionMetadataBuilder(
    AttentionMetadataBuilder[IREEAttentionMetadata]
):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.vllm_config = vllm_config
        self.device = device

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> IREEAttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        seq_lens_gpu = common_attn_metadata.seq_lens[:num_reqs]
        is_prefill = (num_actual_tokens > num_reqs)
        return IREEAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            seq_lens=seq_lens_gpu,
            slot_mapping=common_attn_metadata.slot_mapping[:num_actual_tokens],
            block_tables=common_attn_metadata.block_table_tensor,
            is_prefill=is_prefill,
        )


class IREEAttentionBackendImpl(AttentionImpl):
    """
    Per-layer attention implementation.

    Prefill: PyTorch Flash SDPA — causal attention over full prompt.
    Decode:  Paged KV gather + single-token SDPA per request.
    Phase 2: replace with IREE kernel dispatch.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = "decoder",
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = sliding_window
        self.attn_type = attn_type
        self.key_cache: torch.Tensor | None = None
        self.value_cache: torch.Tensor | None = None

    def _bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        if self.key_cache is None:
            self.key_cache = kv_cache[0]  # [num_blocks, block_size, nkv, d]
            self.value_cache = kv_cache[1]

    def _write_kv_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Scatter new K/V tokens into the paged KV cache."""
        key_headed = key.view(-1, self.num_kv_heads, self.head_size)
        val_headed = value.view(-1, self.num_kv_heads, self.head_size)
        flat_k = self.key_cache.view(-1, self.num_kv_heads, self.head_size)
        flat_v = self.value_cache.view(-1, self.num_kv_heads, self.head_size)
        flat_k[slot_mapping] = key_headed
        flat_v[slot_mapping] = val_headed

    def _gather_kv_for_decode(
        self,
        req_idx: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gather full K/V sequence for one decode request from paged cache.
        Returns k, v of shape [seq_len, num_kv_heads, head_size].
        """
        block_size = self.key_cache.shape[1]
        num_blocks = (seq_len + block_size - 1) // block_size
        blocks = self.block_tables[req_idx, :num_blocks]  # [num_blocks]
        # Gather blocks: [num_blocks, block_size, nkv, d]
        k_blocks = self.key_cache[blocks]
        v_blocks = self.value_cache[blocks]
        # Flatten and trim to seq_len
        k = k_blocks.reshape(-1, self.num_kv_heads, self.head_size)[:seq_len]
        v = v_blocks.reshape(-1, self.num_kv_heads, self.head_size)[:seq_len]
        return k, v

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,           # [num_tokens, num_heads * head_size]
        key: torch.Tensor,             # [num_tokens, num_kv_heads * head_size]
        value: torch.Tensor,           # [num_tokens, num_kv_heads * head_size]
        kv_cache: torch.Tensor,        # [2, num_blocks, block_size, nkv, d]
        attn_metadata: IREEAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:

        n = query.shape[0]
        n_actual = attn_metadata.num_actual_tokens if attn_metadata is not None else n

        if output is None:
            output = torch.zeros(
                n, self.num_heads * self.head_size,
                dtype=query.dtype, device=query.device,
            )

        # Bind KV cache and write new tokens
        self._bind_kv_cache(kv_cache)
        if key is not None and value is not None and attn_metadata is not None:
            self._write_kv_cache(
                key[:n_actual], value[:n_actual],
                attn_metadata.slot_mapping,
            )

        if attn_metadata is None or attn_metadata.is_prefill:
            # ── Prefill: full causal attention over all prompt tokens ──────
            q = query[:n_actual].view(n_actual, self.num_heads, self.head_size)
            k = key[:n_actual].view(n_actual, self.num_kv_heads, self.head_size)
            v = value[:n_actual].view(n_actual, self.num_kv_heads, self.head_size)

            if self.num_heads != self.num_kv_heads:
                groups = self.num_heads // self.num_kv_heads
                k = k.repeat_interleave(groups, dim=1)
                v = v.repeat_interleave(groups, dim=1)

            q_s = q.unsqueeze(0).transpose(1, 2)   # [1, H, n, d]
            k_s = k.unsqueeze(0).transpose(1, 2)
            v_s = v.unsqueeze(0).transpose(1, 2)

            # Use Flash Attention when available (sm_80+), falls back to math
            with torch.nn.attention.sdpa_kernel([
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                torch.nn.attention.SDPBackend.MATH,
            ]):
                attn_out = F.scaled_dot_product_attention(
                    q_s, k_s, v_s,
                    scale=self.scale,
                    is_causal=True,
                )

            output[:n_actual] = (
                attn_out.squeeze(0)
                .transpose(0, 1)
                .reshape(n_actual, self.num_heads * self.head_size)
            )

        else:
            # ── Decode: one new token per request, attends to cached KV ───
            # Store block_tables for _gather_kv_for_decode
            self.block_tables = attn_metadata.block_tables

            num_reqs = attn_metadata.seq_lens.shape[0]
            token_offset = 0

            for req_idx in range(num_reqs):
                seq_len = int(attn_metadata.seq_lens[req_idx].item())

                # Query for this request: [1, num_heads, head_size]
                q = query[token_offset].view(self.num_heads, self.head_size)
                q_s = q.unsqueeze(0).unsqueeze(0).transpose(1, 2)
                # [1, H, 1, d]

                # Gather full K/V from paged cache
                k_full, v_full = self._gather_kv_for_decode(req_idx, seq_len)
                # k_full: [seq_len, nkv, d]

                if self.num_heads != self.num_kv_heads:
                    groups = self.num_heads // self.num_kv_heads
                    k_full = k_full.repeat_interleave(groups, dim=1)
                    v_full = v_full.repeat_interleave(groups, dim=1)

                k_s = k_full.unsqueeze(0).transpose(1, 2)  # [1, H, seq, d]
                v_s = v_full.unsqueeze(0).transpose(1, 2)

                # No causal mask needed — query attends to all cached tokens
                attn_out = F.scaled_dot_product_attention(
                    q_s, k_s, v_s,
                    scale=self.scale,
                    is_causal=False,
                )
                # attn_out: [1, H, 1, d] → [H*d]
                output[token_offset] = (
                    attn_out.squeeze(0).squeeze(1)
                    .reshape(self.num_heads * self.head_size)
                )
                token_offset += 1

        return output