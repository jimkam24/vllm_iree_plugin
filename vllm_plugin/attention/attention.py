
"""
IREEAttentionBackend — vLLM attention plugin for IREE execution.
 
Phase 1: plain PyTorch SDPA (correct, not optimised).
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
    seq_lens: torch.Tensor = None       # [num_reqs] on CPU
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
        cache_dtype_str: str = "",   # add this
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)
    
    @staticmethod
    def swap_blocks(
        src_kv_cache: list[torch.Tensor],
        dst_kv_cache: list[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        """Move KV blocks between devices (e.g. GPU <-> CPU offload)."""
        src_idx, dst_idx = src_to_dst[:, 0], src_to_dst[:, 1]
        dst_kv_cache[0][dst_idx] = src_kv_cache[0][src_idx]
        dst_kv_cache[1][dst_idx] = src_kv_cache[1][src_idx]
 
    @staticmethod
    def copy_blocks(
        kv_caches: list[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        """Copy KV blocks within a single cache (prefix caching)."""
        src_idx, dst_idx = src_to_dists[:, 0], src_to_dists[:, 1]
        for kv_cache in kv_caches:
            kv_cache[0][dst_idx] = kv_cache[0][src_idx]
            kv_cache[1][dst_idx] = kv_cache[1][src_idx]
 
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [16]  # must match cache_config.block_size in IREEPlatform



class IREEAttentionMetadataBuilder(
    AttentionMetadataBuilder[IREEAttentionMetadata]
):
    """
    Converts CommonAttentionMetadata (vLLM's generic per-step metadata)
    into IREEAttentionMetadata (our backend-specific view).
    Called once per step by the model runner before the forward pass.
    """
 
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
        """
        Extract the fields we need from CommonAttentionMetadata.
 
        CommonAttentionMetadata fields used here:
          num_reqs            — number of active requests this step
          num_actual_tokens   — total tokens to process (excl. padding)
          _seq_lens_cpu       — per-request sequence lengths on CPU
          seq_lens            — same but on GPU (fallback)
          slot_mapping        — flat KV cache slot indices [num_actual_tokens]
          block_table_tensor  — paged block table [num_reqs, max_blocks]
        """
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
 
        # Prefer CPU tensor for seq_lens — avoids GPU sync in the builder.
        if common_attn_metadata._seq_lens_cpu is not None:
            seq_lens = common_attn_metadata._seq_lens_cpu[:num_reqs]
        else:
            seq_lens = common_attn_metadata.seq_lens[:num_reqs].cpu()
 
        # is_prefill: true when avg query length > 1 (i.e. not pure decode).
        is_prefill = (num_actual_tokens > num_reqs)
 
        return IREEAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            seq_lens=seq_lens,
            slot_mapping=common_attn_metadata.slot_mapping[:num_actual_tokens],
            block_tables=common_attn_metadata.block_table_tensor,
            is_prefill=is_prefill,
        )



class IREEAttentionBackendImpl(AttentionImpl):
    """
    Per-layer attention implementation.
    Instantiated once per attention layer by vLLM's model executor.
 
    Phase 1: PyTorch SDPA (functionally correct, not paged/efficient).
    Phase 2: replace forward() body with IREE kernel dispatch.
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
 
        # Lazily bound on first forward() from the kv_cache argument.
        self.key_cache: torch.Tensor | None = None
        self.value_cache: torch.Tensor | None = None
 
    def _bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        """Extract K and V cache tensors from the combined cache tensor."""
        if self.key_cache is None:
            # kv_cache shape: [2, num_blocks, block_size, num_kv_heads, head_size]
            self.key_cache = kv_cache[0]
            self.value_cache = kv_cache[1]
 
    def _write_kv_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """
        Scatter new K/V tokens into the paged KV cache.
        key/value arrive as [num_tokens, num_heads * head_size] — flat.
        Cache expects [num_slots, num_kv_heads, head_size] — split into heads.
        """
        # Reshape from flat [T, H*D] to headed [T, num_kv_heads, head_size]
        key_headed = key.view(-1, self.num_kv_heads, self.head_size)
        val_headed = value.view(-1, self.num_kv_heads, self.head_size)

        flat_k = self.key_cache.view(-1, self.num_kv_heads, self.head_size)
        flat_v = self.value_cache.view(-1, self.num_kv_heads, self.head_size)
        flat_k[slot_mapping] = key_headed
        flat_v[slot_mapping] = val_headed
 
    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,           # [num_tokens, num_heads, head_size]
        key: torch.Tensor,             # [num_tokens, num_kv_heads, head_size]
        value: torch.Tensor,           # [num_tokens, num_kv_heads, head_size]
        kv_cache: torch.Tensor,        # [2, num_blocks, block_size, nh, hs]
        attn_metadata: IREEAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,       # quantisation (unused)
        output_block_scale: torch.Tensor | None = None, # quantisation (unused)
    ) -> torch.Tensor:
        """
        Attention forward pass.
 
        Current implementation: plain PyTorch SDPA for correctness.
        TODO (Phase 2): replace with IREE kernel dispatch:
            iree_fn = ctx.modules.module["attention"]
            output = iree_fn(query, key, value, ...)
        """
 
        if attn_metadata is None:
            if output is not None:
                return output.fill_(0)
            return torch.zeros(
                query.shape[0], self.num_heads * self.head_size,
                dtype=query.dtype, device=query.device
            )

 
        n = attn_metadata.num_actual_tokens
        
        # Allocate output buffer if not provided
        if output is None:
            output = torch.zeros(
                query.shape[0], self.num_heads * self.head_size,
                dtype=query.dtype, device=query.device
            )

 
        # Bind and write KV cache for this step.
        self._bind_kv_cache(kv_cache)
        if key is not None and value is not None:
            self._write_kv_cache(key[:n], value[:n], attn_metadata.slot_mapping)
 
        if attn_metadata.is_prefill:
            # Prefill: full causal attention over the prompt tokens.
            # Shape for SDPA: [batch=1, heads, seq, head_dim]
            q = query[:n].unsqueeze(0).transpose(1, 2)  # [1, H, n, d]
            k = key[:n].unsqueeze(0).transpose(1, 2)
            v = value[:n].unsqueeze(0).transpose(1, 2)
 
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                scale=self.scale,
                is_causal=True,
            )
            # attn_out: [1, H, n, d] -> [n, H*d]
            output[:n] = (
                attn_out.squeeze(0)
                .transpose(0, 1)
                .reshape(n, self.num_heads * self.head_size)
            )
        else:
            # Decode: one new token per request attending to cached K/V.
            # TODO: implement proper paged gather + IREE dispatch.
            # For now fill with zeros so the shape is correct.
            output[:n].fill_(0)
 
        return output

"""
Later we need to add:
reshape_and_cache --> once we have KV cache
do_kv_cache_update --> alternative kv cache update path --> called by vLLM if accept_output_buffer = True
build_for_graph_capture --> needed for graph capture
"""