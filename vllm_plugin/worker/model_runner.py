"""
IREEModelRunner — prepares inputs, runs the IREE forward pass, returns outputs.
 
Phase 1: runs PyTorch model via vLLM's model registry on CUDA.
          IREE dispatch replaces the forward call in Step 1 of Phase 1.
Phase 2: swap IREE compile target to amd-aie, rest unchanged.
"""
 
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.core.sched.output import SchedulerOutput
 
logger = init_logger(__name__)
 
 
class IREEModelRunner:
    """
    Sits between IREEWorker and the actual IREE runtime.
    Responsibilities:
      - load weights and compile to .vmfb
      - track per-request state (computed tokens, block ids)
      - build input tensors from SchedulerOutput each step
      - run forward pass (PyTorch now, IREE dispatch later)
      - return ModelRunnerOutput to the worker
    """
 
    def __init__(self, vllm_config: VllmConfig, device: torch.device = None) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        # Default to cuda:0 for Phase 1 — IREE targets this GPU.
        self.device = device or torch.device("cuda:0")
 
        # Set after load_model()
        self.model: nn.Module | None = None
 
        # req_id -> {prompt_token_ids, num_computed_tokens}
        self.requests: dict[str, dict] = {}


    def load_model(self) -> None:
        """
        Load weights via vLLM's model registry.
        TODO (Step 1): replace with iree-turbine export + .vmfb compilation.
        """
        logger.info("IREEModelRunner: loading %s ...", self.model_config.model)
        self.model = get_model(vllm_config=self.vllm_config)
        self.model.eval()
        logger.info("IREEModelRunner: model loaded.")


    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Return a dummy KV cache spec so vLLM's scheduler can compute
        block counts. All fields after block_size are keyword-only.
        TODO: return a real spec once IREE KV buffers are wired up.
        """
        return {
            "iree_attn": FullAttentionSpec(
                self.cache_config.block_size,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float16,
            )
        }


    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Allocate and bind KV cache tensors to model layers.
        TODO (Phase 1 Step 3): implement IREE-side KV buffer allocation.
        """
        pass


    # called for memory profiling
    def profile_run(self) -> None:
        self._dummy_run(num_tokens=self.scheduler_config.max_num_batched_tokens)

    def _dummy_run(self, num_tokens: int) -> None:
        """Single forward pass with fake inputs — used for profiling and warmup."""
        assert self.model is not None, "load_model() must be called first"
        input_ids = torch.zeros(num_tokens, dtype=torch.long, device=self.device)
        positions = torch.arange(num_tokens, dtype=torch.long, device=self.device)
        with torch.no_grad(), set_forward_context(None, self.vllm_config):
            self.model(input_ids=input_ids, positions=positions)


    def warm_up(self) -> None:
        """
        Called after KV cache is allocated.
        TODO (Step 1): trigger IREE .vmfb compilation here instead.
        """
        logger.info("IREEModelRunner: warming up...")
        self._dummy_run(self.scheduler_config.max_num_batched_tokens)
        logger.info("IREEModelRunner: warmup done.")


    def _update_states(self, scheduler_output: SchedulerOutput) -> None:
        """Keep internal request states in sync with the scheduler."""
 
        # Remove finished requests
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
 
        # Register new requests
        for req in scheduler_output.scheduled_new_reqs:
            self.requests[req.req_id] = {
                "prompt_token_ids": req.prompt_token_ids,
                "num_computed_tokens": 0,
            }
 
        # Update computed token counts for continuing requests
        cached = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
            if req_id in self.requests:
                self.requests[req_id]["num_computed_tokens"] = (
                    cached.num_computed_tokens[i]
                )
 


    def _prepare_inputs(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build input_ids and positions tensors from the scheduler output.
        Concatenates tokens from all scheduled requests into a flat batch.
        """
        token_ids: list[int] = []
        positions: list[int] = []
 
        for req in scheduler_output.scheduled_new_reqs:
            state = self.requests[req.req_id]
            start = state["num_computed_tokens"]
            chunk = req.prompt_token_ids[start:]
            token_ids.extend(chunk)
            positions.extend(range(start, start + len(chunk)))
 
        # Also handle continuing (cached) requests scheduled for decode
        cached = scheduler_output.scheduled_cached_reqs
        for req_id in cached.req_ids:
            state = self.requests.get(req_id)
            if state is None:
                continue
            n = state["num_computed_tokens"]
            # Decode step: one new token at position n
            token_ids.append(state["prompt_token_ids"][n]
                             if n < len(state["prompt_token_ids"])
                             else 0)
            positions.append(n)
 
        input_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        pos = torch.tensor(positions, dtype=torch.long, device=self.device)
        return input_ids, pos

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput:
        """
        Main inference entry point called every engine step.
 
        Current flow (PyTorch placeholder):
          update states -> build inputs -> forward -> return output
 
        TODO (Step 1): replace the forward call with IREE runtime dispatch:
          iree_fn = ctx.modules.module["main"]
          logits = iree_fn(iree_input_ids)
        """
        assert self.model is not None, "load_model() must be called first"
 
        if not scheduler_output.total_num_scheduled_tokens:
            return EMPTY_MODEL_RUNNER_OUTPUT
 
        self._update_states(scheduler_output)
        input_ids, positions = self._prepare_inputs(scheduler_output)
 
        with torch.no_grad(), set_forward_context(None, self.vllm_config):
            # TODO: replace with IREE dispatch
            _hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
            )
 
        # Build the output — only req_ids and req_id_to_index are required,
        # everything else defaults to None / empty (see ModelRunnerOutput dataclass).
        req_ids = list(self.requests.keys())
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            # sampled_token_ids defaults to [] — sampler not wired up yet.
            # TODO: add greedy sampler output once logits are extracted.
        )

        
        
"""
May also need:
_build_attention_metadata --> needed once IREEAttentionBackend is active
capture_model --> graph capture (static shapes)
"""