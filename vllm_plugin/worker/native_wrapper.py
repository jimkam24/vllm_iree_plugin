from vllm.v1.worker.gpu_worker import Worker as _NativeWorker
import sys

class NativeWorkerWithSend(_NativeWorker):
    """gpu_worker.Worker patched to send hidden states via PP group
    after the async execute_model phase."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        print(f"[NativeWorkerWithSend] __init__ called, type={type(self).__name__}", 
            file=sys.stderr, flush=True)
    
    def execute_model(self, scheduler_output):
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group
        result = super().execute_model(scheduler_output)
        
        
        # After super().execute_model() returns, check what gpu_worker sent
        # The gpu_worker already called send_tensor_dict internally
        # We can check by running HF reference
        from transformers import AutoModelForCausalLM, AutoConfig
        import torch
        hf_config = AutoConfig.from_pretrained('meta-llama/Llama-3.2-1B')
        hf_config._attn_implementation = 'eager'
        hf_m = AutoModelForCausalLM.from_pretrained(
            'meta-llama/Llama-3.2-1B', config=hf_config, torch_dtype=torch.float32
        ).cuda().eval()
        input_ids = torch.tensor([[791, 6864, 315, 9822, 374]], device='cuda')
        pos_ids = torch.arange(5, device='cuda').unsqueeze(0)
        with torch.no_grad():
            h = hf_m.model.embed_tokens(input_ids)
            pos_emb = hf_m.model.rotary_emb(h, pos_ids)
            seq_len = 5
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device='cuda'))
            cm = torch.zeros(1,1,seq_len,seq_len,dtype=torch.float32,device='cuda')
            cm = cm.masked_fill(~mask.unsqueeze(0).unsqueeze(0), torch.finfo(torch.float32).min)
            for layer in hf_m.model.layers[:14]:
                h = layer(h, attention_mask=cm, position_ids=pos_ids, position_embeddings=pos_emb)
            print(f"[NativeWrapper] HF hidden mean after 14 layers: {h.float().mean().item():.6f}", 
                file=sys.stderr, flush=True)
        
        
        pp = get_pp_group()
        
        print(f"[NativeWrapper] is_last_rank={pp.is_last_rank} result={type(result).__name__}", 
        file=sys.stderr, flush=True)
        
        if not pp.is_last_rank:
            mr = self.model_runner
            
            state = getattr(mr, 'execute_model_state', 'ATTR_MISSING')
            print(f"[NativeWrapper] state={state}", file=sys.stderr, flush=True)
            
            if (hasattr(mr, 'execute_model_state') and 
                mr.execute_model_state is not None):
                hidden = mr.execute_model_state.hidden_states
                
                print(f"[NativeWrapper] model_runner type: {type(mr).__name__}", file=sys.stderr, flush=True)
                print(f"[NativeWrapper] has execute_model_state attr: {hasattr(mr, 'execute_model_state')}", file=sys.stderr, flush=True)
                print(f"[NativeWrapper] state value: {getattr(mr, 'execute_model_state', 'MISSING')}", file=sys.stderr, flush=True)
                
                print(f"[Native rank0] sending hidden_states shape: {hidden.shape}, "
                    f"dtype: {hidden.dtype}, "
                    f"mean: {hidden.float().mean().item():.4f}",
                    file=sys.stderr, flush=True)
                
                
                pp.send_tensor_dict(
                    {"hidden_states": hidden},
                    all_gather_group=get_tp_group(),
                )
        return result