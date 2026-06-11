import torch, torch.nn as nn, numpy as np
import torch.export as torch_export
import iree.turbine.aot as aot
import iree.runtime as ireert
import iree.compiler as iree_compiler
from transformers import AutoModelForCausalLM, AutoConfig
import transformers.models.llama.modeling_llama as llama_module

config = AutoConfig.from_pretrained('meta-llama/Llama-3.2-1B')
config._attn_implementation = 'eager'
hf = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-3.2-1B', config=config, torch_dtype=torch.float32
).cpu().eval()

def patch_mask():
    orig = llama_module.create_causal_mask
    def traceable(config, input_embeds, **kwargs):
        b, s = input_embeds.shape[:2]
        m = torch.tril(torch.ones(s, s, dtype=torch.bool, device=input_embeds.device))
        m = m.unsqueeze(0).unsqueeze(0).expand(b, 1, -1, -1)
        a = torch.zeros_like(m, dtype=input_embeds.dtype)
        a = a.masked_fill(~m, torch.finfo(input_embeds.dtype).min)
        return a
    llama_module.create_causal_mask = traceable
    return orig, llama_module

class FirstRankWrapper(nn.Module):
    def __init__(self, hf_model, layer_end):
        super().__init__()
        self.embed_tokens = hf_model.model.embed_tokens
        self.layers = nn.ModuleList(list(hf_model.model.layers)[:layer_end])
        self.rotary_emb = hf_model.model.rotary_emb

    def forward(self, input_ids):
        hidden = self.embed_tokens(input_ids)
        seq_len = input_ids.shape[1]
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.rotary_emb(hidden, pos_ids)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device))
        causal_mask = torch.zeros(1, 1, seq_len, seq_len, dtype=hidden.dtype, device=input_ids.device)
        causal_mask = causal_mask.masked_fill(~mask.unsqueeze(0).unsqueeze(0), torch.finfo(hidden.dtype).min)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=causal_mask, position_ids=pos_ids, position_embeddings=pos_emb)
        return hidden

class LastRankWrapper(nn.Module):
    def __init__(self, hf_model, layer_start):
        super().__init__()
        self.layers = nn.ModuleList(list(hf_model.model.layers)[layer_start:])
        self.rotary_emb = hf_model.model.rotary_emb
        self.norm = hf_model.model.norm
        self.lm_head = hf_model.lm_head

    def forward(self, hidden_states, position_ids):
        pos_emb = self.rotary_emb(hidden_states, position_ids)
        seq_len = position_ids.shape[1]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device))
        causal_mask = torch.zeros(1, 1, seq_len, seq_len, dtype=hidden_states.dtype, device=hidden_states.device)
        causal_mask = causal_mask.masked_fill(~mask.unsqueeze(0).unsqueeze(0), torch.finfo(hidden_states.dtype).min)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=causal_mask, position_ids=position_ids, position_embeddings=pos_emb)
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)

input_ids = torch.tensor([[791, 6864, 315, 9822, 374]])
pos_ids = torch.arange(5).unsqueeze(0)

# Test eager
first = FirstRankWrapper(hf, 14).eval()
last = LastRankWrapper(hf, 14).eval()
with torch.no_grad():
    h = first(input_ids)
    logits = last(h, pos_ids)
    print('Eager token:', int(torch.argmax(logits[0,-1,:]).item()))

# Export first rank
seq_dim = torch_export.Dim('seq_len', min=1, max=4096)
orig, mod = patch_mask()
try:
    exported = aot.export(first, args=(input_ids,), dynamic_shapes={'input_ids': {1: seq_dim}})
    exported.save_mlir('/tmp/first_rank.mlir')
    print('First rank export OK')
finally:
    mod.create_causal_mask = orig

iree_compiler.tools.compile_file('/tmp/first_rank.mlir', output_file='/tmp/first_rank.vmfb',
    target_backends=['cuda'], extra_args=['--iree-cuda-target=sm_70', '--iree-input-type=torch'])
print('First rank compile OK')

# Export last rank
orig, mod = patch_mask()
try:
    exported = aot.export(last, args=(h, pos_ids),
        dynamic_shapes={'hidden_states': {1: seq_dim}, 'position_ids': {1: seq_dim}})
    exported.save_mlir('/tmp/last_rank.mlir')
    print('Last rank export OK')
finally:
    mod.create_causal_mask = orig

iree_compiler.tools.compile_file('/tmp/last_rank.mlir', output_file='/tmp/last_rank.vmfb',
    target_backends=['cuda'], extra_args=['--iree-cuda-target=sm_70', '--iree-input-type=torch'])
print('Last rank compile OK')

# Run through IREE
cfg = ireert.Config('cuda://0')
ctx = ireert.SystemContext(config=cfg)
# for name, path in [('first', '/tmp/first_rank.vmfb'), ('last', '/tmp/last_rank.vmfb')]:
#     with open(path, 'rb') as f:
#         vm = ireert.VmModule.copy_buffer(ctx.instance, f.read())
#     ctx.add_vm_module(vm)

# Need separate contexts for separate modules — use two contexts
cfg1 = ireert.Config('cuda://0')
ctx1 = ireert.SystemContext(config=cfg1)
with open('/tmp/first_rank.vmfb', 'rb') as f:
    vm1 = ireert.VmModule.copy_buffer(ctx1.instance, f.read())
ctx1.add_vm_module(vm1)
fn1 = ctx1.modules.module['main']

cfg2 = ireert.Config('cuda://0')
ctx2 = ireert.SystemContext(config=cfg2)
with open('/tmp/last_rank.vmfb', 'rb') as f:
    vm2 = ireert.VmModule.copy_buffer(ctx2.instance, f.read())
ctx2.add_vm_module(vm2)
fn2 = ctx2.modules.module['main']

# Run
ids_np = input_ids.numpy().astype(np.int64)
iree_h = fn1(ireert.asdevicearray(cfg1.device, ids_np))
h_np = np.array(iree_h)
p_np = pos_ids.numpy().astype(np.int64)
iree_logits = fn2(ireert.asdevicearray(cfg2.device, h_np), ireert.asdevicearray(cfg2.device, p_np))
logits_t = torch.from_numpy(np.array(iree_logits))
print('IREE pipeline token:', int(torch.argmax(logits_t[0,-1,:]).item()))
print('Match:', int(torch.argmax(logits_t[0,-1,:]).item()) == 12366)