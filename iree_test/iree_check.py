"""
What this script does:
  1. Load Llama 3.2 1B in PyTorch, run greedy decoding -> reference tokens
  2. Export the model to MLIR via iree-turbine AOT
  3. Compile to .vmfb targeting cuda
  4. Run inference via IREE runtime
  5. Compare token outputs — pass/fail
"""

import os
import torch
import numpy as np

import iree.turbine.aot as aot
import iree.compiler as compiler
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import transformers.models.llama.modeling_llama as llama_module
import torch.export as torch_export

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID   = "meta-llama/Llama-3.2-1B"
PROMPT     = "The capital of France is"
MAX_TOKENS = 10          # short — just enough to catch divergence
DEVICE_PT  = "cuda:0"    # PyTorch runs on GPU 0 (V100S)
VMFB_PATH  = "llama32_1b.vmfb"
MLIR_PATH  = "llama32_1b.mlir"
IREE_DEVICE = "cuda://0" # IREE also targets GPU 0
# ─────────────────────────────────────────────────────────────────────────────

def pytorch_greedy(model, tokenizer, prompt, max_new_tokens, device):
    """Run greedy decoding with PyTorch and return generated token ids."""
    print("\n[PyTorch] Tokenizing prompt...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    print(f"[PyTorch] Running greedy decoding for {max_new_tokens} tokens...")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,        # greedy — deterministic
            temperature=1.0,
            use_cache=True,
        )

    # only the newly generated tokens (strip the prompt)
    generated = output[0, input_ids.shape[1]:]
    tokens = generated.cpu().tolist()
    text   = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"[PyTorch] Generated tokens : {tokens}")
    print(f"[PyTorch] Decoded text     : '{text}'")
    return tokens, input_ids.cpu()


def export_to_mlir(model, input_ids):
    """Export the model forward pass to MLIR using iree-turbine AOT."""
    print("\n[IREE] Exporting model to MLIR via iree-turbine AOT...")

    print("\n[IREE] Patching create_causal_mask for torch.export compatibility...")

    # Save original
    original_create_causal_mask = llama_module.create_causal_mask

    def traceable_causal_mask(config, input_embeds, past_key_values_length=0,
                            sliding_window=None, cache_position=None,
                            attention_mask=None, **kwargs):
        """Simple traceable causal mask — no vmap, no .item() calls."""
        batch_size, seq_len = input_embeds.shape[:2]
        full_len = seq_len + past_key_values_length
        # causal mask: lower triangular, shape [batch, 1, seq, full_len]
        mask = torch.tril(
            torch.ones((seq_len, full_len), dtype=torch.bool, device=input_embeds.device)
        )
        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
        # convert to float additive mask (0 = attend, -inf = ignore)
        additive = torch.zeros_like(mask, dtype=input_embeds.dtype)
        additive = additive.masked_fill(~mask, torch.finfo(input_embeds.dtype).min)
        return additive

    # Patch
    llama_module.create_causal_mask = traceable_causal_mask

    try:
        model_cpu = model.cpu()
        input_ids_cpu = input_ids.cpu()
        
        # Concrete example input for tracing
        example_input = input_ids_cpu  # shape [1, 6]

        # Tell torch.export that dim 1 (sequence length) is dynamic
        dynamic_shapes = {
            "input_ids": {1: torch_export.Dim("seq_len", min=1, max=4096)}
}
        
        class LlamaWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.model = m
            def forward(self, input_ids):
                out = self.model(input_ids=input_ids)
                return out.logits

        wrapper = LlamaWrapper(model_cpu).eval()

        print("[IREE] Exporting model to MLIR via iree-turbine AOT...")
        print("[DEBUG] create_causal_mask is:", llama_module.create_causal_mask.__name__)
        exported = aot.export(
            wrapper,
            args=(example_input,),
            dynamic_shapes=dynamic_shapes,
        )
        exported.save_mlir(MLIR_PATH)
        print(f"[IREE] MLIR saved to {MLIR_PATH}")

    finally:
        # Always restore original — even if export fails
        llama_module.create_causal_mask = original_create_causal_mask
        print("[IREE] create_causal_mask restored.")

    return exported


def compile_to_vmfb():
    """Compile the MLIR to a .vmfb artifact targeting CUDA."""
    print(f"\n[IREE] Compiling MLIR -> {VMFB_PATH} (target: cuda)...")

    compiler.tools.compile_file(
        MLIR_PATH,
        output_file=VMFB_PATH,
        target_backends=["cuda"],
        extra_args=[
            "--iree-cuda-target=sm_70",  # V100S = sm_70
        ],
    )
    print(f"[IREE] Compiled artifact saved to {VMFB_PATH}")


def iree_greedy(tokenizer, input_ids, max_new_tokens):
    """Run greedy decoding using the IREE runtime (.vmfb)."""
    print(f"\n[IREE] Loading {VMFB_PATH} into IREE runtime...")
    import iree.runtime as ireert

    config  = ireert.Config(IREE_DEVICE)
    ctx     = ireert.SystemContext(config=config)

    with open(VMFB_PATH, "rb") as f:
        vmfb = f.read()

    vm_module = ireert.VmModule.copy_buffer(ctx.instance, vmfb)
    ctx.add_vm_module(vm_module)
    main_fn = ctx.modules.module["main"]

    # Autoregressive loop — IREE runs one forward pass at a time
    current_ids = input_ids.numpy().astype(np.int64)
    generated_tokens = []

    print(f"[IREE] Running greedy decoding for {max_new_tokens} tokens...")
    for step in range(max_new_tokens):
        iree_input = ireert.asdevicearray(config.device, current_ids)
        logits_iree = main_fn(iree_input)
        logits_np   = np.asarray(logits_iree)          # shape: [1, seq_len, vocab]

        # Greedy: argmax over last token position
        next_token = int(np.argmax(logits_np[0, -1, :]))
        generated_tokens.append(next_token)

        # Append to sequence for next step
        next_token_arr = np.asarray([[next_token]], dtype=np.int64)
        current_ids    = np.concatenate([current_ids, next_token_arr], axis=1)

    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"[IREE] Generated tokens : {generated_tokens}")
    print(f"[IREE] Decoded text     : '{text}'")
    return generated_tokens


def compare(pt_tokens, iree_tokens):
    """Compare token sequences and report pass/fail."""
    print("\n" + "="*60)
    print("CORRECTNESS CHECK")
    print("="*60)
    print(f"PyTorch tokens : {pt_tokens}")
    print(f"IREE tokens    : {iree_tokens}")

    if pt_tokens == iree_tokens:
        print("\n✅  PASS — tokens match exactly")
        return True
    else:
        mismatches = [(i, p, r) for i, (p, r) in
                      enumerate(zip(pt_tokens, iree_tokens)) if p != r]
        print(f"\n❌  FAIL — {len(mismatches)} mismatch(es):")
        for i, p, r in mismatches:
            print(f"   position {i}: PyTorch={p}, IREE={r}")
        print("\nNext step: try Qwen2.5-3B before changing anything else.")
        return False


def main():

    print("="*60)
    print("IREE Standalone Correctness Check — Llama 3.2 1B")
    print("="*60)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n[Setup] Loading tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"[Setup] Loading model onto {DEVICE_PT} (this downloads ~2.5GB on first run)...")
    config = AutoConfig.from_pretrained(MODEL_ID)
    config._attn_implementation = "eager"   # bypass the vmap-based sdpa mask

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        torch_dtype=torch.float32,
        device_map=DEVICE_PT,
    )
    model.eval()
    print("[Setup] Model loaded.")

    # ── PyTorch reference ─────────────────────────────────────────────────────
    pt_tokens, input_ids = pytorch_greedy(
        model, tokenizer, PROMPT, MAX_TOKENS, DEVICE_PT
    )

    # ── IREE compile (skip if .vmfb already exists) ───────────────────────────
    if not os.path.exists(VMFB_PATH):
        export_to_mlir(model, input_ids)
        compile_to_vmfb()
    else:
        print(f"\n[IREE] Found existing {VMFB_PATH} — skipping compile step.")
        print("       Delete it to force recompile.")

    # # ── IREE inference ────────────────────────────────────────────────────────
    iree_tokens = iree_greedy(tokenizer, input_ids, MAX_TOKENS)

    # # ── Compare ───────────────────────────────────────────────────────────────
    compare(pt_tokens, iree_tokens)


if __name__ == "__main__":
    main()