import torch
import subprocess
import iree.compiler.tools

SAVE_FILES = 0

#@title 1. Define a program using `torch.nn.Module`
torch.manual_seed(0)

class LinearModule(torch.nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    self.weight = torch.nn.Parameter(torch.randn(in_features, out_features))
    self.bias = torch.nn.Parameter(torch.randn(out_features))

  def forward(self, input):
    return (input @ self.weight) + self.bias

linear_module = LinearModule(4, 3)

#@title 2. Export the program using `aot.export()`
import iree.turbine.aot as aot

example_arg = torch.randn(4)
export_output = aot.export(linear_module, example_arg)


#@title 3a. Compile fully to a deployable artifact, in our existing Python session

# Staying in Python gives the API a chance to reuse memory, improving
# performance when compiling large programs.

compiled_binary = export_output.compile(save_to=None)

# Use the IREE runtime API to test the compiled program.
import numpy as np
import iree.runtime as ireert

config = ireert.Config("local-task")
vm_module = ireert.load_vm_module(
    ireert.VmModule.wrap_buffer(config.vm_instance, compiled_binary.map_memory()),
    config,
)

input = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
result = vm_module.main(input)
print(result.to_host())

#@title 3b. Output MLIR then continue from Python or native tools later

# Leaving Python allows for file system checkpointing and grants access to
# native development workflows.

mlir_file_path = "linear_module_pytorch.mlirbc"
vmfb_file_path = "linear_module_pytorch_llvmcpu.vmfb"

if SAVE_FILES:
    print("Exported .mlir:")
    export_output.print_readable()
    export_output.save_mlir(mlir_file_path)

# 4. Compile using Python API (equivalent to iree-compile CLI)
# extra_args passes the flags that don't have direct Python API equivalents
compiled_flatbuffer = iree.compiler.tools.compile_file(
    mlir_file_path,
    input_type="torch",
    target_backends=["llvm-cpu"],
    extra_args=[
        "--iree-hal-target-device=local",
        "--iree-llvmcpu-target-cpu=host",
    ],
)

print("Compiled successfully, binary size:", len(compiled_flatbuffer), "bytes")

if SAVE_FILES:
    # Optionally save the .vmfb to disk
    with open(vmfb_file_path, "wb") as f:
        f.write(compiled_flatbuffer)
    print("Saved vmfb to", vmfb_file_path)

# 5. Run using Python API (equivalent to iree-run-module CLI)
config = ireert.Config("local-task")
ctx = ireert.SystemContext(config=config)
vm_module = ireert.VmModule.copy_buffer(ctx.instance, compiled_flatbuffer)
ctx.add_vm_module(vm_module)

input_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
result = ctx.modules.module["main"](input_data)
print("Result:", result.to_host())
