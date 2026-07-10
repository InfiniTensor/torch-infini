# torch-infini

Experimental PyTorch plugin for the Infini stack.

The plugin keeps upstream PyTorch unchanged and registers the `infini`
device through PyTorch's `PrivateUse1` backend slot:

```python
import torch
import torch_infini

x = torch.empty((4, 4), device="infini:0")
torch.infini.synchronize()
```

This first-step bridge is intentionally narrow. It wires PyTorch device
management, allocation, synchronization, and contiguous tensor copies to
InfiniRT. General ATen operator coverage is left to later InfiniCore and
InfiniOps integration work.

The implementation follows PyTorch's documented out-of-tree backend path:
`PrivateUse1` is renamed to `infini`, C++ kernels are registered through the
dispatcher, and the extension is built with `torch.utils.cpp_extension`.

## Build

Build and install InfiniRT first, then point this package at that prefix:

```bash
export INFINI_RT_PREFIX=/path/to/infini-rt-prefix
pip install --no-build-isolation --no-deps .
```

`INFINI_RT_INCLUDE_DIRS`, `INFINI_RT_LIBRARY_DIRS`, and
`INFINI_RT_RUNTIME_LIBRARY_DIRS` can be used when the headers or library are not
under a single install prefix.

Set `CUDA_INCLUDE_DIRS` if an installed InfiniRT backend header depends on CUDA
headers outside the standard toolkit paths.

## Scope

The current MVP supports:

- `device="infini:0"`
- `torch.infini.is_available()`
- `torch.infini.device_count()`
- `torch.infini.current_device()`
- `torch.infini.set_device(index)`
- `torch.infini.synchronize()`
- `torch.empty(..., device="infini")`
- `torch.empty_strided(..., device="infini")`
- contiguous `copy_` between CPU and Infini tensors

Unsupported ATen operators should fail clearly instead of silently falling back
through CPU.
