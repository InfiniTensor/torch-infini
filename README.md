# torch-infini

Experimental PyTorch plugin for the Infini stack.

The plugin keeps upstream PyTorch unchanged and registers the `infini`
device through PyTorch's `PrivateUse1` backend slot:

```python
import torch
import torch_infini

src = torch.arange(16, dtype=torch.float32).reshape(4, 4)
x = torch.empty(src.shape, dtype=src.dtype, device="infini:0")
x.copy_(src)

out = torch.empty_like(src)
out.copy_(x)
torch.testing.assert_close(out, src)
```

This first-step bridge is intentionally narrow. It wires PyTorch device and
stream management, allocation, synchronization, and contiguous tensor copies
to InfiniRT. General ATen operator coverage is left to later InfiniCore and
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

`INFINI_RT_INCLUDE_DIRS` and `INFINI_RT_LIBRARY_DIRS` can be used when the
headers or library are not under a single install prefix.

The wheel links to `libinfinirt.so` but does not bundle it. Before importing
`torch_infini`, make the InfiniRT library directory available to the dynamic
loader. Add the directory to the system loader configuration and run
`ldconfig`, or expose it for the current shell:

```bash
export LD_LIBRARY_PATH="/path/to/infini-rt-prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Use `lib64` instead of `lib` when that is the library directory in the InfiniRT
install prefix.

Some InfiniRT installations currently expose CUDA headers through their public
headers. For those installations, set `CUDA_INCLUDE_DIRS` when CUDA headers are
outside the standard toolkit paths. `torch-infini` does not otherwise depend on
CUDA; InfiniRT should eventually export any required transitive include paths.

## Runtime backend

torch-infini automatically uses an accelerator backend compiled into InfiniRT
when that backend reports at least one available device. When no compiled
accelerator backend has an available device, it falls back to CPU when CPU
support is included in the InfiniRT build. torch-infini requires this CPU
support so the fallback is always available. InfiniRT stores its selection per
thread, so torch-infini binds the selected backend whenever a thread enters a
runtime operation.

## Scope

The initial implementation supports:

- `device="infini:0"`
- `torch.infini.is_available()`
- `torch.infini.device_count()`
- `torch.infini.current_device()`
- `torch.infini.set_device(index)`
- `torch.infini.synchronize()`
- `torch.infini.Stream()`
- `torch.infini.current_stream()`
- `torch.infini.default_stream()`
- `torch.infini.set_stream(stream)`
- `torch.infini.stream(stream)`
- `torch.empty(..., device="infini")`
- `torch.empty_strided(..., device="infini")`
- contiguous `copy_` between CPU and Infini tensors

The `torch.infini` module follows `torch.cuda` naming and semantics for the
device and stream-management operations it implements. Stream priorities,
events, random-number generation, asynchronous memory and copy behavior, and
general ATen operators are not exposed yet. Unsupported operations should fail
clearly instead of silently falling back through CPU.
