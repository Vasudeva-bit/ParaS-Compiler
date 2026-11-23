# Intel Gaudi 2 preview

ParaS v0.1 adds an explicitly identified `gaudi` backend. Invoke it with:

```bash
parascc -parasdevice gaudi:gaudi2 source.cpp <compiler_flags> \
    -I/usr/include/habanalabs -L/usr/lib/habanalabs -lSynapse -Wl,-rpath,/usr/lib/habanalabs
```

The Gaudi target is parsed and compiled as ordinary host C++. ParaS does not pass
CUDA/HIP language or offload flags. The SDK include directory and `libSynapse.so`
link option are deliberately supplied by the application build command.

## Supported kernel

The initial matcher accepts one-dimensional `float` vector addition in a command
group, with the same work-item index used for all three buffers:

```cpp
q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        c[i] = a[i] + b[i];
    });
});
```

The arrays must be named pointer or array variables with `float` elements backed by
host-accessible storage. The recognized operation lowers to a Synapse graph node
using the `add_fwd_f32` GUID; the runtime copies the host inputs to HBM, launches
the compiled recipe, and copies the result back. Other submitted kernels fail
compilation with a message describing the supported form. `gemm`, arbitrary
lambdas, nd-ranges, accessors, custom TPC kernels, and general SYCL-to-TPC lowering
are outside this preview.

This runtime uses the SynapseAI C API and the Gaudi 2 device type. Confirm that the
installed SynapseAI SDK provides the `add_fwd_f32` GUID before running this preview;
the development environment used to prepare this backend did not include the SDK or
Gaudi hardware for an SDK-version or device execution check.

Each supported submission currently builds and compiles a graph for that operation
and synchronizes before returning. This is a correctness-oriented initial path, not
an asynchronous queue implementation.

## PyTorch / Torch_ParaS Integration (v0.1 Prototype)

To run PyTorch tensor addition on Intel Gaudi 2 through `torch_paras`:

1. **Device Discovery**: Set `export PARAS_GAUDI_AS_GPU=1` so `torch_paras` detects the Gaudi 2 accelerator as `paras:0`.
2. **Unsupported Kernel Fallback**: When compiling `torch_paras`, optionally define `-DPARAS_GAUDI_HOST_FALLBACK` to route any auxiliary SYCL kernels outside of vector addition to the CPU host queue.
3. **Alpha Guard**: In `c[i] = a[i] + alpha * b[i]`, `alpha=1.0` (standard tensor `x + y`) executes on Gaudi 2 via `add_fwd_f32`. Non-unit `alpha` values raise a descriptive `std::runtime_error` rather than silently calculating incorrect results.

### Sample Verification Script:

```python
import os
import torch
import torch_paras

# Ensure Gaudi is reported to torch_paras as device 0
# export PARAS_GAUDI_AS_GPU=1

# 1. Prepare inputs on CPU
x_cpu = torch.randn(1024, dtype=torch.float32)
y_cpu = torch.randn(1024, dtype=torch.float32)

# 2. Transfer to Gaudi (paras:0)
x = x_cpu.to("paras:0")
y = y_cpu.to("paras:0")

# 3. Execute addition on Intel Gaudi 2
z = x + y

# 4. Verify against CPU ground truth
z_cpu = z.cpu()
expected = x_cpu + y_cpu
print("Gaudi 2 vs CPU allclose:", torch.allclose(z_cpu, expected, atol=1e-5))

# 5. Verify alpha guard raises on non-1.0 alpha
try:
    z_alpha = torch.add(x, y, alpha=2.0)
    print("ERROR: alpha=2.0 should have raised!")
except RuntimeError as e:
    print("SUCCESS: alpha=2.0 correctly raised:", e)
```
