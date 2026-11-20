# Intel Gaudi 2 preview

ParaS v0.1 adds an explicitly identified `gaudi` backend. Invoke it with:

```bash
parascc -parasdevice gaudi:gaudi2 source.cpp <compiler_flags> \
    -I/usr/include/habanalabs -L/usr/lib/habanalabs -lsynapse
```

The Gaudi target is parsed and compiled as ordinary host C++. ParaS does not pass
CUDA/HIP language or offload flags. The SDK include directory and `libsynapse.so`
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
