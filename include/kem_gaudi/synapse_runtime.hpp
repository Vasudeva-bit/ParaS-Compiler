/*
 * Copyright (c) 2026 Centre for Development of Advanced Computing (C-DAC)
 * SPDX-License-Identifier: Apache-2.0

 Authors = [Vasudeva-bit, Codex]; Auditors = [Vasudeva-bit]
 */
#ifndef __PARAS_GAUDI_SYNAPSE_RUNTIME_HPP__
#define __PARAS_GAUDI_SYNAPSE_RUNTIME_HPP__

#include <synapse_api.h>
#include <sycl/event.hpp>

#if defined(PARAS_GAUDI_HOST_FALLBACK)
#include <sycl/queue.hpp>
#include <sycl/device_selector.hpp>
#endif

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

namespace paras_gaudi {

template <typename> inline constexpr bool dependent_false_v = false;

template <typename Kernel>
inline sycl::event unsupported_kernel(Kernel&& k) {
#if defined(PARAS_GAUDI_HOST_FALLBACK)
    sycl::queue host_q{sycl::cpu_selector_v};
    host_q.submit(std::forward<Kernel>(k));
    host_q.wait();
    return sycl::event{sycl::backend::gaudi};
#else
    static_assert(dependent_false_v<Kernel>,
                  "Gaudi v0.1 supports only 1D float vector-add kernels of the form "
                  "out[i] = in_a[i] + in_b[i].");
    while (true) {}
#endif
}

inline void check(synStatus status, const char* operation) {
    if (status != synSuccess) {
        throw std::runtime_error(std::string("ParaS Gaudi SynapseAI call failed: ") + operation +
                                 " (status " + std::to_string(static_cast<int>(status)) + ")");
    }
}

struct Runtime {
    synDeviceId device{};
    bool device_acquired{false};
    synStreamHandle h2d{};
    synStreamHandle compute{};
    synStreamHandle d2h{};
    std::mutex mutex;

    Runtime() {
        check(synInitialize(), "synInitialize");
        try {
            check(synDeviceAcquireByDeviceType(&device, synDeviceGaudi2),
                  "synDeviceAcquireByDeviceType(Gaudi2)");
            device_acquired = true;
            check(synStreamCreate(&h2d, device, STREAM_TYPE_COPY_HOST_TO_DEVICE, 0),
                  "synStreamCreate(H2D)");
            check(synStreamCreate(&compute, device, STREAM_TYPE_COMPUTE, 0),
                  "synStreamCreate(compute)");
            check(synStreamCreate(&d2h, device, STREAM_TYPE_COPY_DEVICE_TO_HOST, 0),
                  "synStreamCreate(D2H)");
        } catch (...) {
            if (d2h) synStreamDestroy(d2h);
            if (compute) synStreamDestroy(compute);
            if (h2d) synStreamDestroy(h2d);
            if (device_acquired) synDeviceRelease(device);
            synDestroy();
            throw;
        }
    }

    ~Runtime() {
        if (d2h) synStreamDestroy(d2h);
        if (compute) synStreamDestroy(compute);
        if (h2d) synStreamDestroy(h2d);
        if (device_acquired) synDeviceRelease(device);
        synDestroy();
    }
};

inline Runtime& runtime() {
    static Runtime instance;
    return instance;
}

inline sycl::event add_f32(std::size_t count, const float* a, const float* b, float* c, float alpha = 1.0f) {
    if (alpha != 1.0f) {
        throw std::runtime_error("paras_gaudi::add_f32: alpha != 1.0 is not supported on Gaudi v0.1 (got alpha=" +
                                 std::to_string(alpha) + ")");
    }
    if (count == 0)
        return sycl::event{sycl::backend::gaudi};
    if (!a || !b || !c)
        throw std::invalid_argument("Gaudi vector add received a null pointer");
    if (count > std::numeric_limits<unsigned>::max())
        throw std::length_error("Gaudi v0.1 vector length exceeds Synapse tensor limits");

    Runtime& rt = runtime();
    std::lock_guard<std::mutex> lock(rt.mutex);
    const std::uint64_t bytes = static_cast<std::uint64_t>(count) * sizeof(float);
    synGraphHandle graph{};
    synTensor tensors[3]{};
    synRecipeHandle recipe{};
    synEventHandle copyDone{};
    synEventHandle computeDone{};
    void* hostA = nullptr;
    void* hostB = nullptr;
    void* hostC = nullptr;
    std::uint64_t deviceA = 0, deviceB = 0, deviceC = 0, workspace = 0;
    std::uint64_t workspaceBytes = 0;

    auto cleanup = [&]() {
        if (rt.h2d) synStreamSynchronize(rt.h2d);
        if (rt.compute) synStreamSynchronize(rt.compute);
        if (rt.d2h) synStreamSynchronize(rt.d2h);
        if (computeDone) synEventDestroy(computeDone);
        if (copyDone) synEventDestroy(copyDone);
        if (recipe) synRecipeDestroy(recipe);
        if (graph) synGraphDestroy(graph);
        for (synTensor tensor : tensors)
            if (tensor) synTensorDestroy(tensor);
        if (deviceA) synDeviceFree(rt.device, deviceA, 0);
        if (deviceB) synDeviceFree(rt.device, deviceB, 0);
        if (deviceC) synDeviceFree(rt.device, deviceC, 0);
        if (workspace) synDeviceFree(rt.device, workspace, 0);
        if (hostA) synHostFree(rt.device, hostA, 0);
        if (hostB) synHostFree(rt.device, hostB, 0);
        if (hostC) synHostFree(rt.device, hostC, 0);
    };

    try {
        check(synGraphCreate(&graph, synDeviceGaudi2), "synGraphCreate(Gaudi2)");
        unsigned sizes[SYN_MAX_TENSOR_DIM] = {static_cast<unsigned>(count), 1, 1, 1};
        const char* names[] = {"paras_add_a", "paras_add_b", "paras_add_c"};
        for (unsigned i = 0; i < 3; ++i) {
            synTensorDescriptor desc{};
            desc.m_dataType = syn_type_float;
            desc.m_dims = 4;
            desc.m_name = names[i];
            for (unsigned d = 0; d < 4; ++d) {
                desc.m_sizes[d] = sizes[d];
                desc.m_minSizes[d] = sizes[d];
            }
            check(synTensorCreate(&tensors[i], &desc, nullptr, 0), "synTensorCreate");
        }
        synTensor inputs[] = {tensors[0], tensors[1]};
        synTensor outputs[] = {tensors[2]};
        check(synNodeCreate(graph, inputs, outputs, 2, 1, nullptr, 0, "add_fwd_f32",
                            "ParaS vector add", nullptr, nullptr),
              "synNodeCreate(add_fwd_f32)");
        check(synGraphCompile(&recipe, graph, "paras_vector_add_f32", nullptr),
              "synGraphCompile");

        check(synHostMalloc(rt.device, bytes, 0, &hostA), "synHostMalloc(input A)");
        check(synHostMalloc(rt.device, bytes, 0, &hostB), "synHostMalloc(input B)");
        check(synHostMalloc(rt.device, bytes, 0, &hostC), "synHostMalloc(output)");
        std::memcpy(hostA, a, bytes);
        std::memcpy(hostB, b, bytes);

        check(synDeviceMalloc(rt.device, bytes, 0, 0, &deviceA), "synDeviceMalloc(input A)");
        check(synDeviceMalloc(rt.device, bytes, 0, 0, &deviceB), "synDeviceMalloc(input B)");
        check(synDeviceMalloc(rt.device, bytes, 0, 0, &deviceC), "synDeviceMalloc(output)");
        check(synWorkspaceGetSize(&workspaceBytes, recipe), "synWorkspaceGetSize");
        if (workspaceBytes)
            check(synDeviceMalloc(rt.device, workspaceBytes, 0, 0, &workspace),
                  "synDeviceMalloc(workspace)");

        synLaunchTensorInfo launchTensors[3]{};
        const std::uint64_t addresses[] = {deviceA, deviceB, deviceC};
        for (unsigned i = 0; i < 3; ++i) {
            launchTensors[i].pTensorAddress = addresses[i];
            launchTensors[i].tensorName = names[i];
            launchTensors[i].tensorType = DATA_TENSOR;
        }
        check(synEventCreate(&copyDone, rt.device, 0), "synEventCreate(copy)");
        check(synEventCreate(&computeDone, rt.device, 0), "synEventCreate(compute)");
        check(synMemCopyAsync(rt.h2d, reinterpret_cast<std::uint64_t>(hostA), bytes, deviceA,
                              HOST_TO_DRAM),
              "synMemCopyAsync(input A)");
        check(synMemCopyAsync(rt.h2d, reinterpret_cast<std::uint64_t>(hostB), bytes, deviceB,
                              HOST_TO_DRAM),
              "synMemCopyAsync(input B)");
        check(synEventRecord(copyDone, rt.h2d), "synEventRecord(copy)");
        check(synStreamWaitEvent(rt.compute, copyDone, 0), "synStreamWaitEvent(copy)");
        check(synLaunch(rt.compute, launchTensors, 3, workspace, recipe), "synLaunch");
        check(synEventRecord(computeDone, rt.compute), "synEventRecord(compute)");
        check(synStreamWaitEvent(rt.d2h, computeDone, 0), "synStreamWaitEvent(compute)");
        check(synMemCopyAsync(rt.d2h, deviceC, bytes,
                              reinterpret_cast<std::uint64_t>(hostC), DRAM_TO_HOST),
              "synMemCopyAsync(output)");
        check(synStreamSynchronize(rt.d2h), "synStreamSynchronize");
        std::memcpy(c, hostC, bytes);
    } catch (...) {
        cleanup();
        throw;
    }
    cleanup();
    return sycl::event{sycl::backend::gaudi};
}

} // namespace paras_gaudi

#endif // __PARAS_GAUDI_SYNAPSE_RUNTIME_HPP__
