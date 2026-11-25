/*
 * Copyright (c) 2026 Centre for Development of Advanced Computing (C-DAC)
 * SPDX-License-Identifier: Apache-2.0

 Authors = [Vasudeva-bit, AI (Codex, Anti-Gravity models, GitHub Copilot via VSCode)]; Auditors = [Vasudeva-bit]
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
    synStreamHandle stream{};
    std::mutex mutex;

    Runtime() {
        check(synInitialize(), "synInitialize");
        try {
            check(synDeviceAcquireByDeviceType(&device, synDeviceGaudi2),
                  "synDeviceAcquireByDeviceType(Gaudi2)");
            device_acquired = true;
            check(synStreamCreateGeneric(&stream, device, 0),
                  "synStreamCreateGeneric");
        } catch (...) {
            if (stream) synStreamDestroy(stream);
            if (device_acquired) synDeviceRelease(device);
            synDestroy();
            throw;
        }
    }

    ~Runtime() {
        if (stream) synStreamDestroy(stream);
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
    synSectionHandle sections[3]{};
    synTensor tensors[3]{};
    synRecipeHandle recipe{};
    void* hostA = nullptr;
    void* hostB = nullptr;
    void* hostC = nullptr;
    std::uint64_t deviceA = 0, deviceB = 0, deviceC = 0, workspace = 0;
    std::uint64_t workspaceBytes = 0;

    auto cleanup = [&]() {
        if (rt.stream) synStreamSynchronize(rt.stream);
        if (recipe) synRecipeDestroy(recipe);
        for (synSectionHandle sec : sections)
            if (sec) synSectionDestroy(sec);
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
        const char* names[3] = {"A", "B", "C"};
        for (unsigned i = 0; i < 3; ++i) {
            check(synSectionCreate(&sections[i], 0, graph), "synSectionCreate");
            check(synSectionSetPersistent(sections[i], true), "synSectionSetPersistent");
            synTensorDescriptor desc{};
            desc.m_dataType = syn_type_float;
            desc.m_dims = 1;
            desc.m_sizes[0] = static_cast<unsigned>(count);
            desc.m_name = names[i];
            desc.m_tensorType = DATA_TENSOR;
            check(synTensorCreate(&tensors[i], &desc, sections[i], 0), "synTensorCreate");
        }
        synTensor inputs[] = {tensors[0], tensors[1]};
        synTensor outputs[] = {tensors[2]};
        check(synNodeCreate(graph, inputs, outputs, 2, 1, nullptr, 0, "add_fwd_f32",
                            "paras_vector_add", nullptr, nullptr),
              "synNodeCreate(add_fwd_f32)");
        check(synGraphCompile(&recipe, graph, "paras_vector_add_f32", nullptr),
              "synGraphCompile");

        check(synWorkspaceGetSize(&workspaceBytes, recipe), "synWorkspaceGetSize");

        uint64_t ids[3] = {0};
        check(synTensorRetrieveIds(recipe, names, ids, 3), "synTensorRetrieveIds");

        check(synHostMalloc(rt.device, bytes, 0, &hostA), "synHostMalloc(input A)");
        check(synHostMalloc(rt.device, bytes, 0, &hostB), "synHostMalloc(input B)");
        check(synHostMalloc(rt.device, bytes, 0, &hostC), "synHostMalloc(output)");
        std::memcpy(hostA, a, bytes);
        std::memcpy(hostB, b, bytes);

        check(synDeviceMalloc(rt.device, bytes, 0, 0, &deviceA), "synDeviceMalloc(input A)");
        check(synDeviceMalloc(rt.device, bytes, 0, 0, &deviceB), "synDeviceMalloc(input B)");
        check(synDeviceMalloc(rt.device, bytes, 0, 0, &deviceC), "synDeviceMalloc(output)");
        if (workspaceBytes)
            check(synDeviceMalloc(rt.device, workspaceBytes, 0, 0, &workspace),
                  "synDeviceMalloc(workspace)");

        synLaunchTensorInfo launchTensors[3]{};
        const std::uint64_t addresses[] = {deviceA, deviceB, deviceC};
        for (unsigned i = 0; i < 3; ++i) {
            launchTensors[i].tensorName = names[i];
            launchTensors[i].pTensorAddress = addresses[i];
            launchTensors[i].tensorType = DATA_TENSOR;
            launchTensors[i].tensorId = ids[i];
            launchTensors[i].tensorSize[0] = static_cast<unsigned>(count);
        }

        check(synMemCopyAsync(rt.stream, (uint64_t)(uintptr_t)hostA, bytes, deviceA, HOST_TO_DRAM),
              "synMemCopyAsync(input A)");
        check(synMemCopyAsync(rt.stream, (uint64_t)(uintptr_t)hostB, bytes, deviceB, HOST_TO_DRAM),
              "synMemCopyAsync(input B)");
        check(synLaunch(rt.stream, launchTensors, 3, workspace, recipe, 0),
              "synLaunch");
        check(synMemCopyAsync(rt.stream, deviceC, bytes, (uint64_t)(uintptr_t)hostC, DRAM_TO_HOST),
              "synMemCopyAsync(output)");
        check(synStreamSynchronize(rt.stream), "synStreamSynchronize");

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
