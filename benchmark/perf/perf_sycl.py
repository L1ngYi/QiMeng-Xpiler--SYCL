import argparse
import ctypes
import os
import tempfile
from string import Template

import numpy as np

from benchmark.template.sycl_host_template import (
    get_sycl_function_metadata,
    get_sycl_matmul_size_exprs,
    get_sycl_invocation_args,
    get_sycl_shape_from_file_name,
)
from benchmark.utils import (
    configure_sycl_environment,
    get_sycl_ctype,
    get_sycl_numpy_dtype,
    preload_sycl_runtime,
    run_sycl_compilation as run_compilation,
)

_SYCL_PERF_TEMPLATE = Template(
    """
#include <sycl/sycl.hpp>
#include <chrono>
#include <iostream>

using namespace sycl;
using namespace std::chrono;

${kernel_code}

extern "C" float timed_${kernel_name}_kernel(${extern_c_params}) {
    try {
        queue q;
        ${alloc_code}

        ${memcpy_htod_code}
        q.wait();

        for (int i = 0; i < 5; ++i) {
            ${kernel_call};
        }
        q.wait();

        auto t0 = high_resolution_clock::now();
        for (int i = 0; i < 50; ++i) {
            ${kernel_call};
        }
        q.wait();
        auto t1 = high_resolution_clock::now();

        float elapsed_ms = duration_cast<microseconds>(t1 - t0).count() / 1000.0f / 50.0f;

        ${memcpy_dtoh_code}
        q.wait();

        ${free_code}

        return elapsed_ms;
    } catch (sycl::exception const &e) {
        std::cerr << "[SYCL Perf Error] " << e.what() << std::endl;
        return 1000000.0f;
    }
}
"""
)

_FAILURE = 1_000_000.0

def _build_perf_harness(kernel_code, metadata, shape):
    size_exprs = get_sycl_matmul_size_exprs(metadata, shape=shape)
    pointer_params = metadata["pointer_params"]

    alloc_code = []
    memcpy_htod = []
    free_code = []
    pointer_name_map = {}

    for index, param in enumerate(pointer_params):
        name = param["name"]
        storage_dtype = param["storage_dtype"]
        alloc_code.append(f"{storage_dtype} *d_{name} = sycl::malloc_device<{storage_dtype}>({size_exprs[index]}, q);")
        free_code.append(f"sycl::free(d_{name}, q);")
        pointer_name_map[name] = f"d_{name}"

        if index < len(pointer_params) - 1:
            memcpy_htod.append(
                f"q.memcpy(d_{name}, {name}, {size_exprs[index]} * sizeof({storage_dtype}));"
            )

    result_param = pointer_params[-1]
    memcpy_dtoh_code = (
        f"q.memcpy({result_param['name']}, d_{result_param['name']}, "
        f"{size_exprs[-1]} * sizeof({result_param['storage_dtype']}));"
    )

    return _SYCL_PERF_TEMPLATE.substitute(
        kernel_code=kernel_code,
        kernel_name=metadata["kernel_name"],
        extern_c_params=", ".join(
            [param["full"] for param in metadata["data_params"]]
        ),
        alloc_code="\n    ".join(alloc_code),
        memcpy_htod_code="\n    ".join(memcpy_htod),
        kernel_call=f"{metadata['kernel_name']}({', '.join(get_sycl_invocation_args(metadata, pointer_name_map=pointer_name_map))})",
        memcpy_dtoh_code=memcpy_dtoh_code,
        free_code="\n    ".join(free_code),
    )


def benchmark(file_name):
    try:
        configure_sycl_environment()
        metadata = get_sycl_function_metadata(file_name)
        with open(file_name, "r", encoding="utf-8") as fh:
            kernel_code = fh.read()
    except Exception as e:
        print(f"[Perf Error] Failed to prepare SYCL benchmark: {e}")
        return _FAILURE

    try:
        shape = get_sycl_shape_from_file_name(file_name)
        harness = _build_perf_harness(kernel_code, metadata, shape)
    except Exception as e:
        print(f"[Perf Error] Failed to prepare SYCL benchmark: {e}")
        return _FAILURE
    m_dim, k_dim, n_dim = shape[:3]

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "perf_tmp.cpp")
        so_path = os.path.join(tmpdir, "perf_tmp.so")

        with open(src_path, "w", encoding="utf-8") as f:
            f.write(harness)

        success, output = run_compilation(so_path, src_path)
        if not success:
            print(f"[Perf Error] Compilation failed: {output}")
            return _FAILURE

        try:
            preload_sycl_runtime()
            rtld_flag = getattr(os, "RTLD_GLOBAL", getattr(ctypes, "RTLD_GLOBAL", 0))
            lib = ctypes.CDLL(so_path, mode=rtld_flag)
            func = getattr(lib, f"timed_{metadata['kernel_name']}_kernel")

            pointer_ctypes = [
                ctypes.POINTER(get_sycl_ctype(param["dtype"]))
                for param in metadata["pointer_params"]
            ]
            scalar_ctypes = [
                get_sycl_ctype(param["dtype"])
                for param in metadata["scalar_params"]
            ]
            func.argtypes = pointer_ctypes + scalar_ctypes
            func.restype = ctypes.c_float

            pointer_params = metadata["pointer_params"]
            arrays = [
                np.random.randn(m_dim, k_dim).astype(
                    get_sycl_numpy_dtype(pointer_params[0]["dtype"])
                ),
                np.random.randn(k_dim, n_dim).astype(
                    get_sycl_numpy_dtype(pointer_params[1]["dtype"])
                ),
                np.zeros((m_dim, n_dim), dtype=get_sycl_numpy_dtype(pointer_params[2]["dtype"])),
            ]

            scalar_values = [m_dim, k_dim, n_dim][: len(metadata["scalar_params"])]
            elapsed_time = func(
                arrays[0].ctypes.data_as(pointer_ctypes[0]),
                arrays[1].ctypes.data_as(pointer_ctypes[1]),
                arrays[2].ctypes.data_as(pointer_ctypes[2]),
                *scalar_values,
            )
            return float(elapsed_time)
        except Exception as e:
            print(f"[Perf Error] Runtime error: {e}")
            return _FAILURE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", "-f", required=True)
    args = parser.parse_args()
    print(f"Execution time: {benchmark(args.file_name):.4f} ms")
