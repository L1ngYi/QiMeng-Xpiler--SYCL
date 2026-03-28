import argparse
import ctypes
import os
import uuid

import numpy as np

from benchmark.template.sycl_host_template import (
    create_sycl_func,
    get_sycl_function_metadata,
)
from benchmark.utils import (
    configure_sycl_environment,
    get_sycl_ctype,
    get_sycl_numpy_dtype,
    preload_sycl_runtime,
    run_sycl_compilation as run_compilation,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")
    args = parser.parse_args()

    configure_sycl_environment()
    metadata = get_sycl_function_metadata(args.file)

    base_name = os.path.basename(args.file)
    shape = [
        int(token)
        for token in os.path.splitext(base_name)[0].split("_")[1:]
    ]
    if len(shape) < 3:
        raise ValueError(f"Invalid GEMM shape encoded in file name: {base_name}")
    print(f"Testing Shapes: {shape[:3]}", flush=True)

    pointer_params = metadata["pointer_params"]
    scalar_params = metadata["scalar_params"]
    if len(pointer_params) != 3 or len(scalar_params) < 3:
        raise ValueError("SYCL GEMM test expects 3 pointer params and 3 scalar dims.")

    m_dim, k_dim, n_dim = shape[:3]
    A_data = np.ones(
        (m_dim, k_dim), dtype=get_sycl_numpy_dtype(pointer_params[0]["dtype"])
    )
    B_data = np.ones(
        (k_dim, n_dim), dtype=get_sycl_numpy_dtype(pointer_params[1]["dtype"])
    )
    C_data = np.zeros(
        (m_dim, n_dim), dtype=get_sycl_numpy_dtype(pointer_params[2]["dtype"])
    )

    y_np = np.matmul(A_data.astype(np.float32), B_data.astype(np.float32)).astype(
        np.float32
    )

    unique_id = uuid.uuid4().hex[:8]
    so_name = args.file.replace(".cpp", f"_{unique_id}.so")
    wrapper_file = create_sycl_func(args.file, op_type="matmul")

    try:
        success, output = run_compilation(so_name, wrapper_file)
        if not success:
            print("Compilation Failed:\n" + output, flush=True)
            raise SystemExit(1)

        preload_sycl_runtime()
        rtld_flag = getattr(os, "RTLD_GLOBAL", getattr(ctypes, "RTLD_GLOBAL", 0))
        lib = ctypes.CDLL(os.path.abspath(so_name), mode=rtld_flag)
        kernel_func = getattr(lib, f"{metadata['kernel_name']}_kernel")

        pointer_ctypes = [
            ctypes.POINTER(get_sycl_ctype(param["dtype"]))
            for param in pointer_params
        ]
        scalar_ctypes = [
            get_sycl_ctype(param["dtype"]) for param in scalar_params
        ]
        kernel_func.argtypes = pointer_ctypes + scalar_ctypes
        kernel_func.restype = None

        scalar_values = [m_dim, k_dim, n_dim][: len(scalar_params)]

        print("Running SYCL Kernel...", flush=True)
        kernel_func(
            A_data.ctypes.data_as(pointer_ctypes[0]),
            B_data.ctypes.data_as(pointer_ctypes[1]),
            C_data.ctypes.data_as(pointer_ctypes[2]),
            *scalar_values,
        )

        np.testing.assert_allclose(
            C_data.astype(np.float32), y_np, rtol=1e-03, atol=1e-03
        )
        print("Verification successful!", flush=True)
    finally:
        if os.path.exists(wrapper_file):
            os.remove(wrapper_file)
        if os.path.exists(so_name):
            os.remove(so_name)
