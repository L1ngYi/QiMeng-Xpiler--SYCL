import argparse
import ctypes
import os
import subprocess
import numpy as np
import uuid

from benchmark.template.sycl_host_template import create_sycl_func
from benchmark.utils import run_sycl_compilation as run_compilation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")
    args = parser.parse_args()
    
    base_name = os.path.basename(args.file)
    shapes = base_name.split(".")[0]
    shape = [int(intg) for intg in shapes.split("_")[1:]] # [32, 32, 128]
    print(f"Testing Shapes: {shape}", flush=True)
    
    # 1. 准备数据并强制保活 (防 GC)
    A = np.ones((shape[0], shape[1]), dtype=np.float32)
    B = np.ones((shape[1], shape[2]), dtype=np.float32)
    C_ctypes = np.zeros((shape[0], shape[2]), dtype=np.float32)

    A_data = np.array(A, copy=True, order='C')
    B_data = np.array(B, copy=True, order='C')
    C_data = np.array(C_ctypes, copy=True, order='C')
    
    y_np = np.matmul(A_data, B_data)

    # 2. 编译过程
    unique_id = uuid.uuid4().hex[:8]
    so_name = args.file.replace(".cpp", f"_{unique_id}.so")
    file_name = create_sycl_func(args.file, op_type="matmul") # 生成 _wrapped.cpp
    
    success, output = run_compilation(so_name, file_name)
    if not success:
        print("Compilation Failed:\n" + output, flush=True)
        exit(1)
        
    # 3. 加载库 (RTLD_GLOBAL 解决 SYCL 锁问题)
    rtld_flag = getattr(os, 'RTLD_GLOBAL', 256)
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name), mode=rtld_flag)
    kernel_func = getattr(lib, "gemm_kernel")

    # 4. 【核心咬合】：精准匹配你的 sycl_host_template.py 生成的参数列表
    # 参数是：float *A, float *B, float *result, int size1, int size2, int size3
    kernel_func.argtypes = [
        ctypes.POINTER(ctypes.c_float), # A
        ctypes.POINTER(ctypes.c_float), # B
        ctypes.POINTER(ctypes.c_float), # result
        ctypes.c_int,                  # size1 (M*K)
        ctypes.c_int,                  # size2 (K*N)
        ctypes.c_int                   # size3 (M*N)
    ]
    kernel_func.restype = None

    # 计算尺寸
    size1 = shape[0] * shape[1] # 32*32
    size2 = shape[1] * shape[2] # 32*128
    size3 = shape[0] * shape[2] # 32*128

    print("Running SYCL Kernel...", flush=True)
    # 5. 调用：3个指针 + 3个尺寸
    kernel_func(
        A_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        C_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        size1, size2, size3
    )
    
    # 6. 校验
    np.testing.assert_allclose(C_data, y_np, rtol=1e-03, atol=1e-03)
    print("Verification successful!", flush=True)
    
    if os.path.exists(so_name):
        os.remove(so_name)