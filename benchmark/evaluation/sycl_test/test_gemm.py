import argparse
import ctypes
import os
import subprocess
import numpy as np

# [修改] 导入 SYCL 模板工具
from benchmark.template.sycl_host_template import create_sycl_func
# [修改] 导入 SYCL 编译工具 (需确保 utils.py 里有 run_sycl_compilation)
from benchmark.utils import run_sycl_compilation as run_compilation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")
    args = parser.parse_args()
    
    # 假设文件名 "gemm_32_32_128.cpp"
    base_name = os.path.basename(args.file)
    shapes = base_name.split(".")[0]
    shape = [int(intg) for intg in shapes.split("_")[1:]]
    print(f"Testing Shapes: {shape}")
    
    # 初始化数据 (int8 -> float 计算)
    A = np.ones((shape[0], shape[1]), dtype=np.int8)
    x = np.ones((shape[1], shape[2]), dtype=np.int8)
    
    name = base_name.split("_")[0] # "gemm"
    y_ctypes = np.zeros((shape[0], shape[2]), dtype=np.float32)

    # 指针转换
    A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)) # 模拟 half (2 bytes) 的一部分，这里为了测试流程通畅，先用 int8 传
    # [修正] 真正的 half 是 2 字节。为了配合 C++ 的 half*，我们在 Python 端最好用 int16 容器来装数据，否则步长不对
    # 但你的原版代码用的是 int8，我们先保持原版逻辑。
    # 如果 C++ 是 half*，Python 传 c_int8*，会导致 C++ 读数据步长翻倍。
    # 这里我们严格对应 C++ 的 half (2 bytes)。
    A_half = A.astype(np.float16)
    x_half = x.astype(np.float16)
    
    A_ptr = A_half.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
    x_ptr = x_half.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
    y_ptr = y_ctypes.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Numpy Golden Result
    y_np = np.matmul(A_half.astype(np.float32), x_half.astype(np.float32))

    # [修改] 编译过程
    so_name = args.file.replace(".cpp", ".so")
    # 使用 SYCL 模板生成 _wrapped.cpp
    file_name = create_sycl_func(args.file, op_type="matmul")
    
    # 编译
    success, output = run_compilation(so_name, file_name)
    if not success:
        print("Compilation Failed:")
        print(output)
        exit(1)
        
    # os.remove(file_name) # 调试时可以先注释掉删除
    
    # 加载库
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
    function = getattr(lib, name + "_kernel")
    
    # 定义参数: A*, B*, C*, m, k, n, size1, size2, size3
    function.argtypes = [
        ctypes.POINTER(ctypes.c_uint16), # half*
        ctypes.POINTER(ctypes.c_uint16), # half*
        ctypes.POINTER(ctypes.c_float),  # float*
        ctypes.c_int, ctypes.c_int, ctypes.c_int, # m, k, n
        ctypes.c_int, ctypes.c_int, ctypes.c_int  # size1, size2, size3
    ]
    function.restype = None
    
    # 计算 Size
    size1 = shape[0] * shape[1] # M*K
    size2 = shape[1] * shape[2] # K*N
    size3 = shape[0] * shape[2] # M*N

    # 调用
    print("Running SYCL Kernel...")
    function(
        A_ptr, x_ptr, y_ptr,
        shape[0], shape[1], shape[2],
        size1, size2, size3
    )
    
    # 校验
    np.testing.assert_allclose(
        y_ctypes, y_np, rtol=1e-03, atol=1e-03, verbose=True
    )
    print("Verification successful!")
    subprocess.run(["rm", so_name])