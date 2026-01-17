import argparse
import ctypes
import os
import subprocess
import numpy as np

# 【改动 1】导入 SYCL 的宿主代码模板工具 
from benchmark.template.sycl_host_template import create_sycl_func
# 【改动 2】导入 SYCL 的编译工具 
from benchmark.utils import run_sycl_compilation as run_compilation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")
    args = parser.parse_args()
    
    base_name = os.path.basename(args.file)
    shapes = base_name.split(".")[0]
    shape = [int(intg) for intg in shapes.split("_")[1:]]
    print(f"Testing Shapes: {shape}")
    
    # 保持与 CUDA 版本一致的数据初始化逻辑
    # 这里的 int8 和 float32 类型必须与你的 Kernel 签名一致
    A = np.ones((shape[0], shape[1]), dtype=np.int8) 
    x = np.ones((shape[1], shape[2]), dtype=np.int8) 
    
    name = base_name.split("_")[0] # e.g., "gemm"
    y_ctypes = np.zeros((shape[0], shape[2]), dtype=np.float32)

    # 转换为 ctypes 指针
    A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    y_ptr = y_ctypes.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Numpy 计算基准结果 (Golden Result)
    y_np = np.matmul(A.astype(np.int16), x.astype(np.int16)).astype(np.float32)

    # 【改动 3】文件后缀处理：SYCL 代码通常是 .cpp
    so_name = args.file.replace(".cpp", ".so")
    
    # 【改动 4】生成带有 SYCL Runtime (Queue/Malloc) 的包装代码
    file_name = create_sycl_func(args.file, op_type="matmul")
    
    # 【改动 5】调用 icpx/dpcpp 编译
    success, output = run_compilation(so_name, file_name)
    
    if not success:
        print(f"Compilation Failed:\n{output}")
        exit(1)

    os.remove(file_name) # 清理临时生成的包装文件
    
    # 加载动态库 (逻辑与 CUDA 完全一致)
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
    
    # 【注意】你的模板里生成的函数名必须加上 "_kernel" 后缀，或者在这里调整
    try:
        function = getattr(lib, name + "_kernel")
    except AttributeError:
        # 如果 SYCL 编译器没有正确处理 extern "C"，名字可能会乱，这里是调试的好地方
        print(f"Error: Could not find function '{name}_kernel' in {so_name}")
        exit(1)

    # 定义参数类型：3个指针 + 3个尺寸 (M, N, K)
    function.argtypes = [
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    function.restype = None
    
    # 执行 SYCL Kernel
    print("Running SYCL kernel...")
    function(
        A_ptr,
        x_ptr,
        y_ptr,
        np.prod([shape[0], shape[1]]), # Size A
        np.prod([shape[1], shape[2]]), # Size B
        np.prod([shape[0], shape[2]]), # Size C (注意：这里可能是传递 M, N, K 还是 total size，取决于你的 kernel 写法)
    )
    
    # 验证结果
    try:
        np.testing.assert_allclose(
            y_ctypes,
            y_np,
            rtol=1e-03,
            atol=1e-03,
            equal_nan=True,
            err_msg="SYCL result does not match Numpy baseline!",
            verbose=True,
        )
        print("Verification successful! SYCL works!")
    except AssertionError as e:
        print(str(e))
        print("First 10 mismatches (Calculated vs Expected):")
        print(y_ctypes.flatten()[:10])
        print(y_np.flatten()[:10])

    # 清理 .so 文件
    subprocess.run(["rm", so_name])