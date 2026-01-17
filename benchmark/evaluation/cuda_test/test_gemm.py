import argparse
import ctypes
import os
import subprocess

import numpy as np

# 导入生成CUDA宿主代码模板的工具，用于包裹Kernel以便调用
from benchmark.template.cuda_host_template import create_cuda_func
# 导入编译工具，用于调用nvcc将生成的代码编译为共享库(.so)
from benchmark.utils import run_cuda_compilation as run_compilation

if __name__ == "__main__":
    # 解析命令行参数，获取输入文件路径
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")
    args = parser.parse_args()
    
    # 从文件名中解析矩阵维度信息
    # 假设文件名格式如 "gemm_16_16_16.cu"，分割后获取 [16, 16, 16]
    base_name = os.path.basename(args.file)
    shapes = base_name.split(".")[0]
    shape = [int(intg) for intg in shapes.split("_")[1:]]
    print(shape)
    
    # Generate random matrices for testing
    # Define the input matrix A and vector x
    # 初始化输入矩阵 A (Shape[0] x Shape[1]) 和 x (Shape[1] x Shape[2])
    # 这里使用全1矩阵进行测试，数据类型为 int8
    A = np.ones((shape[0], shape[1]), dtype=np.int8)  # Keeping as int8
    x = np.ones((shape[1], shape[2]), dtype=np.int8)  # Keeping as int8
    
    # 获取算子名称（如 "gemm"），用于后续通过 ctypes 加载函数
    name = base_name.split("_")[0]
    
    # Create an empty vector y
    # 初始化结果矩阵 y，类型为 float32 以防止累加溢出
    y_ctypes = np.zeros((shape[0], shape[2]), dtype=np.float32)

    # Convert the matrices to contiguous memory for ctypes
    # 获取 numpy 数组的底层数据指针，转换为 ctypes 指针以便传递给 C 函数
    A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    y_ptr = y_ctypes.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Perform gemm using numpy
    # 计算 Numpy 的基准结果 (Golden Result) 用于正确性校验
    # 先转为 int16 运算防止溢出，最后转回 float32
    y_np = np.matmul(A.astype(np.int16), x.astype(np.int16)).astype(np.float32)

    # Load the shared library with the batch matrix multiplication function
    # 准备编译：确定输出的共享库文件名
    so_name = args.file.replace(".cu", ".so")
    # 使用模板生成包含 host 调用代码的完整 CUDA 源文件
    # op_type="matmul" 指示生成适用于矩阵乘法的 host 代码
    file_name = create_cuda_func(args.file, op_type="matmul")
    
    # Load the shared library with the batch matrix multiplication function
    # 调用编译器生成 .so 文件
    success, output = run_compilation(so_name, file_name)
    # 删除生成的临时源文件
    os.remove(file_name)
    
    # 加载编译好的动态链接库
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
    # 获取库中的核心函数，函数名通常约定为 "算子名_kernel"
    function = getattr(lib, name + "_kernel")
    
    # Define the function parameters and return types.
    # 定义 ctypes 函数调用的参数类型列表
    # 参数依次为：A指针, B指针, C指针, A的大小, B的大小, C的大小
    function.argtypes = [
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    function.restype = None
    
    # Call the function with the matrices and dimensions
    # 调用 CUDA 函数执行计算
    # 传入指针和各矩阵的元素总数 (size = M*K, K*N, M*N)
    function(
        A_ptr,
        x_ptr,
        y_ptr,
        np.prod([shape[0], shape[1]]),
        np.prod([shape[1], shape[2]]),
        np.prod([shape[0], shape[2]]),
    )
    
    # Check if the results match
    # 对比 CUDA 计算结果 (y_ctypes) 和 Numpy 基准结果 (y_np)
    # 允许绝对误差和相对误差在 1e-3 范围内
    np.testing.assert_allclose(
        y_ctypes,
        y_np,
        rtol=1e-03,
        atol=1e-03,
        equal_nan=True,
        err_msg="",
        verbose=True,
    )
    print("Verification successful!")
    # 清理生成的共享库文件
    result = subprocess.run(["rm", so_name])
