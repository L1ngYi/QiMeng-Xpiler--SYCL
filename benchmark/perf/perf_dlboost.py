import argparse
import ctypes
import os
import re
from string import Template

import numpy as np
import torch

from benchmark.utils import conv2d_nchw, maxpool_np
from benchmark.utils import run_dlboost_compilation as run_compilation


def perf_function(file_name):
    with open(file_name, "r") as f:
        original_function = f.read()

    # 提取函数签名
    function_signature_pattern = r"void (\w+)\(([^()]*)\)"
    match = re.search(function_signature_pattern, original_function, re.DOTALL)
    if not match:
        raise ValueError("Could not find function signature.")

    kernel_name = match.group(1)
    param_list_str = match.group(2)

    # 规范化参数列表定义
    params = [param_str.strip() for param_str in param_list_str.split(",")]
    param_list = ", ".join(
        [
            " ".join(param.split()[:-1]) + " " + param.split()[-1]
            for param in params
        ]
    )

    # 动态生成调用参数 (去掉指针和引用，提取纯变量名)
    arg_names = []
    for param in params:
        clean_param = param.replace('*', ' ').replace('&', ' ')
        var_name = clean_param.split()[-1]
        arg_names.append(var_name)
    called_param_list = ", ".join(arg_names)

    # 构造测速模板
    cpp_pef_template = Template(
        """
    #include <sys/time.h>
    #include <math.h>
    #include <float.h>
    #include <stdio.h>
    #include <immintrin.h>
    #include <stdint.h>
    typedef unsigned short half;

    // Original function
    ${original_function}

    extern "C" float timed_${kernel_name}(${param_list}) {
        struct timeval start, end;
        for (int i = 0; i < 10; i++) {
            ${kernel_name}(${called_param_list});
        }
        // 获取开始时间
        gettimeofday(&start, NULL);
        for (int i = 0; i < 1000; i++) {
            ${kernel_name}(${called_param_list});
        }
        // 获取结束时间
        gettimeofday(&end, NULL);

        int time_us = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
        float us_time = time_us / 1000.0f / 1000.0f;
        return us_time;
    }
    """
    )

    pattern = r'extern\s*"C"\s*'
    cleaned_code = re.sub(pattern, "", original_function)

    # 替换模板
    new_code = cpp_pef_template.substitute(
        kernel_name=kernel_name,
        param_list=param_list,
        called_param_list=called_param_list,
        original_function=cleaned_code,
    )

    # 保存文件
    output_file = file_name.replace(".cpp", "_bak.cpp")
    with open(output_file, "w") as f:
        f.write(new_code)


def perf_pipeline(file_name):
    perf_function(file_name)
    backup_file_name = file_name.replace(".cpp", "_bak.cpp")
    so_name = file_name.replace(".cpp", ".so")
    success, output = run_compilation(so_name, backup_file_name)
    if not success:
        raise RuntimeError(f"DLBoost Compilation Failed: {output}")


def benchmark(file_name):
    execution_time = 0
    base_name = os.path.basename(file_name)
    name = base_name.split("_")[0]
    
    so_path = os.path.join(os.getcwd(), file_name.replace(".cpp", ".so"))
    bak_path = os.path.join(os.getcwd(), file_name.replace(".cpp", "_bak.cpp"))

    try:
        perf_pipeline(file_name)
        lib = ctypes.CDLL(so_path)
        function = getattr(lib, "timed_" + name)
        
        # 针对 gemm 的 Ctypes 传参处理
        if name == "gemm":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            
            A = np.random.rand(shape[0], shape[1]).astype("float16")
            B = np.random.rand(shape[1], shape[2]).astype("float16")
            C = np.zeros((shape[0], shape[2]), dtype="float32")

            A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
            B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
            C_ptr = C.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

            function.argtypes = [
                ctypes.POINTER(ctypes.c_uint16),
                ctypes.POINTER(ctypes.c_uint16),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            function.restype = ctypes.c_float
            
            execution_time = function(A_ptr, B_ptr, C_ptr, shape[0], shape[1], shape[2])
        else:
            print(f"Warning: Setup for {name} is not fully implemented in this script.")
            return 0.1 

    except Exception as e:
        print(f"Benchmark failed: {e}")
        return 0.0

    finally:
        # 清理垃圾
        if os.path.exists(bak_path):
            os.remove(bak_path)
        if os.path.exists(so_path):
            os.remove(so_path)

    return execution_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the transcompile benchmark"
    )
    parser.add_argument(
        "--file_name",
        "-f",
        required=True,
        help="Path to the input DLboost file to benchmark",
    )
    args = parser.parse_args()
    execution_time = benchmark(file_name=args.file_name)
    print(f"Execution time: {execution_time:.4f} ms")