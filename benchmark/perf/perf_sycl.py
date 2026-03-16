import argparse
import ctypes
import os
import re
import subprocess
import tempfile
import numpy as np
from string import Template

def _run_sycl_compilation(output_file, source_file):
    """使用 icpx -fsycl 编译 SYCL 共享库 (.so)"""
    try:
        # 注意这里加了 -shared 和 -fPIC，为了让 Python 能作为动态库加载
        result = subprocess.run(
            [
                "icpx",
                "-fsycl",
                "-O2",
                "-std=c++17",
                "-shared",
                "-fPIC",
                source_file,
                "-o",
                output_file,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            timeout=120,
        )
        return result.returncode == 0, result.stdout
    except Exception as e:
        return False, str(e)

# --- 核心修改：遵循 CUDA 的 timed_xxx 逻辑 ---
_SYCL_PERF_TEMPLATE = Template(
    """
#include <sycl/sycl.hpp>
#include <chrono>

using namespace sycl;
using namespace std::chrono;

// 原始 Kernel 代码
${kernel_code}

// 导出 C 接口，返回 float 类型的时间 (ms)
extern "C" float timed_${kernel_name}_kernel(float *A, float *B, float *result, int M, int K, int N) {
    queue q;
    // 这里的 M, K, N 必须参与计算，确保分配的大小正确
    float *d_A = malloc_device<float>(M * K, q);
    float *d_B = malloc_device<float>(K * N, q);
    float *d_result = malloc_device<float>(M * N, q);

    q.memcpy(d_A, A, M * K * sizeof(float));
    q.memcpy(d_B, B, K * N * sizeof(float)).wait();

    // 1. 预热 (Warm-up)
    for (int i = 0; i < 5; ++i) {
        ${kernel_call};
    }
    q.wait();

    // 2. 测速
    auto t0 = high_resolution_clock::now();
    for (int i = 0; i < 50; ++i) {
        ${kernel_call};
    }
    q.wait();
    auto t1 = high_resolution_clock::now();

    float elapsed_ms = duration_cast<microseconds>(t1 - t0).count() / 1000.0f / 50.0f;

    free(d_A, q);
    free(d_B, q);
    free(d_result, q);

    return elapsed_ms; // 直接返回时间，不再 printf
}
"""
)

def benchmark(file_name):
    FAILURE = 1_000_000.0
    
    with open(file_name, "r", encoding="utf-8") as fh:
        kernel_code = fh.read()

    # 解析文件名获取维度 [M, K, N]
    base_name = os.path.basename(file_name)
    name = base_name.split("_")[0] # "gemm"
    shapes = base_name.split(".")[0].split("_")[1:]
    M, K, N = [int(i) for i in shapes]

    # 构造内核调用字符串：这里强制匹配 A, B, result, q
    # 因为大模型生成的 gemm 签名固定是 (float *A, float *B, float *result, queue &q)
    kernel_call = f"{name}(d_A, d_B, d_result, q)"

    # 填充模板
    harness = _SYCL_PERF_TEMPLATE.substitute(
        kernel_code=kernel_code,
        kernel_name=name,
        kernel_call=kernel_call
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "perf_tmp.cpp")
        so_path = os.path.join(tmpdir, "perf_tmp.so")

        with open(src_path, "w", encoding="utf-8") as f:
            f.write(harness)

        success, output = _run_sycl_compilation(so_path, src_path)
        if not success:
            print(f"[Perf Error] Compilation failed: {output}")
            return FAILURE

        try:
            # 加载动态库 (使用 RTLD_GLOBAL 避开 SYCL 锁问题)
            lib = ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
            func = getattr(lib, f"timed_{name}_kernel")
            
            # 设置 ctypes 参数类型
            func.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int, ctypes.c_int, ctypes.c_int
            ]
            func.restype = ctypes.c_float # 必须声明返回 float

            # 准备随机测试数据
            A_np = np.random.randn(M, K).astype(np.float32)
            B_np = np.random.randn(K, N).astype(np.float32)
            C_np = np.zeros((M, N), dtype=np.float32)

            # 调用并获取时间
            elapsed_time = func(
                A_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                B_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                C_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                M, K, N
            )
            return float(elapsed_time)
        except Exception as e:
            print(f"[Perf Error] Runtime error: {e}")
            return FAILURE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", "-f", required=True)
    args = parser.parse_args()
    print(f"Execution time: {benchmark(args.file_name):.4f} ms")