#!/usr/bin/env python3
import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

# 从 benchmark.utils 导入通用的测试运行函数
from benchmark.utils import run_test

# 算子前缀与测试脚本的映射表
# 这里的映射关系与 CUDA 版本保持一致，因为测试脚本（如 test_gemm.py）
# 会根据传入的文件类型自动处理 SYCL 逻辑（前提是您已经按照之前的步骤实现了 SYCL 版的 test_xxx.py）
TEST_FILE_MAPPING = {
    "deformable": "test_deformable_attention.py",
    "layernorm": "test_layer_norm.py",
    "mha": "test_mha.py",
    "rmsnorm": "test_rms_norm.py",
    "gemm": "test_gemm.py",
    "gemv": "test_gemv.py",
    "bmm": "test_bmm.py",
    "conv1d": "test_conv1d.py",
    "conv2d": "test_conv2d.py",
    "conv2dnchw": "test_conv2dNCHW.py",
    "depthwiseconv": "test_depthwiseconv.py",
    "add": "test_add.py",
    "sign": "test_sign.py",
    "avgpool": "test_avgpool.py",
    "maxpool": "test_maxpool.py",
    "minpool": "test_minpool.py",
    "sumpool": "test_sumpool.py",
    "relu": "test_relu.py",
    "sigmoid": "test_sigmoid.py",
    "gelu": "test_gelu.py",
    "softmax": "test_softmax.py",
}


def process_file(file_path, test_dir):
    """Run the corresponding test for a single .cpp (SYCL) file.

    Returns (file_path, output) where output is the subprocess result or error
    string.
    """
    # 获取文件名（不含路径），例如 "gemm_128_128_128.cpp"
    base_name = os.path.basename(file_path)
    # 提取算子名称，例如 "gemm"
    name = base_name.split("_")[0]
    # 根据算子名称查找对应的测试脚本文件名
    script_name = TEST_FILE_MAPPING.get(name)
    
    if not script_name:
        return file_path, f"[WARN] No test mapping for prefix '{name}'"

    # 拼接测试脚本的完整路径
    test_file = os.path.join(test_dir, script_name)
    # 检查测试脚本文件是否存在
    if not os.path.isfile(test_file):
        return file_path, f"[ERROR] Test script not found: {test_file}"

    # 调用 run_test 执行测试
    # 这里的 file_path 是生成的 SYCL .cpp 文件
    # test_file 是对应的 python 驱动脚本 (benchmark/evaluation/sycl_test/test_xxx.py)
    success, output = run_test(file_path, test_file)
    return file_path, output


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="Run functional tests on translated SYCL programs"
    )
    # 参数 1: 包含待测试 SYCL 文件的源目录 (通常是 generated/sycl/ 或类似的目录)
    parser.add_argument(
        "source_dir",
        help="Directory containing translated .cpp files (SYCL source)",
    )
    # 参数 2: 包含测试脚本的目录 (e.g. benchmark/evaluation/sycl_test/)
    parser.add_argument(
        "test_dir",
        help="Directory containing test scripts",
    )
    args = parser.parse_args()

    # 【关键修改】匹配源目录下所有的 .cpp 文件  
    pattern = os.path.join(args.source_dir, "*.cpp")
    files = glob.glob(pattern)
    
    if not files:
        print(
            f"[WARN] no .cpp files found in {args.source_dir}",
            file=sys.stderr,
        )
        sys.exit(0)

    total = len(files)
    success_count = 0

    print(f"[INFO] Starting SYCL functionality tests on {total} files...")

    # 使用进程池并行执行测试任务
    with ProcessPoolExecutor() as executor:
        future_to_file = {
            executor.submit(process_file, fp, args.test_dir): fp
            for fp in files
        }

        # 使用 tqdm 显示进度条
        for future in tqdm(as_completed(future_to_file), total=total):
            future_to_file[future]
            file_path, output = future.result()
            
            # 检查输出结果
            # 注意：benchmark.utils.run_test 返回的 output 如果是 subprocess.CompletedProcess 对象
            # 则包含 stdout 属性。如果是错误字符串（如 "timeout"），则没有。
            # 我们的测试脚本 (test_gemm.py 等) 需要在成功时打印 "Verification successful!"
            if (
                hasattr(output, "stdout")
                and "Verification successful!" in output.stdout
            ):
                success_count += 1
            else:
                # 打印失败文件的信息以便调试
                print(f"--- FAILED: {os.path.basename(file_path)} ---")
                if hasattr(output, "stdout"):
                     # 打印标准输出（可能包含 Python 的 Traceback 或 Assert Error）
                     print(output.stdout)
                else:
                     # 打印错误字符串（如 timeout 或编译错误）
                     print(output)

    # 打印最终统计
    print("-" * 30)
    print(f"Successful tests: {success_count}")
    print(f"Total files: {total}")
    rate = success_count / total if total else 0.0
    print(f"[INFO] SYCL test success rate: {rate:.2%}")


if __name__ == "__main__":
    main()