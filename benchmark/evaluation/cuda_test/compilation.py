#!/usr/bin/env python3
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from benchmark.utils import run_cuda_compilation as run_compilation


def compile_cuda_file(file_path):
    """Compile a single .cuda file into a .so and clean up.

    Returns True if compilation succeeded, False otherwise.
    """
    # Derive names and paths
    # 获取目录路径、基础文件名、无后缀文件名以及目标动态库(.so)的路径
    dir_name, base_name = os.path.split(file_path)
    name_no_ext, _ = os.path.splitext(base_name)
    so_path = os.path.join(dir_name, f"{name_no_ext}.so")

    # Read source and macro
    # 读取原始 CUDA 源文件内容
    with open(file_path, "r") as f:
        src = f.read()
    # 读取预定义的宏文件（通常包含必要的头文件引用或宏定义）
    macro_path = os.path.join("benchmark", "macro", "cuda_macro.txt")
    with open(macro_path, "r") as f:
        macro = f.read()

    # Write temporary backed-up file
    # 将宏内容和源代码拼接，写入一个临时的 _bak.cu 文件用于编译
    bak_path = os.path.join(dir_name, f"{name_no_ext}_bak.cu")
    with open(bak_path, "w") as f:
        f.write(macro + src)

    # Compile
    # 调用编译工具尝试编译该临时文件
    success, output = run_compilation(so_path, bak_path)
    # Clean up backup
    # 无论成功与否，删除临时的源文件
    os.remove(bak_path)

    if success:
        # Remove the .so if it was produced
        # 如果编译成功且生成了 .so 文件，将其删除（此脚本仅用于测试可编译性，不需要保留产物）
        if os.path.exists(so_path):
            os.remove(so_path)
        return True
    else:
        # 编译失败，输出错误信息到标准错误流
        print(f"[ERROR] Failed to compile {file_path}", file=sys.stderr)
        print(output, file=sys.stderr)
        return False


def main():
    # 检查命令行参数，确保提供了一个目录路径
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <cuda_source_directory>", file=sys.stderr)
        sys.exit(1)

    src_dir = sys.argv[1]
    # 匹配目录下所有的 .cu 文件
    pattern = os.path.join(src_dir, "*.cu")
    files = glob.glob(pattern)
    if not files:
        print(f"[WARN] No .cu files found in {src_dir}", file=sys.stderr)
        sys.exit(0)

    # Parallel compile
    # 使用进程池并行执行编译任务，利用多核 CPU 加速，并用 tqdm 显示进度条
    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(executor.map(compile_cuda_file, files), total=len(files))
        )

    # 计算并打印成功率
    total = len(files)
    succ = sum(results)
    print(
        f"[INFO] cuda compilation success rate: {succ}/{total} = {succ/total:.2%}"
    )


if __name__ == "__main__":
    main()