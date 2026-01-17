#!/usr/bin/env python3
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

# 【修改 1】导入 benchmark/utils.py 中新写的 SYCL 编译函数
from benchmark.utils import run_sycl_compilation as run_compilation


def compile_sycl_file(file_path):
    """Compile a single .cpp (SYCL) file into a .so and clean up.

    Returns True if compilation succeeded, False otherwise.
    """
    # Derive names and paths
    # 获取目录路径、基础文件名
    dir_name, base_name = os.path.split(file_path)
    name_no_ext, _ = os.path.splitext(base_name)
    
    # 目标动态库路径
    so_path = os.path.join(dir_name, f"{name_no_ext}.so")

    # Read source and macro
    # 读取原始 SYCL (.cpp) 源文件内容
    with open(file_path, "r") as f:
        src = f.read()
    
    # 【关键修改 2】读取 SYCL 的宏文件
    # 在 benchmark/macro/ 目录下也要创建一个 sycl_macro.txt
    macro_path = os.path.join("benchmark", "macro", "sycl_macro.txt")
    
    # 如果没有宏文件，就用空字符串代替，防止报错
    if os.path.exists(macro_path):
        with open(macro_path, "r") as f:
            macro = f.read()
    else:
        macro = ""

    # Write temporary backed-up file
    # 【关键修改 3】生成的临时文件后缀使用 .cpp
    bak_path = os.path.join(dir_name, f"{name_no_ext}_bak.cpp")
    with open(bak_path, "w") as f:
        f.write(macro + "\n" + src)

    # Compile
    # 调用 run_sycl_compilation 尝试编译
    success, output = run_compilation(so_path, bak_path)
    
    # Clean up backup
    # 删除临时源文件
    if os.path.exists(bak_path):
        os.remove(bak_path)

    if success:
        # Remove the .so if it was produced
        # 编译成功后删除 .so，因为我们只测试是否“可编译”
        if os.path.exists(so_path):
            os.remove(so_path)
        return True
    else:
        # 编译失败，打印具体的编译器报错信息 (icpx output)
        print(f"[ERROR] Failed to compile {file_path}", file=sys.stderr)
        print(output, file=sys.stderr)
        return False


def main():
    # 检查命令行参数
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <sycl_source_directory>", file=sys.stderr)
        sys.exit(1)

    src_dir = sys.argv[1]
    
    # 【关键修改 4】匹配 .cpp 文件 
    pattern = os.path.join(src_dir, "*.cpp")
    files = glob.glob(pattern)
    
    if not files:
        print(f"[WARN] No .cpp files found in {src_dir}", file=sys.stderr)
        sys.exit(0)

    print(f"[INFO] Found {len(files)} SYCL files. Starting compilation test...")

    # Parallel compile
    # 使用进程池并行编译
    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(executor.map(compile_sycl_file, files), total=len(files))
        )

    # 统计结果
    total = len(files)
    succ = sum(results)
    print(
        f"[INFO] SYCL compilation success rate: {succ}/{total} = {succ/total:.2%}"
    )


if __name__ == "__main__":
    main()