#!/usr/bin/env bash
set -euo pipefail

# 你的 Python 主程序路径
TRANSLATOR_PY="falcon/mcts/transcompile.py"
# Benchmark 根目录
BENCH_DIR="benchmark/data"

# 1. 修改方向：只保留 sycl:cpu
DIRECTIONS=(
    "sycl:cpu"
    # "cpu:hip"
    # "cpu:cuda"
    # "cuda:hip"
    # "cuda:cpu"
    # "hip:cuda"
    # "hip:cpu"
)

# 2. 检查你的文件是否存在 (根据你提供的路径)
SYCL_SRC_DIR="$BENCH_DIR/sycl_code_test"
if [ ! -f "$SYCL_SRC_DIR/gemm_32_32_128.cpp" ]; then
    echo "[ERROR] Target file not found at: $SYCL_SRC_DIR/gemm_32_32_128.cpp"
    echo "Please check the path."
    # exit 1  #如果你想强制退出，取消注释
fi

for dir_pair in "${DIRECTIONS[@]}"; do
    src_plat=${dir_pair%%:*}
    dst_plat=${dir_pair##*:}

    # 自动对应到 benchmark/data/sycl_code_test
    src_dir="$BENCH_DIR/${src_plat}_code_test"

    echo
    echo "=========================================="
    echo "=== Pipeline: $src_plat -> $dst_plat ==="
    echo "=========================================="
    
    # 查找所有 .cpp 文件
    # 使用 compgen 处理通配符，防止没有文件时报错
    files=$(compgen -G "$src_dir"/*.cpp || echo "")
    
    if [ -z "$files" ]; then
        echo "   [WARN] No .cpp files found in $src_dir"
        continue
    fi
    
    i=0
    # 统计文件数量
    file_arr=($files)
    total=${#file_arr[@]}

    for src_file in $files; do
        ((i+=1))
        filename=$(basename "$src_file")

        printf "   [%3d/%3d] Translating %-30s ... " "$i" "$total" "$filename"
        
        # 3. 运行转换命令
        # 关键修改：
        # --max_depth 2: 因为我们只做 Loop Recovery，不需要搜很深
        # --num_simulations 4: 减少模拟次数，加快速度（因为路径是确定的）
        # > /tmp/falcon.log 2>&1: 捕获日志，只有失败时才打印出来
        
        if python3 "$TRANSLATOR_PY" \
            --source "$src_plat" \
            --target "$dst_plat" \
            --file_name "$src_file" \
            --max_depth 2 \
            --num_simulations 4 > /tmp/falcon_trans.log 2>&1; then
            
            echo "✅ Success"
        else
            echo "❌ Failed"
            echo "--- Error Log (Last 20 lines) ---"
            tail -n 20 /tmp/falcon_trans.log
            echo "---------------------------------"
            # 失败后是否继续？取消注释下面的 exit 1 可以遇到错误立即停止
            # exit 1 
        fi
    done
    printf "\n"
done
echo "=== All Done ==="