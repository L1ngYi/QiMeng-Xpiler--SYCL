#!/usr/bin/env bash
set -euo pipefail

# Resolve the repository root relative to this script so it can be run
# from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Python transcompiler entry point
TRANSLATOR_PY="${SCRIPT_DIR}/falcon/mcts/transcompile.py"
# Benchmark data root directory
BENCH_DIR="${SCRIPT_DIR}/benchmark/data"

# CPU → SYCL transcompilation direction
DIRECTIONS=(
    "cpu:sycl"
)

# Source directory containing plain C++ kernels
CPU_SRC_DIR="$BENCH_DIR/cpp_code_test"
if [ ! -d "$CPU_SRC_DIR" ]; then
    echo "[ERROR] CPU source directory not found: $CPU_SRC_DIR"
    exit 1
fi

# 指定要处理的单个文件
TARGET_FILE="gemm_32_32_128.cpp"
src_file="${CPU_SRC_DIR}/${TARGET_FILE}"

if [ ! -f "$src_file" ]; then
    echo "[ERROR] Target file not found: $src_file"
    exit 1
fi

for dir_pair in "${DIRECTIONS[@]}"; do
    src_plat=${dir_pair%%:*}
    dst_plat=${dir_pair##*:}

    src_dir="$BENCH_DIR/${src_plat}_code_test"

    echo
    echo "=========================================="
    echo "=== Pipeline: $src_plat -> $dst_plat ==="
    echo "=========================================="

    filename=$(basename "$src_file")
    echo "   Translating ${filename} ... "

    # Run the transcompiler with PYTHONPATH set so benchmark/ is importable
    if PYTHONPATH="${SCRIPT_DIR}" python3 "$TRANSLATOR_PY" \
        --source "$src_plat" \
        --target "$dst_plat" \
        --file_name "$src_file" \
        --max_depth 2 \
        --num_simulations 4 > /tmp/falcon_trans.log 2>&1; then

        echo "   ✅ Success"
    else
        echo "   ❌ Failed"
        echo "--- Error Log (Last 20 lines) ---"
        tail -n 20 /tmp/falcon_trans.log
        echo "---------------------------------"
    fi
    printf "\n"
done

echo "=== All Done ==="
