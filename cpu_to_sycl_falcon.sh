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

for dir_pair in "${DIRECTIONS[@]}"; do
    src_plat=${dir_pair%%:*}
    dst_plat=${dir_pair##*:}

    src_dir="$BENCH_DIR/${src_plat}_code_test"

    echo
    echo "=========================================="
    echo "=== Pipeline: $src_plat -> $dst_plat ==="
    echo "=========================================="

    # Collect all .cpp files; handle empty directories gracefully
    files=$(compgen -G "$src_dir"/*.cpp || echo "")

    if [ -z "$files" ]; then
        echo "   [WARN] No .cpp files found in $src_dir"
        continue
    fi

    i=0
    file_arr=($files)
    total=${#file_arr[@]}

    for src_file in $files; do
        ((i+=1))
        filename=$(basename "$src_file")

        printf "   [%3d/%3d] Translating %-30s ... " "$i" "$total" "$filename"

        # Run the transcompiler with PYTHONPATH set so benchmark/ is importable
        if PYTHONPATH="${SCRIPT_DIR}" python3 "$TRANSLATOR_PY" \
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
        fi
    done
    printf "\n"
done
echo "=== All Done ==="
