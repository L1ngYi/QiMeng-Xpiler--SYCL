#!/usr/bin/env bash
set -euo pipefail

# 脚本所在目录（仓库根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Python 解释器（可覆盖）
PYTHON_BIN="${PYTHON_BIN:-python3}"

# 转编译器入口
TRANSLATOR_PY="${SCRIPT_DIR}/falcon/mcts/transcompile.py"

# Benchmark 数据根目录
BENCH_DIR="${SCRIPT_DIR}/benchmark/data"

# 转编译参数（可覆盖）
MAX_DEPTH="${MAX_DEPTH:-5}"
NUM_SIMULATIONS="${NUM_SIMULATIONS:-5}"

# 日志与输出目录（可覆盖）
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/tmp/sycl_to_cpu_logs}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/sycl_cpu}"

# 要处理的转译方向（可自行增减）
DIRECTIONS=(
    "sycl:cpu"
)

# 创建输出和日志目录
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

# 检查转译器是否存在
if [ ! -f "$TRANSLATOR_PY" ]; then
    echo "[ERROR] Translator entry not found: $TRANSLATOR_PY"
    exit 1
fi

for dir_pair in "${DIRECTIONS[@]}"; do
    src_plat="${dir_pair%%:*}"
    dst_plat="${dir_pair##*:}"

    # 源目录路径
    src_dir="$BENCH_DIR/${src_plat}_code_test"

    if [ ! -d "$src_dir" ]; then
        echo "[ERROR] Source directory not found: $src_dir"
        exit 1
    fi

    # 获取所有 .cpp 文件（安全处理通配符）
    shopt -s nullglob
    files=("$src_dir"/*.cpp)
    shopt -u nullglob

    if [ ${#files[@]} -eq 0 ]; then
        echo "[ERROR] No .cpp files found in: $src_dir"
        exit 1
    fi

    echo
    echo "=========================================="
    echo "=== Pipeline: $src_plat -> $dst_plat ==="
    echo "=========================================="
    echo "Source dir   : $src_dir"
    echo "Output dir   : $OUTPUT_DIR"
    echo "Log dir      : $LOG_DIR"
    echo "Max depth    : $MAX_DEPTH"
    echo "Simulations  : $NUM_SIMULATIONS"
    echo

    success_count=0
    failure_count=0
    total_count=${#files[@]}

    for ((i = 0; i < total_count; ++i)); do
        src_file="${files[$i]}"
        filename="$(basename "$src_file")"
        stem="${filename%.cpp}"                     # 去掉 .cpp 后缀用于日志名
        log_file="${LOG_DIR}/${stem}.log"

        printf "   [%3d/%3d] Translating %-28s ... " "$((i + 1))" "$total_count" "$filename"

        # 运行转编译器
        if PYTHONPATH="$SCRIPT_DIR" "$PYTHON_BIN" "$TRANSLATOR_PY" \
            --source "$src_plat" \
            --target "$dst_plat" \
            --file_name "$src_file" \
            --max_depth "$MAX_DEPTH" \
            --num_simulations "$NUM_SIMULATIONS" > "$log_file" 2>&1; then
            echo "✅ Success"
            success_count=$((success_count + 1))
        else
            echo "❌ Failed"
            failure_count=$((failure_count + 1))
            echo "      log: $log_file"
            echo "      last 20 lines:"
            tail -n 20 "$log_file" 2>/dev/null || echo "      (log file empty or not created)"
        fi
    done

    echo
    echo "=========================================="
    echo "Completed: $success_count succeeded, $failure_count failed, total $total_count"
    echo "Generated $src_plat → $dst_plat candidates are expected under: $OUTPUT_DIR"
    echo "Detailed logs are under: $LOG_DIR"
    echo "=========================================="
done

# 显示输出目录内容
if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ]; then
    echo
    echo "Output files generated:"
    ls -la "$OUTPUT_DIR"
fi

echo "=== All Done ==="