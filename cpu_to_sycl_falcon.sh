#!/usr/bin/env bash
set -euo pipefail

# Resolve the repository root relative to this script so it can be run
# from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Python interpreter, can be overridden by environment
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Python transcompiler entry point
TRANSLATOR_PY="${SCRIPT_DIR}/falcon/mcts/transcompile.py"

# Benchmark data root directory
BENCH_DIR="${SCRIPT_DIR}/benchmark/data"

# Transcompiler parameters (can be overridden by environment)
MAX_DEPTH="${MAX_DEPTH:-5}"
NUM_SIMULATIONS="${NUM_SIMULATIONS:-5}"

# Directories for logs and outputs (can be overridden)
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/tmp/cpu_to_sycl_logs}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/cpu_sycl}"

# CPU → SYCL transcompilation direction
SRC_PLAT="cpu"
DST_PLAT="sycl"

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

# Create output and log directories
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

echo
echo "=========================================="
echo "=== Pipeline: $SRC_PLAT -> $DST_PLAT ==="
echo "=========================================="
echo "Source file   : $src_file"
echo "Output dir    : $OUTPUT_DIR"
echo "Log dir       : $LOG_DIR"
echo "Max depth     : $MAX_DEPTH"
echo "Simulations   : $NUM_SIMULATIONS"
echo

filename=$(basename "$src_file")
stem="${filename%.cpp}"          # remove extension for log naming
log_file="${LOG_DIR}/${stem}.log"

printf "   Translating %s ... " "$filename"

# Run the transcompiler with PYTHONPATH set so benchmark/ is importable
if PYTHONPATH="$SCRIPT_DIR" "$PYTHON_BIN" "$TRANSLATOR_PY" \
    --source "$SRC_PLAT" \
    --target "$DST_PLAT" \
    --file_name "$src_file" \
    --max_depth "$MAX_DEPTH" \
    --num_simulations "$NUM_SIMULATIONS" > "$log_file" 2>&1; then
    echo "✅ Success"
    success_count=1
    failure_count=0
else
    echo "❌ Failed"
    success_count=0
    failure_count=1
    echo "      log: $log_file"
    echo "      last 20 lines:"
    tail -n 20 "$log_file" 2>/dev/null || echo "      (log file empty or not created)"
fi

echo
echo "=========================================="
echo "Completed: $success_count succeeded, $failure_count failed"
echo "Generated CPU → SYCL candidates are expected under: $OUTPUT_DIR"
echo "Detailed logs are under: $LOG_DIR"
echo "=========================================="

# 显示输出目录内容
if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ]; then
    echo
    echo "Output files generated:"
    ls -la "$OUTPUT_DIR"
fi

echo "=== All Done ==="