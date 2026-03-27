#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRANSLATOR_PY="${SCRIPT_DIR}/falcon/mcts/transcompile.py"
CUDA_SRC_DIR="${SCRIPT_DIR}/benchmark/data/cuda_code_test"
OUTPUT_DIR="${SCRIPT_DIR}/cuda_sycl"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/tmp/cuda_to_sycl_logs}"
MAX_DEPTH="${MAX_DEPTH:-2}"
NUM_SIMULATIONS="${NUM_SIMULATIONS:-4}"

if [ ! -f "$TRANSLATOR_PY" ]; then
    echo "[ERROR] Translator entry not found: $TRANSLATOR_PY"
    exit 1
fi

if [ ! -d "$CUDA_SRC_DIR" ]; then
    echo "[ERROR] CUDA source directory not found: $CUDA_SRC_DIR"
    exit 1
fi

mkdir -p "$LOG_DIR"

shopt -s nullglob
files=("$CUDA_SRC_DIR"/gemm_32_32_128.cu)
shopt -u nullglob

if [ "${#files[@]}" -eq 0 ]; then
    echo "[ERROR] No gemm CUDA kernels found in: $CUDA_SRC_DIR"
    exit 1
fi

echo
echo "=========================================="
echo "=== Pipeline: cuda -> sycl (gemm only) ==="
echo "=========================================="
echo "Source dir : $CUDA_SRC_DIR"
echo "Output dir : $OUTPUT_DIR"
echo "Log dir    : $LOG_DIR"
echo "Max depth  : $MAX_DEPTH"
echo "Simulations: $NUM_SIMULATIONS"
echo

success_count=0
failure_count=0
total_count="${#files[@]}"

for ((i = 0; i < total_count; ++i)); do
    src_file="${files[$i]}"
    filename="$(basename "$src_file")"
    stem="${filename%.cu}"
    log_file="${LOG_DIR}/${stem}.log"

    printf "   [%3d/%3d] Translating %-28s ... " "$((i + 1))" "$total_count" "$filename"

    if PYTHONPATH="$SCRIPT_DIR" "$PYTHON_BIN" "$TRANSLATOR_PY" \
        --source cuda \
        --target sycl \
        --file_name "$src_file" \
        --max_depth "$MAX_DEPTH" \
        --num_simulations "$NUM_SIMULATIONS" >"$log_file" 2>&1; then
        echo "✅ Success"
        success_count=$((success_count + 1))
    else
        echo "❌ Failed"
        failure_count=$((failure_count + 1))
        echo "      log: $log_file"
        echo "      last 20 lines:"
        tail -n 20 "$log_file"
    fi
done

echo
echo "=========================================="
echo "Completed: $success_count succeeded, $failure_count failed, total $total_count"
echo "Generated SYCL candidates are expected under: $OUTPUT_DIR"
echo "Detailed logs are under: $LOG_DIR"
echo "=========================================="
