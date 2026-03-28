source env.sh

load_sycl_env() {
    if [ -n "${SYCL_ENV_SCRIPT:-}" ] && [ -f "${SYCL_ENV_SCRIPT}" ]; then
        source "${SYCL_ENV_SCRIPT}"
        return 0
    fi

    if [ -f "./env_sycl.sh" ]; then
        source "./env_sycl.sh"
        return 0
    fi

    if [ -f "$HOME/env_sycl.sh" ]; then
        source "$HOME/env_sycl.sh"
        return 0
    fi

    return 1
}

echo "Running CPP tests..."

echo "==============CPP Compilation Test==============="
python benchmark/evaluation/cpu_test/compilation.py benchmark/data/cpp_code_test
echo "==============CPP Computation Test==============="
python benchmark/evaluation/cpu_test/result_test.py benchmark/data/cpp_code_test benchmark/evaluation/cpu_test/
# # Check for NVIDIA GPU presence
#寒武纪
if cnmon >/dev/null 2>&1; then

	echo "==============DL Boost Compilation Test==============="
	python benchmark/evaluation/dlboost_test/compilation.py
	echo "==============DL Boost Computation Test==============="
	python benchmark/evaluation/dlboost_test/result_test.py
fi
# Check for NVIDIA GPU presence
if nvidia-smi >/dev/null 2>&1; then
	# echo "NVIDIA GPU detected. Running CUDA tests..."

	echo "==============GPU Compilation Test==============="
	python benchmark/evaluation/cuda_test/compilation.py benchmark/data/cuda_code_test
	echo "==============GPU Computation Test==============="
	python benchmark/evaluation/cuda_test/result_test.py benchmark/data/cuda_code_test benchmark/evaluation/cuda_org_test/
fi
# 3. [新增] SYCL 测试
SYCL_ENV_READY=0
if load_sycl_env; then
    SYCL_ENV_READY=1
fi

# 检查是否已显式加载 SYCL 环境，或者是否存在可用的 SYCL 工具链
if command -v sycl-ls >/dev/null 2>&1 || command -v icpx >/dev/null 2>&1 || [ "$SYCL_ENV_READY" -eq 1 ] || [ -n "${SYCL_COMPILER:-}" ]; then
    echo "==============SYCL Environment Detected==============="
    
    echo "==============SYCL Compilation Test==============="
    python benchmark/evaluation/sycl_test/compilation.py benchmark/data/sycl_code_test
    
    echo "==============SYCL Computation Test==============="
    python benchmark/evaluation/sycl_test/result_test.py benchmark/data/sycl_code_test benchmark/evaluation/sycl_test/
else
    echo "No SYCL environment detected, skipping SYCL tests."
fi
