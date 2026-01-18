source env.sh

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
# 检查 sycl-ls 命令是否存在 或者 检查 icpx 编译器是否存在
if command -v sycl-ls >/dev/null 2>&1 || command -v icpx >/dev/null 2>&1; then
    echo "==============SYCL Environment Detected==============="
    
    echo "==============SYCL Compilation Test==============="
    python benchmark/evaluation/sycl_test/compilation.py benchmark/data/sycl_code_test
    
    echo "==============SYCL Computation Test==============="
    python benchmark/evaluation/sycl_test/result_test.py benchmark/data/sycl_code_test benchmark/evaluation/sycl_test/
else
    echo "No SYCL environment (sycl-ls/icpx) detected, skipping SYCL tests."
fi