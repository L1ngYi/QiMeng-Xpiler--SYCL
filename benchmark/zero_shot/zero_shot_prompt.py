# NVIDIA GPU → CPU DL Boost

CUDA_TO_CPU_PROMPT = """
You are an expert in low-level deep learning compiler optimization.

Task:
Translate the following CUDA kernel into optimized CPU code using AVX VNNI intrinsics if possible.

Constraints:
- Match the numerical accuracy.
- Try to preserve parallelism using SIMD (e.g., AVX VNNI).
- Use intrinsics instead of OpenMP or naive loops.
- Keep code complete with includes, main function, and comments.

Input CUDA code:
```cpp
{input_code}
```
Now generate the equivalent optimized CPU code:
"""

# NVIDIA GPU → AMD GPU
CUDA_TO_AMD_PROMPT = """
You are an expert in GPU kernel development.

Task:
Translate the following CUDA kernel into HIP kernel code optimized for AMD GPUs.

Constraints:
- Ensure correctness and numerical accuracy.
- Preserve thread/block-level parallelism.
- Use HIP-specific APIs and syntax.
- Include necessary headers and kernel launch code.

Input CUDA code:
```cpp
{input_code}
```
Now generate the equivalent HIP kernel for AMD GPU: """

# AMD GPU → CPU DL Boost
HIP_TO_CPU_PROMPT = """
You are an expert in compiler optimization.

Task:
Convert the following HIP (AMD GPU) kernel into optimized CPU code using AVX VNNI intrinsics if applicable.

Constraints:
- Preserve numerical accuracy and data layout.
- Use SIMD intrinsics rather than simple scalar loops.
- Include full C++ code with headers, intrinsics, and comments.

Input HIP code:
```cpp
{input_code}
```
Now generate the equivalent MLU kernel code: """

# AMD GPU → NVIDIA GPU
HIP_TO_CUDA_PROMPT = """
You are a GPU programming expert.

Task:
Convert the following HIP (AMD GPU) kernel into equivalent CUDA kernel code for NVIDIA GPUs.

Constraints:
- Preserve thread hierarchy and parallelism.
- Match behavior and performance as much as possible.
- Replace HIP API with appropriate CUDA API.
- Include complete kernel function and launch setup.

Input HIP code:
```cpp
{input_code}
```
Now generate the equivalent MLU kernel code: """

# CPU DL Boost → NVIDIA GPU
CPU_TO_CUDA_PROMPT = """
You are a high-performance code generation expert.

Task:
Convert the following CPU code (with AVX intrinsics or scalar operations) into a CUDA kernel optimized for NVIDIA GPUs.

Constraints:
- Exploit thread-level parallelism using CUDA.
- Ensure correctness and similar performance characteristics.
- Include complete kernel function and necessary CUDA headers.

Input CPU code:
```cpp
{input_code}
```
Now generate the equivalent MLU kernel code: """

# CPU DL Boost → AMD GPU
CPU_TO_HIP_PROMPT = """
You are an expert in heterogeneous programming.

Task:
Translate the following AVX-accelerated CPU code into a HIP kernel targeting AMD GPUs.

Constraints:
- Match data layout and computation logic.
- Use HIP kernel launch structure.
- Preserve performance via parallel threads and memory coalescing.

Input CPU code:
```cpp
{input_code}
```
Now generate the equivalent MLU kernel code: """
