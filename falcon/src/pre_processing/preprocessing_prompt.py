LOOP_RECOVERY_PROMPT_CUDA = """
Loop recovery

Function Overview:
Loop recovery is designed to convert CUDA C code that utilizes GPU thread indexing (using `threadIdx`, `blockIdx`, `blockDim`, and `gridDim`)
into standard C++ code with `for` loop structures. The goal is to remove CUDA-specific parallelism while preserving the logical flow and structure of the code.

Application Scenario:
- Use this prompt when you want to convert CUDA C code into sequential C++ code, either for environments without GPU support or to analyze/debug the logic in a CPU-based system. This transformation is useful for simplifying GPU code or porting it to CPU environments.

### Input:
A CUDA C kernel function that uses `threadIdx`, `blockIdx`, `blockDim`, and `gridDim` to define parallel execution on a GPU.

### Output:
The same logic rewritten using standard C++ for loops to emulate the behavior of thread and block indexing in a sequential CPU-based program.

### Steps for Conversion:
1. Identify the use of `blockIdx`, `threadIdx`, `blockDim`, and `gridDim` in the input CUDA code.
2. Convert the parallel structure into nested `for` loops in standard C++.
3. Replace the GPU thread index expressions with loop index variables (e.g., `blockIdx.x * blockDim.x + threadIdx.x` becomes a C++ `for` loop with the same arithmetic).
4. Ensure that all CUDA-specific syntax, such as `__global__` and `__device__`, is removed or replaced.
5. Maximum number of threads per block is 1024, and the maximum number of blocks per grid is 256.

### GPT Task:
Transform the following CUDA C code into equivalent C++ for loop code that sequentially emulates the CUDA threading structure. The output should use nested `for` loops to replace CUDA thread indexing.
"""

LOOP_RECOVERY_DEMO_CUDA = """
Example:

#### CUDA C Code:

```cuda
extern "C" __global__ void vector_add(float* A, float* B, float* C, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        C[index] = A[index] + B[index];
    }
}
```

#### Desired C++ Output:

```cpp
void vector_add(float* A, float* B, float* C, int N) {
    // Loop over all blocks
    for (int blockIdx = 0; blockIdx < 256; ++blockIdx) {
        // Loop over all threads in each block
        for (int threadIdx = 0; threadIdx < 1024; ++threadIdx) {
            int index = blockIdx * 1024 + threadIdx;
            if (index < N) {
                C[index] = A[index] + B[index];
            }
        }
    }
}
```
"""