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



LOOP_RECOVERY_PROMPT_SYCL = """
SYCL Loop recovery

Function Overview:
Loop recovery is designed to convert SYCL C++ code that utilizes `queue.submit`, `handler.parallel_for`, and Kernel Lambdas
into standard sequential C++ code with explicit `for` loop structures. The goal is to strip away the SYCL runtime boilerplate (Host/Device separation) and extract the core computation logic into a sequential format.

Application Scenario:
- Use this prompt when you need to transform a SYCL Single-source program into a plain C++ function. This is essential for analyzing the algorithm logic on a CPU, debugging without a SYCL runtime, or preparing the code as a Unified IR for further source-to-source translation.

### Input:
A SYCL C++ function that contains a `q.submit` block and a `parallel_for` kernel invocation. The kernel is typically defined as a Lambda expression capturing external variables.

### Output:
A standard C++ function where the parallel execution logic (defined by `range` or `nd_range`) is replaced by sequential `for` loops, and the kernel body is placed inside these loops.

### Steps for Conversion:
1. Identify the `parallel_for` kernel launch. Extract the execution range (e.g., `range<1>(N)` or `nd_range<3>(...)`).
2. Identify captured variables in the Lambda (e.g., pointers like `float *A`, `accessor` or scalars like `int N`) and promote them to function arguments of the new C++ function.
3. Replace the SYCL parallel structure with nested `for` loops.
   - If `range<1>(N)` is used: Create one loop from `0` to `N`.
   - If `nd_range` is used: Mimic the Group/Local structure if possible, or flatten it to global loops based on the logic.
4. Replace SYCL indexing methods with loop iterators:
   - `item.get_global_id(0)` or `idx[0]` becomes the loop iterator variable (e.g., `i`).
   - `item.get_local_id(0)` becomes the inner loop iterator if applicable.
5. Remove all SYCL-specific syntax: `queue`, `handler`, `submit`, `accessor`, `[=]`, `id<1>`, etc.
6. Ensure memory accesses via Accessors are converted to standard pointer indexing.

### GPT Task:
Transform the following SYCL code into equivalent sequential C++ code. The output should be a standalone C++ function that performs the same computation using standard loops.
"""

LOOP_RECOVERY_DEMO_SYCL = """
Example:

#### SYCL Code:

```cpp
void vector_add(queue &q, float *A, float *B, float *C, int N) {
    q.submit([&](handler &h) {
        // Parallel execution over range N
        h.parallel_for(range<1>(N), [=](id<1> idx) {
            int i = idx[0];
            if (i < N) {
                C[i] = A[i] + B[i];
            }
        });
    });
    q.wait();
}

```

#### Desired C++ Output:

```cpp
void vector_add(float* A, float* B, float* C, int N) {
    // Reconstructed sequential loop based on SYCL range<1>(N)
    for (int i = 0; i < N; ++i) {
        if (i < N) {
            C[i] = A[i] + B[i];
        }
    }
}

```

Example 2 (2D Matrix Add):

#### SYCL Code:

```cpp
void mat_add(queue &q, float *A, float *B, float *C, int M, int N) {
    q.submit([&](handler &h) {
        h.parallel_for(range<2>(M, N), [=](item<2> item) {
            int r = item.get_global_id(0);
            int c = item.get_global_id(1);
            int index = r * N + c;
            C[index] = A[index] + B[index];
        });
    });
}

```

#### Desired C++ Output:

```cpp
void mat_add(float* A, float* B, float* C, int M, int N) {
    // Reconstructed nested loops based on SYCL range<2>(M, N)
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            int index = r * N + c;
            C[index] = A[index] + B[index];
        }
    }
}

```

"""
