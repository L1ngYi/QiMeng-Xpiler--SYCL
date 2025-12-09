# NVIDIA GPU → CPU DL Boost

CUDA_TO_CPU_PROMPT = """
You are an expert in low-level deep learning compiler optimization.

## Task:
Translate the following CUDA kernel into optimized CPU code for AVX VNNI or scalar instructions.

## Constraints:
- Maintain numerical correctness.
- Target integer operations (e.g., int8, uint8 with dot-product accumulation).
- Use AVX VNNI intrinsics (like `_mm512_dpbusd_epi32`) or scalar integer code when applicable.
- Avoid floating point AVX (e.g., `_mm256_add_ps`) unless explicitly required.
- Avoid OpenMP or thread-level parallelism.
- Keep code self-contained with includes and comments.

---

### Example 1

**Input CUDA code:**
```cpp
extern "C" __global__ void dot_add(const uint8_t* A, const int8_t* B, int* C) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < 16) {
    int acc = 0;
    for (int j = 0; j < 64; ++j) {
      acc += A[i * 64 + j] * B[i * 64 + j];
    }
    C[i] = acc;
  }
}
```

**Output CPU code (with AVX VNNI):**
```cpp
#include <immintrin.h>
#include <stdint.h>
#include <string.h>

extern "C" void dot_add(const uint8_t* A, const int8_t* B, int* C) {
  for (int i = 0; i < 16; ++i) {
    __m512i acc = _mm512_setzero_si512();
    for (int j = 0; j < 64; j += 64) {
      __m512i va = _mm512_loadu_si512((__m512i*)(A + i * 64 + j));
      __m512i vb = _mm512_loadu_si512((__m512i*)(B + i * 64 + j));
      acc = _mm512_dpbusd_epi32(acc, va, vb);
    }
    _mm512_storeu_si512((__m512i*)&C[i], acc); // or scalar extract + store
  }
}
```
---

### Example 2

**Input CUDA code:**
```cpp
extern "C" __global__ void add_kernel(const int* A, const int* B, int* C) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < 32) {
    C[idx] = A[idx] + B[idx];
  }
}
```

**Output CPU code (with scalar ops):**
```cpp
#include <stdint.h>

extern "C" void add_kernel(const int* A, const int* B, int* C) {
  for (int i = 0; i < 32; ++i) {
    C[i] = A[i] + B[i];
  }
}
```

---

Now, translate the following CUDA code into AVX VNNI or scalar CPU code:

```cpp
{input_code}
```

Generate the complete and optimized CPU code:
"""

# NVIDIA GPU → AMD GPU (HIP)

CUDA_TO_AMD_PROMPT = """
You are an expert in GPU compiler optimization and cross-platform GPU kernel translation.

## Task:
Translate the following CUDA kernel into HIP for AMD GPUs.

## Constraints:
- Ensure functional correctness and maintain the same parallel structure.
- Convert CUDA-specific APIs (e.g., thread/block indexing, memory access, intrinsics) to their HIP equivalents.
- Ensure all headers and kernel launch configurations are valid under HIP.
- Do not include device query or runtime setup; focus on the kernel and launch syntax.

---

### Example 1

**Input CUDA code:**
```cpp
extern "C" __global__ void add_kernel(const float* A, const float* B, float* C) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < 1024) {
    C[idx] = A[idx] + B[idx];
  }
}
```

**Output HIP code:**
```cpp
#include <hip/hip_runtime.h>

extern "C" __global__ void add_kernel(const float* A, const float* B, float* C) {
  int idx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  if (idx < 1024) {
    C[idx] = A[idx] + B[idx];
  }
}
```

---

### Example 2

**Input CUDA code:**
```cpp
extern "C" __global__ void scale(const float* A, float* B, float alpha) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < 256) {
    B[tid] = alpha * A[tid];
  }
}
```

**Output HIP code:**
```cpp
#include <hip/hip_runtime.h>

extern "C" __global__ void scale(const float* A, float* B, float alpha) {
  int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  if (tid < 256) {
    B[tid] = alpha * A[tid];
  }
}
```

---

Now, translate the following CUDA code into HIP:

```cpp
{input_code}
```

Generate the complete and correct HIP kernel:
"""

# AMD GPU → CPU DL Boost

HIP_TO_CPU_PROMPT = """
You are an expert in low-level deep learning compiler optimization.

## Task:
Translate the following HIP kernel for AMD GPU into optimized CPU code using AVX VNNI or scalar instructions.

## Constraints:
- Maintain numerical correctness.
- Focus on integer operations when possible (e.g., int8/uint8 with dot-product accumulation).
- Use AVX VNNI intrinsics like `_mm512_dpbusd_epi32` or scalar integer code.
- Avoid floating point AVX intrinsics (e.g., `_mm256_add_ps`) unless required.
- Avoid OpenMP or thread-level parallelism.
- Keep code self-contained with includes and comments.

---

### Example 1

**Input HIP code:**
```cpp
#include <hip/hip_runtime.h>

extern "C" __global__ void dot_add(const uint8_t* A, const int8_t* B, int* C) {
  int i = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  if (i < 16) {
    int acc = 0;
    for (int j = 0; j < 64; ++j) {
      acc += A[i * 64 + j] * B[i * 64 + j];
    }
    C[i] = acc;
  }
}
```

**Output CPU code (with AVX VNNI):**
```cpp
#include <immintrin.h>
#include <stdint.h>
#include <string.h>

extern "C" void dot_add(const uint8_t* A, const int8_t* B, int* C) {
  for (int i = 0; i < 16; ++i) {
    __m512i acc = _mm512_setzero_si512();
    for (int j = 0; j < 64; j += 64) {
      __m512i va = _mm512_loadu_si512((__m512i*)(A + i * 64 + j));
      __m512i vb = _mm512_loadu_si512((__m512i*)(B + i * 64 + j));
      acc = _mm512_dpbusd_epi32(acc, va, vb);
    }
    _mm512_storeu_si512((__m512i*)&C[i], acc); // optional scalar store
  }
}
```

---

### Example 2

**Input HIP code:**
```cpp
#include <hip/hip_runtime.h>

extern "C" __global__ void add_kernel(const int* A, const int* B, int* C) {
  int idx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  if (idx < 32) {
    C[idx] = A[idx] + B[idx];
  }
}
```

**Output CPU code (scalar):**
```cpp
#include <stdint.h>

extern "C" void add_kernel(const int* A, const int* B, int* C) {
  for (int i = 0; i < 32; ++i) {
    C[i] = A[i] + B[i];
  }
}
```

---

Now, translate the following HIP kernel into optimized CPU code using AVX VNNI or scalar instructions:

```cpp
{input_code}
```

Generate the complete and optimized CPU implementation:
"""

# AMD GPU → NVIDIA GPU

HIP_TO_CUDA_PROMPT = """
You are an expert in GPU kernel development and cross-platform code translation.

## Task:
Translate the following HIP kernel code (AMD GPU) into CUDA kernel code (NVIDIA GPU).

## Constraints:
- Match functionality and numerical correctness.
- Translate HIP-specific APIs (e.g., `hipThreadIdx_x`) into their CUDA equivalents (e.g., `threadIdx.x`).
- Replace HIP headers with CUDA headers.
- Maintain the kernel launch structure and semantics.
- Output self-contained, compilable CUDA code with includes and comments.

---

### Example 1

**Input HIP code:**
```cpp
#include <hip/hip_runtime.h>

extern "C" __global__ void vector_add(const float* A, const float* B, float* C, int N) {
  int idx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  if (idx < N) {
    C[idx] = A[idx] + B[idx];
  }
}
```

**Output CUDA code:**
```cpp
#include <cuda_runtime.h>

extern "C" __global__ void vector_add(const float* A, const float* B, float* C, int N) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    C[idx] = A[idx] + B[idx];
  }
}
```

---

### Example 2

**Input HIP code:**
```cpp
#include <hip/hip_runtime.h>

__global__ void scale_kernel(float* data, float scale, int size) {
  int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  if (idx < size) {
    data[idx] *= scale;
  }
}
```

**Output CUDA code:**
```cpp
#include <cuda_runtime.h>

__global__ void scale_kernel(float* data, float scale, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] *= scale;
  }
}
```

---

Now, translate the following HIP kernel into CUDA kernel code:

```cpp
{input_code}
```

Generate the complete and correct CUDA kernel code:
"""

# CPU DLBoost → NVIDIA GPU

CPU_TO_CUDA_PROMPT = """
You are an expert in GPU kernel development and compiler backend translation.

## Task:
Translate the following CPU kernel code (using scalar operations or AVX VNNI intrinsics) into CUDA kernel code for NVIDIA GPUs.

## Constraints:
- Match functionality and numerical accuracy.
- Utilize GPU parallelism using threads (`threadIdx`, `blockIdx`) and avoid scalar-only code.
- Do not use AVX or CPU-specific intrinsics.
- Ensure the CUDA code is self-contained and ready to compile, with includes and comments.

---

### Example 1

**Input CPU code (scalar):**
```cpp
#include <stdint.h>

extern "C" void add_kernel(const int* A, const int* B, int* C) {
  for (int i = 0; i < 32; ++i) {
    C[i] = A[i] + B[i];
  }
}
```

**Output CUDA code:**
```cpp
#include <cuda_runtime.h>

__global__ void add_kernel(const int* A, const int* B, int* C) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < 32) {
    C[idx] = A[idx] + B[idx];
  }
}
```

---

### Example 2

**Input CPU code (AVX VNNI):**
```cpp
#include <immintrin.h>
#include <stdint.h>

extern "C" void dot_add(const uint8_t* A, const int8_t* B, int* C) {
  for (int i = 0; i < 16; ++i) {
    __m512i acc = _mm512_setzero_si512();
    for (int j = 0; j < 64; j += 64) {
      __m512i va = _mm512_loadu_si512((__m512i*)(A + i * 64 + j));
      __m512i vb = _mm512_loadu_si512((__m512i*)(B + i * 64 + j));
      acc = _mm512_dpbusd_epi32(acc, va, vb);
    }
    C[i] = ((int*)&acc)[0]; // simplified scalar store
  }
}
```

**Output CUDA code:**
```cpp
#include <cuda_runtime.h>

__global__ void dot_add(const uint8_t* A, const int8_t* B, int* C) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < 16) {
    int acc = 0;
    for (int j = 0; j < 64; ++j) {
      acc += A[i * 64 + j] * B[i * 64 + j];
    }
    C[i] = acc;
  }
}
```

---

Now, translate the following DLBoost CPU code into a correct and complete CUDA kernel:

```cpp
{input_code}
```

Generate the full CUDA implementation:
"""

# CPU DLBoost → AMD GPU (HIP)

CPU_TO_HIP_PROMPT = """
You are an expert in GPU kernel optimization and system-level code translation.

## Task:
Translate the following CPU kernel (scalar or AVX VNNI) code into an equivalent AMD GPU kernel using HIP.

## Constraints:
- Maintain numerical correctness and functional behavior.
- Use HIP GPU thread parallelism (e.g., `hipThreadIdx_x`, `hipBlockIdx_x`, etc.).
- Avoid CPU-specific intrinsics like AVX, SSE, or VNNI.
- Output must be complete HIP code with includes and comments.

---

### Example 1

**Input CPU code (scalar):**
```cpp
#include <stdint.h>

extern "C" void add_kernel(const int* A, const int* B, int* C) {
  for (int i = 0; i < 32; ++i) {
    C[i] = A[i] + B[i];
  }
}
```

**Output HIP code:**
```cpp
#include <hip/hip_runtime.h>

__global__ void add_kernel(const int* A, const int* B, int* C) {
  int idx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  if (idx < 32) {
    C[idx] = A[idx] + B[idx];
  }
}
```

---

### Example 2

**Input CPU code (AVX VNNI style, simplified):**
```cpp
#include <immintrin.h>
#include <stdint.h>

extern "C" void dot_add(const uint8_t* A, const int8_t* B, int* C) {
  for (int i = 0; i < 16; ++i) {
    int acc = 0;
    for (int j = 0; j < 64; ++j) {
      acc += A[i * 64 + j] * B[i * 64 + j];
    }
    C[i] = acc;
  }
}
```

**Output HIP code:**
```cpp
#include <hip/hip_runtime.h>
#include <stdint.h>

__global__ void dot_add(const uint8_t* A, const int8_t* B, int* C) {
  int i = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  if (i < 16) {
    int acc = 0;
    for (int j = 0; j < 64; ++j) {
      acc += A[i * 64 + j] * B[i * 64 + j];
    }
    C[i] = acc;
  }
}
```

---

Now translate the following CPU DLBoost code into an equivalent HIP kernel for AMD GPUs:

```cpp
{input_code}
```

Generate the full HIP implementation:
"""