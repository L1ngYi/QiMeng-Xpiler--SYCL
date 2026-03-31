This directory contains generated GEMM baselines for the SYCL adaptation
paper experiments.

- CUDA files use a plain FP32 shared-memory tiled GEMM.
- CPU files use the matching naive triple-loop GEMM.
- File names encode the intended benchmark shape as gemm_M_K_N.
- The code itself keeps (m, k, n) as runtime scalar parameters so the same
  source stays migration-friendly for direct-tool baselines such as DPC++.
