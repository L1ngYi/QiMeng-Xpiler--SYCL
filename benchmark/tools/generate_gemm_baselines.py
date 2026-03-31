#!/usr/bin/env python3
"""
Generate a matched GEMM baseline suite for CUDA and CPU experiments.

The generated CUDA implementation is a plain FP32 tiled GEMM that uses
shared-memory blocking but avoids Tensor Core / WMMA intrinsics. The CPU
implementation is the corresponding naive triple-loop version.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from textwrap import dedent


DEFAULT_CASES = [
    ("square", 128, 128, 128),
    ("square", 256, 256, 256),
    ("square", 512, 512, 512),
    ("square", 1024, 1024, 1024),
    ("square", 2048, 2048, 2048),
    ("small_k", 1024, 16, 1024),
    ("small_k", 1024, 32, 1024),
    ("small_k", 1024, 64, 1024),
    ("small_k", 2048, 32, 2048),
    ("small_k", 2048, 64, 2048),
    ("wide_n", 256, 256, 2048),
    ("wide_n", 512, 256, 4096),
    ("wide_n", 1024, 256, 4096),
    ("wide_n", 2048, 256, 4096),
    ("wide_n", 4096, 128, 4096),
    ("tall_m", 2048, 256, 256),
    ("tall_m", 4096, 256, 256),
    ("tall_m", 4096, 512, 256),
    ("tall_m", 4096, 1024, 256),
    ("tall_m", 4096, 1024, 512),
]


def _parse_shape(shape_spec: str) -> tuple[str, int, int, int]:
    parts = [part.strip() for part in shape_spec.split(",")]
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError(
            "Shape must be M,K,N or category,M,K,N."
        )

    if len(parts) == 3:
        category = "custom"
        dims = parts
    else:
        category = parts[0]
        dims = parts[1:]

    try:
        m_dim, k_dim, n_dim = (int(dim) for dim in dims)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Shape dimensions must be integers."
        ) from exc

    return category, m_dim, k_dim, n_dim


CUDA_TEMPLATE = dedent(
    """\
    #include <cuda_runtime.h>

    namespace {
    constexpr int TILE = 16;
    }

    __global__ void gemm(const float *A, const float *B, float *C, int m, int k, int n) {
      __shared__ float tile_a[TILE][TILE];
      __shared__ float tile_b[TILE][TILE];

      int row = blockIdx.y * TILE + threadIdx.y;
      int col = blockIdx.x * TILE + threadIdx.x;
      float sum = 0.0f;

      for (int tile_k = 0; tile_k < k; tile_k += TILE) {
        int a_col = tile_k + threadIdx.x;
        int b_row = tile_k + threadIdx.y;

        tile_a[threadIdx.y][threadIdx.x] =
            (row < m && a_col < k) ? A[row * k + a_col] : 0.0f;
        tile_b[threadIdx.y][threadIdx.x] =
            (b_row < k && col < n) ? B[b_row * n + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
          sum += tile_a[threadIdx.y][kk] * tile_b[kk][threadIdx.x];
        }

        __syncthreads();
      }

      if (row < m && col < n) {
        C[row * n + col] = sum;
      }
    }

    extern "C" void gemm_kernel(float *A, float *B, float *C, int m, int k, int n) {
      float *d_A;
      float *d_B;
      float *d_C;

      cudaMalloc(&d_A, static_cast<size_t>(m) * k * sizeof(float));
      cudaMalloc(&d_B, static_cast<size_t>(k) * n * sizeof(float));
      cudaMalloc(&d_C, static_cast<size_t>(m) * n * sizeof(float));

      cudaMemcpy(d_A, A, static_cast<size_t>(m) * k * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_B, B, static_cast<size_t>(k) * n * sizeof(float), cudaMemcpyHostToDevice);

      dim3 block_size(TILE, TILE);
      dim3 grid_size((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
      gemm<<<grid_size, block_size>>>(d_A, d_B, d_C, m, k, n);

      cudaMemcpy(C, d_C, static_cast<size_t>(m) * n * sizeof(float), cudaMemcpyDeviceToHost);

      cudaFree(d_A);
      cudaFree(d_B);
      cudaFree(d_C);
    }
    """
)


CPU_TEMPLATE = dedent(
    """\
    extern "C" void gemm(float *A, float *B, float *C, int m, int k, int n) {
      for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
          float sum = 0.0f;
          for (int inner = 0; inner < k; ++inner) {
            sum += A[row * k + inner] * B[inner * n + col];
          }
          C[row * n + col] = sum;
        }
      }
    }
    """
)


README_TEMPLATE = dedent(
    """\
    This directory contains generated GEMM baselines for the SYCL adaptation
    paper experiments.

    - CUDA files use a plain FP32 shared-memory tiled GEMM.
    - CPU files use the matching naive triple-loop GEMM.
    - File names encode the intended benchmark shape as gemm_M_K_N.
    - The code itself keeps (m, k, n) as runtime scalar parameters so the same
      source stays migration-friendly for direct-tool baselines such as DPC++.
    """
)


def _write_text(path: Path, content: str, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force to overwrite.")
    path.write_text(content, encoding="utf-8")


def _case_name(m: int, k: int, n: int) -> str:
    return f"gemm_{m}_{k}_{n}"


def generate_cases(repo_root: Path, cases: list[tuple[str, int, int, int]], force: bool) -> None:
    cuda_dir = repo_root / "benchmark" / "data" / "cuda_baseline_gemm"
    cpu_dir = repo_root / "benchmark" / "data" / "cpp_baseline_gemm"
    cuda_dir.mkdir(parents=True, exist_ok=True)
    cpu_dir.mkdir(parents=True, exist_ok=True)

    _write_text(cuda_dir / "README.txt", README_TEMPLATE, force=force)
    _write_text(cpu_dir / "README.txt", README_TEMPLATE, force=force)

    manifest_rows = []
    for category, m_dim, k_dim, n_dim in cases:
        stem = _case_name(m_dim, k_dim, n_dim)
        cuda_path = cuda_dir / f"{stem}.cu"
        cpu_path = cpu_dir / f"{stem}.cpp"
        _write_text(cuda_path, CUDA_TEMPLATE, force=force)
        _write_text(cpu_path, CPU_TEMPLATE, force=force)
        manifest_rows.append(
            {
                "category": category,
                "m": m_dim,
                "k": k_dim,
                "n": n_dim,
                "cuda_file": str(cuda_path.relative_to(repo_root)),
                "cpu_file": str(cpu_path.relative_to(repo_root)),
            }
        )

    manifest_path = repo_root / "benchmark" / "data" / "gemm_baseline_manifest.csv"
    if manifest_path.exists() and not force:
        raise FileExistsError(
            f"{manifest_path} already exists; pass --force to overwrite."
        )
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["category", "m", "k", "n", "cuda_file", "cpu_file"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=Path(__file__).resolve().parents[2],
        type=Path,
        help="Repository root used to place generated benchmark files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing generated files.",
    )
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        type=_parse_shape,
        help="Optional custom case in M,K,N or category,M,K,N form. Repeatable.",
    )
    args = parser.parse_args()

    cases = args.shape if args.shape else DEFAULT_CASES
    generate_cases(args.repo_root.resolve(), cases, force=args.force)


if __name__ == "__main__":
    main()
