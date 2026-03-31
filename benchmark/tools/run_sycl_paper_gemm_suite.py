#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import ctypes
import math
import os
import re
import statistics
import tempfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from string import Template

import numpy as np

from benchmark.template.sycl_host_template import create_sycl_func
from benchmark.utils import (
    configure_sycl_environment,
    preload_sycl_runtime,
    run_cuda_compilation,
    run_sycl_compilation,
)


FAILURE_MS = 1_000_000.0
CATEGORY_TO_GROUP = {
    "square": "S",
    "small_k": "K",
    "wide_n": "N",
    "tall_m": "M",
}
GROUP_LABELS = {
    "S": "Square",
    "K": "Small-K",
    "N": "Wide-N",
    "M": "Tall-M",
    "A": "Advanced",
}
METHOD_DISPLAY = {
    "cuda": "CUDA-Tiled",
    "our_sycl": "Our-SYCL",
    "dpcpp": "DPC++",
}

CUDA_HARNESS_TEMPLATE = Template(
    """
#include <chrono>
#include <iostream>

${source_code}

extern "C" float timed_gemm_entry(float *A, float *B, float *C, int m, int k, int n) {
    for (int i = 0; i < ${warmup}; ++i) {
        ${entry_name}(A, B, C, m, k, n);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ${iters}; ++i) {
        ${entry_name}(A, B, C, m, k, n);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    cudaDeviceSynchronize();
    return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() /
           1000.0f / ${iters};
}
"""
)

SYCL_HARNESS_TEMPLATE = Template(
    """
#include <chrono>
#include <exception>
#include <iostream>

${source_code}

extern "C" float timed_gemm_entry(float *A, float *B, float *C, int m, int k, int n) {
    try {
        for (int i = 0; i < ${warmup}; ++i) {
            ${entry_name}(A, B, C, m, k, n);
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ${iters}; ++i) {
            ${entry_name}(A, B, C, m, k, n);
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() /
               1000.0f / ${iters};
    } catch (const std::exception &e) {
        std::cerr << "[SYCL Harness Error] " << e.what() << std::endl;
        return ${failure_ms};
    } catch (...) {
        std::cerr << "[SYCL Harness Error] unknown exception" << std::endl;
        return ${failure_ms};
    }
}
"""
)


@dataclass(frozen=True)
class CaseDef:
    case_id: str
    group: str
    category: str
    m: int
    k: int
    n: int
    stem: str
    cuda_file: Path
    cpu_file: Path


@dataclass
class BenchResult:
    method: str
    source_file: str = ""
    compile_success: bool = False
    runnable: bool = False
    correctness_pass: bool = False
    time_ms: float | None = None
    gflops: float | None = None
    peak_pct: float | None = None
    retention_pct: float | None = None
    timing_mode: str = "operator_latency"
    status: str = "missing"
    notes: list[str] = field(default_factory=list)
    log_file: str = ""

    @property
    def performance_eligible(self) -> bool:
        return self.compile_success and self.runnable and self.correctness_pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_results_dir() -> Path:
    return _repo_root() / "benchmark" / "results" / "sycl_paper"


def parse_args() -> argparse.Namespace:
    repo_root = _repo_root()
    results_dir = _default_results_dir()
    parser = argparse.ArgumentParser(
        description=(
            "Run the SYCL adaptation paper GEMM suite for CUDA, Our-SYCL and DPC++."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root / "benchmark" / "data" / "gemm_baseline_manifest.csv",
        help="CSV manifest describing the GEMM baseline cases.",
    )
    parser.add_argument(
        "--cuda-dir",
        type=Path,
        default=repo_root / "benchmark" / "data" / "cuda_baseline_gemm",
        help="Directory with CUDA baseline GEMM sources.",
    )
    parser.add_argument(
        "--our-sycl-dir",
        type=Path,
        default=repo_root / "cpu_sycl",
        help="Directory containing Falcon CPU->SYCL outputs.",
    )
    parser.add_argument(
        "--dpcpp-dir",
        type=Path,
        default=repo_root / "benchmark" / "data" / "dpcpp_baseline_gemm",
        help="Directory containing canonical dpct outputs.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=results_dir,
        help="Directory used to write CSV summaries and logs.",
    )
    parser.add_argument(
        "--peak-gflops",
        type=float,
        default=19.5 * 1000.0,
        help=(
            "Device FP32 peak throughput in GFLOPS. "
            "Default corresponds to an A100 FP32 peak of 19.5 TFLOPS."
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup invocations per benchmarked source.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=10,
        help="Measured invocations per benchmarked source.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance used for correctness checking.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance used for correctness checking.",
    )
    return parser.parse_args()


def _assign_case_id(group: str, group_counts: dict[str, int]) -> str:
    group_counts[group] = group_counts.get(group, 0) + 1
    return f"{group}{group_counts[group]}"


def _load_cases(
    manifest_path: Path, repo_root: Path, cuda_dir: Path, cpu_dir: Path
) -> list[CaseDef]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    cases: list[CaseDef] = []
    group_counts: dict[str, int] = {}
    with manifest_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            category = row["category"].strip()
            if category not in CATEGORY_TO_GROUP:
                continue
            group = CATEGORY_TO_GROUP[category]
            m_dim = int(row["m"])
            k_dim = int(row["k"])
            n_dim = int(row["n"])
            stem = f"gemm_{m_dim}_{k_dim}_{n_dim}"
            cuda_file = repo_root / row.get("cuda_file", "").strip()
            cpu_file = repo_root / row.get("cpu_file", "").strip()
            if not row.get("cuda_file"):
                cuda_file = cuda_dir / f"{stem}.cu"
            if not row.get("cpu_file"):
                cpu_file = cpu_dir / f"{stem}.cpp"
            cases.append(
                CaseDef(
                    case_id=_assign_case_id(group, group_counts),
                    group=group,
                    category=category,
                    m=m_dim,
                    k=k_dim,
                    n=n_dim,
                    stem=stem,
                    cuda_file=cuda_file.resolve(),
                    cpu_file=cpu_file.resolve(),
                )
            )
    return cases


def _seed_for_case(stem: str) -> int:
    return zlib.crc32(stem.encode("utf-8")) & 0xFFFFFFFF


def _make_inputs(case: CaseDef) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(_seed_for_case(case.stem))
    a_mat = rng.standard_normal((case.m, case.k)).astype(np.float32)
    b_mat = rng.standard_normal((case.k, case.n)).astype(np.float32)
    reference = a_mat @ b_mat
    return a_mat, b_mat, reference


def _extract_entry_name(source_text: str) -> str | None:
    patterns = [
        r'extern\s+"C"\s+void\s+(\w+_kernel)\s*\(([^)]*)\)',
        r"\bvoid\s+(\w+_kernel)\s*\(([^)]*)\)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, source_text, re.DOTALL):
            params = match.group(2)
            if "queue" in params:
                continue
            if params.count("*") < 3:
                continue
            return match.group(1)
    return None


def _canonical_dpcpp_source(dpcpp_dir: Path, stem: str) -> Path | None:
    for suffix in (".cpp", ".dp.cpp", ".cc", ".cxx"):
        candidate = dpcpp_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate.resolve()
    matches = sorted(
        path
        for path in dpcpp_dir.glob(f"{stem}*")
        if path.is_file() and path.suffix in {".cpp", ".cc", ".cxx"}
    )
    return matches[0].resolve() if matches else None


def _format_optional(value: float | None, digits: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def _gflops(case: CaseDef, time_ms: float) -> float:
    return 2.0 * case.m * case.k * case.n / (time_ms * 1e6)


def _compile_and_run_harness(
    *,
    source_text: str,
    entry_name: str,
    compile_kind: str,
    case: CaseDef,
    warmup: int,
    iters: int,
    rtol: float,
    atol: float,
    log_path: Path,
) -> BenchResult:
    result = BenchResult(method=compile_kind, source_file="", log_file=str(log_path))
    harness_template = (
        CUDA_HARNESS_TEMPLATE if compile_kind == "cuda" else SYCL_HARNESS_TEMPLATE
    )
    file_suffix = ".cu" if compile_kind == "cuda" else ".cpp"
    compile_func = run_cuda_compilation if compile_kind == "cuda" else run_sycl_compilation

    harness_source = harness_template.substitute(
        source_code=source_text,
        entry_name=entry_name,
        warmup=warmup,
        iters=iters,
        failure_ms=f"{FAILURE_MS:.1f}f",
    )

    a_mat, b_mat, reference = _make_inputs(case)
    c_mat = np.zeros((case.m, case.n), dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        src_path = tmpdir_path / f"gemm_harness{file_suffix}"
        so_path = tmpdir_path / "gemm_harness.so"
        src_path.write_text(harness_source, encoding="utf-8")

        success, output = compile_func(str(so_path), str(src_path))
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"[compile_kind] {compile_kind}\n")
            log_file.write(f"[entry_name] {entry_name}\n")
            log_file.write(f"[compile_success] {success}\n")
            if isinstance(output, str):
                log_file.write("[compiler_output]\n")
                log_file.write(output)
                if not output.endswith("\n"):
                    log_file.write("\n")

        result.compile_success = success
        if not success:
            result.status = "compile_failed"
            result.notes.append("Harness compilation failed.")
            return result

        if compile_kind != "cuda":
            configure_sycl_environment()
            preload_sycl_runtime()

        rtld_flag = getattr(os, "RTLD_GLOBAL", getattr(ctypes, "RTLD_GLOBAL", 0))
        try:
            lib = ctypes.CDLL(str(so_path), mode=rtld_flag)
            func = getattr(lib, "timed_gemm_entry")
            func.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            func.restype = ctypes.c_float

            elapsed_ms = float(
                func(
                    a_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    b_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    c_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    case.m,
                    case.k,
                    case.n,
                )
            )
        except Exception as exc:
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"[runtime_exception] {exc}\n")
            result.status = "runtime_failed"
            result.notes.append(str(exc))
            return result

    result.runnable = math.isfinite(elapsed_ms) and 0.0 < elapsed_ms < FAILURE_MS
    if not result.runnable:
        result.status = "runtime_failed"
        result.notes.append(f"Invalid elapsed time: {elapsed_ms}")
        return result

    result.correctness_pass = bool(np.allclose(c_mat, reference, rtol=rtol, atol=atol))
    result.time_ms = elapsed_ms
    result.gflops = _gflops(case, elapsed_ms)
    result.status = "ok" if result.correctness_pass else "incorrect"
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"[elapsed_ms] {elapsed_ms:.6f}\n")
        log_file.write(f"[correctness] {result.correctness_pass}\n")
        if not result.correctness_pass:
            max_abs_err = float(np.max(np.abs(c_mat - reference)))
            log_file.write(f"[max_abs_error] {max_abs_err:.6e}\n")
    return result


def _benchmark_cuda(
    case: CaseDef,
    cuda_file: Path,
    warmup: int,
    iters: int,
    rtol: float,
    atol: float,
    log_path: Path,
) -> BenchResult:
    result = BenchResult(method="cuda", source_file=str(cuda_file), log_file=str(log_path))
    if not cuda_file.is_file():
        result.status = "missing"
        result.notes.append("CUDA source file is missing.")
        return result
    source_text = cuda_file.read_text(encoding="utf-8")
    entry_name = _extract_entry_name(source_text)
    if not entry_name:
        result.status = "missing_entry"
        result.notes.append("Could not find a callable CUDA host entry.")
        return result
    bench = _compile_and_run_harness(
        source_text=source_text,
        entry_name=entry_name,
        compile_kind="cuda",
        case=case,
        warmup=warmup,
        iters=iters,
        rtol=rtol,
        atol=atol,
        log_path=log_path,
    )
    bench.source_file = result.source_file
    return bench


def _benchmark_sycl_host_entry(
    *,
    case: CaseDef,
    source_file: Path,
    warmup: int,
    iters: int,
    rtol: float,
    atol: float,
    log_path: Path,
    needs_wrap: bool,
) -> BenchResult:
    result = BenchResult(
        method="our_sycl" if needs_wrap else "dpcpp",
        source_file=str(source_file),
        log_file=str(log_path),
    )
    if not source_file.is_file():
        result.status = "missing"
        result.notes.append("SYCL source file is missing.")
        return result

    wrapped_path: Path | None = None
    try:
        effective_path = source_file
        if needs_wrap:
            wrapped_path = Path(
                create_sycl_func(str(source_file), op_type="matmul")
            ).resolve()
            effective_path = wrapped_path
        source_text = effective_path.read_text(encoding="utf-8")
        entry_name = _extract_entry_name(source_text)
        if not entry_name:
            result.status = "missing_entry"
            result.notes.append("Could not find a callable SYCL host entry.")
            return result
        bench = _compile_and_run_harness(
            source_text=source_text,
            entry_name=entry_name,
            compile_kind="sycl",
            case=case,
            warmup=warmup,
            iters=iters,
            rtol=rtol,
            atol=atol,
            log_path=log_path,
        )
        bench.method = result.method
        bench.source_file = result.source_file
        return bench
    finally:
        if wrapped_path and wrapped_path.exists():
            wrapped_path.unlink()


def _aggregate_translation_summary(
    cases: list[CaseDef], results: dict[tuple[str, str], BenchResult]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for group in ("S", "K", "N", "M", "A"):
        group_cases = [case for case in cases if case.group == group]
        total = len(group_cases)
        for method in ("dpcpp", "our_sycl"):
            method_cases = [results[(case.stem, method)] for case in group_cases]
            rows.append(
                {
                    "method": METHOD_DISPLAY[method],
                    "group": group,
                    "compile_success": str(sum(1 for item in method_cases if item.compile_success)),
                    "runnable": str(sum(1 for item in method_cases if item.runnable)),
                    "correctness_pass": str(
                        sum(1 for item in method_cases if item.correctness_pass)
                    ),
                    "total": str(total),
                }
            )
    return rows


def _median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _aggregate_perf_summary(
    cases: list[CaseDef], results: dict[tuple[str, str], BenchResult]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for method in ("cuda", "our_sycl", "dpcpp"):
        for group in ("S", "K", "N", "M"):
            group_cases = [case for case in cases if case.group == group]
            eligible_results = [
                results[(case.stem, method)]
                for case in group_cases
                if results[(case.stem, method)].performance_eligible
            ]
            gflops_values = [
                item.gflops for item in eligible_results if item.gflops is not None
            ]
            time_values = [
                item.time_ms for item in eligible_results if item.time_ms is not None
            ]
            peak_values = [
                item.peak_pct for item in eligible_results if item.peak_pct is not None
            ]
            retention_values = [
                item.retention_pct
                for item in eligible_results
                if item.retention_pct is not None
            ]

            rows.append(
                {
                    "method": METHOD_DISPLAY[method],
                    "group": group,
                    "runnable_over_total": f"{len(eligible_results)}/{len(group_cases)}",
                    "median_time_ms": _format_optional(_median_or_none(time_values)),
                    "median_gflops": _format_optional(_median_or_none(gflops_values)),
                    "median_peak_pct": _format_optional(_median_or_none(peak_values), digits=2),
                    "median_retention_pct": (
                        "100.00"
                        if method == "cuda" and eligible_results
                        else _format_optional(_median_or_none(retention_values), digits=2)
                    ),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_tables(
    path: Path,
    translation_rows: list[dict[str, str]],
    perf_rows: list[dict[str, str]],
) -> None:
    grouped_translation = {
        (row["group"], row["method"]): row for row in translation_rows
    }
    lines = [
        "# SYCL Adaptation Paper Tables",
        "",
        "## Experiment 1: Translation Functionality",
        "",
        "| Group | DPC++ Compile Success | DPC++ Runnable | DPC++ Correctness Pass | Our-SYCL Compile Success | Our-SYCL Runnable | Our-SYCL Correctness Pass |",
        "| ----- | --------------------- | ------------- | ---------------------- | ------------------------ | ---------------- | ------------------------- |",
    ]
    for group in ("S", "K", "N", "M", "A"):
        dpcpp_row = grouped_translation.get((group, "DPC++"))
        our_row = grouped_translation.get((group, "Our-SYCL"))
        if dpcpp_row is None or our_row is None:
            continue
        total = dpcpp_row["total"]
        lines.append(
            "| "
            + " | ".join(
                [
                    group,
                    f"{dpcpp_row['compile_success']}/{total}" if total != "0" else "N/A",
                    f"{dpcpp_row['runnable']}/{total}" if total != "0" else "N/A",
                    f"{dpcpp_row['correctness_pass']}/{total}" if total != "0" else "N/A",
                    f"{our_row['compile_success']}/{total}" if total != "0" else "N/A",
                    f"{our_row['runnable']}/{total}" if total != "0" else "N/A",
                    f"{our_row['correctness_pass']}/{total}" if total != "0" else "N/A",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Experiment 2: Translation Performance",
            "",
            "| Method | Group | Runnable / Total | Median GFLOPS | Median %Peak | Median Retention vs CUDA |",
            "| ------ | ----- | ---------------- | ------------- | ------------ | ------------------------ |",
        ]
    )
    for row in perf_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["method"],
                    row["group"],
                    row["runnable_over_total"],
                    row["median_gflops"] or "N/A",
                    row["median_peak_pct"] or "N/A",
                    (
                        f"{row['median_retention_pct']}%"
                        if row["median_retention_pct"]
                        else "N/A"
                    ),
                ]
            )
            + " |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _benchmark_case_method(
    *,
    method: str,
    case: CaseDef,
    source_file: Path | None,
    warmup: int,
    iters: int,
    rtol: float,
    atol: float,
    log_path: Path,
    needs_wrap: bool = False,
) -> BenchResult:
    try:
        if method == "cuda":
            assert source_file is not None
            return _benchmark_cuda(
                case=case,
                cuda_file=source_file,
                warmup=warmup,
                iters=iters,
                rtol=rtol,
                atol=atol,
                log_path=log_path,
            )
        if source_file is None:
            return BenchResult(
                method=method,
                source_file="",
                status="missing",
                notes=["Source file is missing."],
                log_file=str(log_path),
            )
        return _benchmark_sycl_host_entry(
            case=case,
            source_file=source_file,
            warmup=warmup,
            iters=iters,
            rtol=rtol,
            atol=atol,
            log_path=log_path,
            needs_wrap=needs_wrap,
        )
    except Exception as exc:
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"[unexpected_exception] {exc}\n")
        return BenchResult(
            method=method,
            source_file=str(source_file) if source_file else "",
            status="unexpected_error",
            notes=[str(exc)],
            log_file=str(log_path),
        )


def main() -> int:
    args = parse_args()
    repo_root = _repo_root()
    manifest_path = args.manifest.resolve()
    cuda_dir = args.cuda_dir.resolve()
    our_sycl_dir = args.our_sycl_dir.resolve()
    dpcpp_dir = args.dpcpp_dir.resolve()
    results_dir = args.results_dir.resolve()
    logs_root = results_dir / "perf_logs"
    detail_csv = results_dir / "gemm_detail.csv"
    translation_csv = results_dir / "translation_summary.csv"
    perf_csv = results_dir / "performance_summary.csv"
    markdown_tables = results_dir / "experiment_tables.md"

    cases = _load_cases(
        manifest_path=manifest_path,
        repo_root=repo_root,
        cuda_dir=cuda_dir,
        cpu_dir=repo_root / "benchmark" / "data" / "cpp_baseline_gemm",
    )
    if not cases:
        raise RuntimeError(f"No GEMM cases were loaded from {manifest_path}")

    logs_root.mkdir(parents=True, exist_ok=True)
    for method in METHOD_DISPLAY:
        (logs_root / method).mkdir(parents=True, exist_ok=True)

    print()
    print("==========================================")
    print("=== SYCL Adaptation GEMM Experiment    ===")
    print("==========================================")
    print(f"Manifest      : {manifest_path}")
    print(f"CUDA dir      : {cuda_dir}")
    print(f"Our-SYCL dir  : {our_sycl_dir}")
    print(f"DPC++ dir     : {dpcpp_dir}")
    print(f"Results dir   : {results_dir}")
    print(f"Warmup / Iters: {args.warmup} / {args.iters}")
    print()

    results: dict[tuple[str, str], BenchResult] = {}

    for case in cases:
        print(f"[{case.case_id}] {case.stem}")

        cuda_log = logs_root / "cuda" / f"{case.stem}.log"
        cuda_result = _benchmark_case_method(
            method="cuda",
            case=case,
            source_file=case.cuda_file,
            warmup=args.warmup,
            iters=args.iters,
            rtol=args.rtol,
            atol=args.atol,
            log_path=cuda_log,
        )
        results[(case.stem, "cuda")] = cuda_result

        our_sycl_log = logs_root / "our_sycl" / f"{case.stem}.log"
        our_sycl_source = our_sycl_dir / f"{case.stem}.cpp"
        our_sycl_result = _benchmark_case_method(
            method="our_sycl",
            case=case,
            source_file=our_sycl_source,
            warmup=args.warmup,
            iters=args.iters,
            rtol=args.rtol,
            atol=args.atol,
            log_path=our_sycl_log,
            needs_wrap=True,
        )
        results[(case.stem, "our_sycl")] = our_sycl_result

        dpcpp_log = logs_root / "dpcpp" / f"{case.stem}.log"
        dpcpp_source = _canonical_dpcpp_source(dpcpp_dir, case.stem)
        dpcpp_result = _benchmark_case_method(
            method="dpcpp",
            case=case,
            source_file=dpcpp_source,
            warmup=args.warmup,
            iters=args.iters,
            rtol=args.rtol,
            atol=args.atol,
            log_path=dpcpp_log,
            needs_wrap=False,
        )
        results[(case.stem, "dpcpp")] = dpcpp_result

        cuda_gflops = results[(case.stem, "cuda")].gflops
        for method in ("our_sycl", "dpcpp"):
            current = results[(case.stem, method)]
            if current.gflops is not None:
                current.peak_pct = current.gflops / args.peak_gflops * 100.0
            if current.gflops is not None and cuda_gflops and cuda_gflops > 0:
                current.retention_pct = current.gflops / cuda_gflops * 100.0
        if cuda_gflops is not None:
            results[(case.stem, "cuda")].peak_pct = cuda_gflops / args.peak_gflops * 100.0
            results[(case.stem, "cuda")].retention_pct = 100.0

    detail_rows: list[dict[str, str]] = []
    for case in cases:
        cuda_result = results[(case.stem, "cuda")]
        our_sycl_result = results[(case.stem, "our_sycl")]
        dpcpp_result = results[(case.stem, "dpcpp")]
        detail_rows.append(
            {
                "case_id": case.case_id,
                "group": case.group,
                "category": case.category,
                "m": str(case.m),
                "k": str(case.k),
                "n": str(case.n),
                "cuda_source": str(case.cuda_file),
                "cuda_status": cuda_result.status,
                "cuda_compile_success": str(cuda_result.compile_success),
                "cuda_runnable": str(cuda_result.runnable),
                "cuda_correctness": str(cuda_result.correctness_pass),
                "cuda_time_ms": _format_optional(cuda_result.time_ms),
                "cuda_gflops": _format_optional(cuda_result.gflops),
                "cuda_peak_pct": _format_optional(cuda_result.peak_pct, digits=2),
                "cuda_log_file": cuda_result.log_file,
                "our_sycl_source": our_sycl_result.source_file,
                "our_sycl_status": our_sycl_result.status,
                "our_sycl_compile_success": str(our_sycl_result.compile_success),
                "our_sycl_runnable": str(our_sycl_result.runnable),
                "our_sycl_correctness": str(our_sycl_result.correctness_pass),
                "our_sycl_time_ms": _format_optional(our_sycl_result.time_ms),
                "our_sycl_gflops": _format_optional(our_sycl_result.gflops),
                "our_sycl_peak_pct": _format_optional(our_sycl_result.peak_pct, digits=2),
                "our_sycl_retention_pct": _format_optional(
                    our_sycl_result.retention_pct, digits=2
                ),
                "our_sycl_log_file": our_sycl_result.log_file,
                "dpcpp_source": dpcpp_result.source_file,
                "dpcpp_status": dpcpp_result.status,
                "dpcpp_compile_success": str(dpcpp_result.compile_success),
                "dpcpp_runnable": str(dpcpp_result.runnable),
                "dpcpp_correctness": str(dpcpp_result.correctness_pass),
                "dpcpp_time_ms": _format_optional(dpcpp_result.time_ms),
                "dpcpp_gflops": _format_optional(dpcpp_result.gflops),
                "dpcpp_peak_pct": _format_optional(dpcpp_result.peak_pct, digits=2),
                "dpcpp_retention_pct": _format_optional(
                    dpcpp_result.retention_pct, digits=2
                ),
                "dpcpp_log_file": dpcpp_result.log_file,
            }
        )

    translation_rows = _aggregate_translation_summary(cases, results)
    perf_rows = _aggregate_perf_summary(cases, results)

    _write_csv(
        detail_csv,
        detail_rows,
        [
            "case_id",
            "group",
            "category",
            "m",
            "k",
            "n",
            "cuda_source",
            "cuda_status",
            "cuda_compile_success",
            "cuda_runnable",
            "cuda_correctness",
            "cuda_time_ms",
            "cuda_gflops",
            "cuda_peak_pct",
            "cuda_log_file",
            "our_sycl_source",
            "our_sycl_status",
            "our_sycl_compile_success",
            "our_sycl_runnable",
            "our_sycl_correctness",
            "our_sycl_time_ms",
            "our_sycl_gflops",
            "our_sycl_peak_pct",
            "our_sycl_retention_pct",
            "our_sycl_log_file",
            "dpcpp_source",
            "dpcpp_status",
            "dpcpp_compile_success",
            "dpcpp_runnable",
            "dpcpp_correctness",
            "dpcpp_time_ms",
            "dpcpp_gflops",
            "dpcpp_peak_pct",
            "dpcpp_retention_pct",
            "dpcpp_log_file",
        ],
    )
    _write_csv(
        translation_csv,
        translation_rows,
        ["method", "group", "compile_success", "runnable", "correctness_pass", "total"],
    )
    _write_csv(
        perf_csv,
        perf_rows,
        [
            "method",
            "group",
            "runnable_over_total",
            "median_time_ms",
            "median_gflops",
            "median_peak_pct",
            "median_retention_pct",
        ],
    )
    _write_markdown_tables(markdown_tables, translation_rows, perf_rows)

    print("Wrote:")
    print(f"  detail CSV        : {detail_csv}")
    print(f"  translation CSV   : {translation_csv}")
    print(f"  performance CSV   : {perf_csv}")
    print(f"  markdown tables   : {markdown_tables}")
    print(f"  per-case perf logs: {logs_root}")
    print("==========================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
