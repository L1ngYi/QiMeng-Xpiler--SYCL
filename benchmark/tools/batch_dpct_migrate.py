#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_results_dir() -> Path:
    return _repo_root() / "benchmark" / "results" / "sycl_paper"


def _detect_cuda_include_path() -> str | None:
    candidates: list[Path] = []

    for env_name in ("CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"):
        raw_value = os.environ.get(env_name)
        if raw_value:
            candidates.append(Path(raw_value) / "include")

    candidates.extend(
        [
            Path("/usr/local/cuda/include"),
            Path("/usr/local/cuda-12/include"),
            Path("/usr/local/cuda-11/include"),
            Path("/opt/cuda/include"),
            Path("/usr/include"),
        ]
    )

    for candidate in candidates:
        if (candidate / "cuda_runtime.h").is_file():
            return str(candidate)
    return None


def _find_migrated_source(search_root: Path, stem: str) -> Path | None:
    exact_names = [
        f"{stem}.cpp",
        f"{stem}.cc",
        f"{stem}.cxx",
        f"{stem}.dp.cpp",
        f"{stem}.dp.cc",
        f"{stem}.dp.cxx",
    ]
    for file_name in exact_names:
        matches = sorted(search_root.rglob(file_name))
        if matches:
            return matches[0]

    fallback_matches = sorted(
        path
        for path in search_root.rglob(f"{stem}*")
        if path.is_file() and path.suffix in {".cpp", ".cc", ".cxx"}
    )
    return fallback_matches[0] if fallback_matches else None


def _iter_source_files(
    source_dir: Path, glob_pattern: str, recursive: bool, substring: str | None
) -> list[Path]:
    iterator = source_dir.rglob(glob_pattern) if recursive else source_dir.glob(glob_pattern)
    files = sorted(path for path in iterator if path.is_file())
    if substring:
        files = [path for path in files if substring in path.name]
    return files


def parse_args() -> argparse.Namespace:
    repo_root = _repo_root()
    results_dir = _default_results_dir()
    parser = argparse.ArgumentParser(
        description="Batch-migrate CUDA sources to DPC++/SYCL with per-case logs.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=repo_root / "benchmark" / "data" / "cuda_baseline_gemm",
        help="Directory containing CUDA sources to migrate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "benchmark" / "data" / "dpcpp_baseline_gemm",
        help="Canonical directory used to store successfully migrated sources.",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=repo_root / "tmp" / "dpct_workspace",
        help="Workspace where dpct writes its original migration tree.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=results_dir / "dpct_logs",
        help="Directory for per-case migration logs.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=results_dir / "dpct_conversion_summary.csv",
        help="CSV written with one summary row per CUDA source.",
    )
    parser.add_argument(
        "--failed-list",
        type=Path,
        default=results_dir / "dpct_failed_cases.txt",
        help="Text file listing non-successful cases for quick inspection.",
    )
    parser.add_argument(
        "--dpct-bin",
        default=shutil.which("dpct") or "dpct",
        help="dpct executable to invoke.",
    )
    parser.add_argument(
        "--glob",
        default="*.cu",
        help="Glob used to collect input CUDA files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search source-dir recursively.",
    )
    parser.add_argument(
        "--match",
        default=None,
        help="Optional substring filter applied to file names.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of collected files.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=300,
        help="Per-file dpct timeout in seconds.",
    )
    parser.add_argument(
        "--cuda-include-path",
        default=_detect_cuda_include_path(),
        help="CUDA include directory passed to dpct when available.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra dpct CLI argument. Repeatable.",
    )
    parser.add_argument(
        "--clean-work-root",
        action="store_true",
        help="Delete work-root before starting the batch.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    work_root = args.work_root.resolve()
    log_dir = args.log_dir.resolve()
    summary_csv = args.summary_csv.resolve()
    failed_list = args.failed_list.resolve()

    if not source_dir.is_dir():
        print(f"[ERROR] CUDA source directory not found: {source_dir}", file=sys.stderr)
        return 2

    if shutil.which(args.dpct_bin) is None and not Path(args.dpct_bin).is_file():
        print(
            f"[ERROR] dpct executable not found: {args.dpct_bin}",
            file=sys.stderr,
        )
        return 2

    files = _iter_source_files(source_dir, args.glob, args.recursive, args.match)
    if args.limit is not None:
        files = files[: args.limit]

    if not files:
        print(f"[ERROR] No CUDA sources matched under {source_dir}", file=sys.stderr)
        return 2

    if args.clean_work_root and work_root.exists():
        shutil.rmtree(work_root)

    output_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    failed_list.parent.mkdir(parents=True, exist_ok=True)

    print()
    print("==========================================")
    print("=== Batch DPC++ Migration (dpct)       ===")
    print("==========================================")
    print(f"Source dir        : {source_dir}")
    print(f"Canonical out dir : {output_dir}")
    print(f"Workspace root    : {work_root}")
    print(f"Log dir           : {log_dir}")
    print(f"Summary CSV       : {summary_csv}")
    print(f"dpct executable   : {args.dpct_bin}")
    if args.cuda_include_path:
        print(f"CUDA include path : {args.cuda_include_path}")
    else:
        print("CUDA include path : <not set>")
    print(f"Matched files     : {len(files)}")
    print()

    rows: list[dict[str, str]] = []
    success_count = 0
    failure_count = 0

    for index, src_file in enumerate(files, start=1):
        stem = src_file.stem
        case_work_root = work_root / stem
        case_log_path = log_dir / f"{stem}.log"
        canonical_out = output_dir / f"{stem}.cpp"

        if case_work_root.exists():
            shutil.rmtree(case_work_root)
        case_work_root.mkdir(parents=True, exist_ok=True)
        if canonical_out.exists():
            canonical_out.unlink()

        cmd = [
            args.dpct_bin,
            f"--out-root={case_work_root}",
        ]
        if args.cuda_include_path:
            cmd.append(f"--cuda-include-path={args.cuda_include_path}")
        for extra_arg in args.extra_arg:
            cmd.append(extra_arg)
        cmd.append(str(src_file))

        status = "failed"
        exit_code = ""
        duration_sec = ""
        generated_source = ""

        print(f"[{index:>3}/{len(files):>3}] Migrating {src_file.name:<28} ... ", end="", flush=True)
        start_time = time.time()
        with case_log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(f"[command] {' '.join(shlex.quote(arg) for arg in cmd)}\n")
            log_file.write(f"[source] {src_file}\n")
            log_file.write(f"[workspace] {case_work_root}\n\n")
            try:
                completed = subprocess.run(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                    text=True,
                    timeout=args.timeout_sec,
                )
                exit_code = str(completed.returncode)
                duration_sec = f"{time.time() - start_time:.3f}"
                maybe_generated = _find_migrated_source(case_work_root, stem)
                generated_source = str(maybe_generated) if maybe_generated else ""

                if completed.returncode == 0 and maybe_generated:
                    shutil.copy2(maybe_generated, canonical_out)
                    status = "ok"
                elif completed.returncode == 0 and not maybe_generated:
                    status = "failed_no_output"
                elif completed.returncode != 0 and maybe_generated:
                    status = "partial_output"
                else:
                    status = "failed"
            except subprocess.TimeoutExpired:
                duration_sec = f"{time.time() - start_time:.3f}"
                status = "timeout"
                exit_code = "timeout"
                maybe_generated = _find_migrated_source(case_work_root, stem)
                generated_source = str(maybe_generated) if maybe_generated else ""
                log_file.write("\n[timeout] dpct exceeded the configured timeout.\n")

        if status != "ok" and canonical_out.exists():
            canonical_out.unlink()

        if status == "ok":
            success_count += 1
            print("OK")
        else:
            failure_count += 1
            print(f"{status.upper()}  log={case_log_path}")

        rows.append(
            {
                "case_name": stem,
                "source_file": str(src_file),
                "status": status,
                "exit_code": exit_code,
                "duration_sec": duration_sec,
                "generated_source": generated_source,
                "canonical_output": str(canonical_out) if canonical_out.exists() else "",
                "log_file": str(case_log_path),
            }
        )

    with summary_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "case_name",
                "source_file",
                "status",
                "exit_code",
                "duration_sec",
                "generated_source",
                "canonical_output",
                "log_file",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with failed_list.open("w", encoding="utf-8") as fh:
        for row in rows:
            if row["status"] != "ok":
                fh.write(f"{row['case_name']}\t{row['status']}\t{row['log_file']}\n")

    print()
    print("==========================================")
    print(
        f"Completed: {success_count} succeeded, {failure_count} failed, total {len(files)}"
    )
    print(f"Canonical DPC++ outputs : {output_dir}")
    print(f"Per-case migration logs : {log_dir}")
    print(f"Summary CSV             : {summary_csv}")
    print(f"Failure list            : {failed_list}")
    print("==========================================")
    return 0 if success_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
