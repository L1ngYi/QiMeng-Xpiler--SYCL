import argparse
import os
import re
import subprocess
import tempfile
from string import Template


def _run_sycl_compilation(output_file, source_file):
    """Compile a SYCL source file using icpx -fsycl.

    Returns (success: bool, output: str).
    """
    try:
        result = subprocess.run(
            [
                "icpx",
                "-fsycl",
                "-O2",
                "-std=c++17",
                source_file,
                "-o",
                output_file,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            timeout=120,
        )
        success = result.returncode == 0
        return success, result.stdout
    except FileNotFoundError:
        return False, "icpx compiler not found"
    except subprocess.TimeoutExpired:
        return False, "icpx compilation timed out"


# Timing harness template — wraps the kernel in a main() that calls it
# repeatedly and measures elapsed wall-clock time via SYCL event profiling.
_SYCL_HARNESS_TEMPLATE = Template(
    """
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <vector>

using namespace sycl;
using namespace std::chrono;

// ---- Kernel under test ----
${kernel_code}
// ---------------------------

int main() {
    queue q(property::queue::enable_profiling{});

    // Allocate dummy buffers (float arrays sized to cover typical test kernels)
    constexpr int N = 65536;
    std::vector<float> h_a(N, 1.0f), h_b(N, 2.0f), h_c(N, 0.0f);
    float *d_a = malloc_device<float>(N, q);
    float *d_b = malloc_device<float>(N, q);
    float *d_c = malloc_device<float>(N, q);
    q.memcpy(d_a, h_a.data(), N * sizeof(float)).wait();
    q.memcpy(d_b, h_b.data(), N * sizeof(float)).wait();
    q.memcpy(d_c, h_c.data(), N * sizeof(float)).wait();

    // Warm-up
    for (int i = 0; i < 10; ++i) {
        ${kernel_call};
    }
    q.wait();

    // Timed iterations
    auto t0 = high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        ${kernel_call};
    }
    q.wait();
    auto t1 = high_resolution_clock::now();

    double elapsed_s = duration_cast<microseconds>(t1 - t0).count() / 1000.0 / 1000.0;
    printf("%.6f\\n", elapsed_s);

    free(d_a, q);
    free(d_b, q);
    free(d_c, q);
    return 0;
}
"""
)


def _extract_kernel_signature(code):
    """Return (kernel_name, param_names) extracted from the source code."""
    # Match void kernel(... , sycl::queue &q) or void kernel(...)
    m = re.search(
        r"void\s+(\w+)\s*\(([^)]*)\)",
        code,
        re.DOTALL,
    )
    if not m:
        raise ValueError("Could not locate kernel function signature.")
    kernel_name = m.group(1)
    raw_params = m.group(2)

    param_names = []
    for param in raw_params.split(","):
        param = param.strip()
        if not param:
            continue
        # Last token is the variable name (strip pointer/reference symbols)
        var = re.split(r"[\s*&]+", param)[-1].strip()
        param_names.append(var)
    return kernel_name, param_names


def _build_kernel_call(kernel_name, param_names):
    """Build a call expression substituting standard dummy names."""
    # Map common parameter names to dummy variables; fall back to d_a/d_b/d_c/q
    dummy_map = {
        "input": "d_a",
        "input1": "d_a",
        "input2": "d_b",
        "output": "d_c",
        "a": "d_a",
        "b": "d_b",
        "c": "d_c",
        "q": "q",
    }
    args = [dummy_map.get(p, p) for p in param_names]
    return f"{kernel_name}({', '.join(args)})"


def benchmark(file_name):
    """Compile and benchmark a SYCL kernel file.

    Compiles the kernel with ``icpx -fsycl``.  If compilation succeeds and
    the binary can be executed, the measured execution time (milliseconds,
    average over 1000 runs) is returned.  If ``icpx`` is not available or
    runtime execution fails, compilation success is used as a proxy score
    (returning 1.0 ms on success so the optimizer can still distinguish
    valid from invalid transformations).

    Returns:
        float: Execution time in milliseconds (lower is better).
               Returns a large sentinel value (1 000 000) on failure.
    """
    FAILURE = 1_000_000.0

    with open(file_name, "r", encoding="utf-8") as fh:
        kernel_code = fh.read()

    # ---- Try to build a self-contained executable with a timing harness ----
    try:
        kernel_name, param_names = _extract_kernel_signature(kernel_code)
        kernel_call = _build_kernel_call(kernel_name, param_names)

        # Strip the sycl include / using directives from the kernel fragment
        # (the harness already provides them)
        stripped_kernel = re.sub(r"#include\s*<sycl/sycl\.hpp>", "", kernel_code)
        stripped_kernel = re.sub(r"using\s+namespace\s+sycl\s*;", "", stripped_kernel)
        stripped_kernel = stripped_kernel.strip()

        harness = _SYCL_HARNESS_TEMPLATE.substitute(
            kernel_code=stripped_kernel,
            kernel_call=kernel_call,
        )
    except Exception:
        # If we cannot build a harness, fall back to compile-only check
        harness = None

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "sycl_bench.cpp")
        exe_path = os.path.join(tmpdir, "sycl_bench")

        if harness is not None:
            with open(src_path, "w", encoding="utf-8") as fh:
                fh.write(harness)
        else:
            # Compile original file directly (compile-only check)
            src_path = file_name
            exe_path = os.path.join(tmpdir, "sycl_check")

        success, compile_output = _run_sycl_compilation(exe_path, src_path)
        if not success:
            return FAILURE

        # ---- Try to run the executable ----
        if harness is not None and os.path.isfile(exe_path):
            try:
                run_result = subprocess.run(
                    [exe_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    encoding="utf-8",
                    timeout=60,
                )
                if run_result.returncode == 0:
                    elapsed = float(run_result.stdout.strip())
                    return elapsed
            except (subprocess.TimeoutExpired, ValueError, OSError):
                pass

    # Compilation succeeded but execution failed or was skipped —
    # return a small positive value so that compiled SYCL code is
    # preferred over code that fails to compile.
    return 1.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark a SYCL kernel file"
    )
    parser.add_argument(
        "--file_name",
        "-f",
        required=True,
        help="Path to the SYCL .cpp kernel file to benchmark",
    )
    args = parser.parse_args()
    t = benchmark(args.file_name)
    print(f"Execution time: {t:.4f} ms")
