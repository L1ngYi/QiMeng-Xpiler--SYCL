import ctypes
import os
import shlex
import shutil
import subprocess
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F


def avgpool_np(input_tensor, kernel_stride):
    input_tensor = input_tensor.permute(0, 3, 1, 2)
    avgpool = torch.nn.AvgPool2d(
        kernel_size=kernel_stride[:2], stride=kernel_stride[2:]
    )
    # Perform average pooling.
    output_tensor = avgpool(input_tensor)
    output_tensor = output_tensor.permute(0, 2, 3, 1)
    return output_tensor


def sumpool_np(input_tensor, kernel_stride):
    input_tensor = input_tensor.permute(0, 3, 1, 2)
    avgpool = torch.nn.AvgPool2d(
        kernel_size=kernel_stride[:2], stride=kernel_stride[2:]
    )
    # Perform average pooling.
    output_tensor = avgpool(input_tensor)
    output_tensor = output_tensor.permute(0, 2, 3, 1)
    return output_tensor * kernel_stride[0] * kernel_stride[1]


def maxpool_np(input_tensor, kernel_stride):
    input_tensor = input_tensor.permute(0, 3, 1, 2)
    avgpool = torch.nn.AvgPool2d(
        kernel_size=kernel_stride[:2], stride=kernel_stride[2:]
    )
    # Perform average pooling.
    output_tensor = avgpool(input_tensor)
    output_tensor = output_tensor.permute(0, 2, 3, 1)
    return output_tensor


def minpool_np(input_tensor, kernel_stride):
    class MinPool2d(torch.nn.Module):
        def __init__(self, kernel_size, stride=None, padding=0):
            super(MinPool2d, self).__init__()
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding

        def forward(self, x):
            # Inverted input
            x_neg = -x
            # Perform maximum pooling.
            x_maxpool = F.max_pool2d(
                x_neg,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
            # Reversing the result again
            return -x_maxpool

    # Using a custom MinPool2d
    pool = MinPool2d(kernel_size=kernel_stride[:2], stride=kernel_stride[2:])
    input_tensor = input_tensor.permute(0, 3, 1, 2)
    output_tensor = pool(input_tensor)
    output_tensor = output_tensor.permute(0, 2, 3, 1)
    return output_tensor


def conv2d_nchw(
    input_tensor, in_channels, out_channels, kernel, stride, padding=0
):
    # Define the convolutional layer.
    conv_layer = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel,
        stride=stride,
        padding=padding,
    )
    output = conv_layer(input_tensor)
    return output


def conv2d_nhwc(
    input_nhwc, in_channels, out_channels, kernel, stride, padding
):
    weight_hwio = torch.randn(
        [out_channels, kernel, kernel, input_nhwc.shape[3]], device="cpu"
    )

    # Convert the input from NHWC to NCHW.
    input_nchw = input_nhwc.permute(0, 3, 1, 2)

    # Convert the kernel from HWIO (H, W, in_channels, out_channels) format to
    # PyTorch's OIHW format.
    weight_oihw = weight_hwio.permute(0, 3, 1, 2)

    # Perform convolution operations using the transformed convolution kernel
    # and input.
    output_nchw = F.conv2d(
        input_nchw, weight_oihw, stride=stride, padding=padding
    )

    # Convert the output from NCHW back to NHWC.
    output_nhwc = output_nchw.permute(0, 3, 1, 2)
    return output_nhwc


def run_dlboost_compilation(so_name, file_name):
    try:
        output = subprocess.run(
            [
                "g++",
                "-shared",
                "-fPIC",
                "-march=icelake-server",
                "-O3",
                file_name,
                "-o",
                so_name,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            check=True,
            timeout=15,
        )
        return True, output
    except subprocess.CalledProcessError as e:
        return False, e.output


def run_cpp_compilation(so_name, file_name):
    try:
        output = subprocess.run(
            ["g++", "-shared", "-fPIC", "-O3", file_name, "-o", so_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            check=True,
            text=True,
            timeout=15,
        )
        return True, output
    except subprocess.CalledProcessError as e:
        return False, e.output



def run_cuda_compilation(so_name, file_name):
    try:
        output = subprocess.run(
            [
                "nvcc",
                "-Xcompiler",
                "-fPIC",
                "-shared",
                "-arch=sm_80",
                "-o",
                so_name,
                file_name,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            check=True,
            text=True,
            timeout=15,
        )
        return True, output
    except subprocess.CalledProcessError as e:
        return False, e.output
    

def run_sycl_compilation(so_name, file_name):
    """
    使用用户配置的 DPCPP/Clang SYCL 编译器将 SYCL 代码编译为共享库。
    """
    try:
        env = configure_sycl_environment()
        compiler_cmd = shlex.split(env.get("SYCL_COMPILER", ""))
        if not compiler_cmd:
            detected_compiler = shutil.which("icpx", path=env.get("PATH"))
            if not detected_compiler:
                detected_compiler = shutil.which("clang++", path=env.get("PATH"))
            if not detected_compiler:
                raise FileNotFoundError(
                    "No SYCL compiler found. Set SYCL_COMPILER or load env_sycl.sh."
                )
            compiler_cmd = [detected_compiler]

        extra_flags = shlex.split(env.get("SYCL_EXTRA_FLAGS", ""))
        cmd = compiler_cmd + [
            "-fsycl",
            "-fPIC",
            "-shared",
            "-O3",
            "-std=c++17",
        ]
        cmd.extend(extra_flags)
        cmd.extend([file_name, "-o", so_name])

        output = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            check=True,
            text=True,
            timeout=120,
            env=env,
        )
        return True, output
    except subprocess.CalledProcessError as e:
        return False, e.output
    except subprocess.TimeoutExpired:
        return False, "Compilation timed out"
    except Exception as e:
        return False, str(e)
 

def run_hip_compilation(so_name, file_name):
    try:
        output = subprocess.run(
            [
                "hipcc",
                "-fPIC",
                "-shared",
                "-arch=sm_80",
                "-o",
                so_name,
                file_name,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            check=True,
            text=True,
            timeout=15,
        )
        return True, output
    except subprocess.CalledProcessError as e:
        return False, e.output

 
def _resolve_sycl_env_script():
    explicit_script = os.environ.get("SYCL_ENV_SCRIPT")
    if explicit_script:
        script_path = os.path.abspath(os.path.expanduser(explicit_script))
        if not os.path.isfile(script_path):
            raise FileNotFoundError(
                f"SYCL environment script not found: {script_path}"
            )
        return script_path

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidate_paths = [
        os.path.join(os.getcwd(), "env_sycl.sh"),
        os.path.join(repo_root, "env_sycl.sh"),
        os.path.expanduser("~/env_sycl.sh"),
    ]

    for candidate in candidate_paths:
        script_path = os.path.abspath(os.path.expanduser(candidate))
        if os.path.isfile(script_path):
            return script_path
    return None


@lru_cache(maxsize=None)
def _load_sycl_env_from_script(script_path):
    command = f"source {shlex.quote(script_path)} >/dev/null 2>&1 && env -0"
    result = subprocess.run(
        ["bash", "-lc", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    loaded_env = {}
    for item in result.stdout.decode("utf-8", errors="ignore").split("\0"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        loaded_env[key] = value
    return loaded_env


def get_sycl_environment():
    env = os.environ.copy()
    script_path = _resolve_sycl_env_script()
    if script_path:
        env.update(_load_sycl_env_from_script(script_path))

    device_selector = os.environ.get("SYCL_DEVICE_SELECTOR")
    if device_selector:
        env["ONEAPI_DEVICE_SELECTOR"] = device_selector
    return env


def configure_sycl_environment():
    env = get_sycl_environment()
    os.environ.update(env)
    return env


@lru_cache(maxsize=1)
def preload_sycl_runtime():
    env = configure_sycl_environment()
    rtld_flag = getattr(os, "RTLD_GLOBAL", getattr(ctypes, "RTLD_GLOBAL", 0))
    library_dirs = env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    library_names = ("libsycl.so", "libsycl.so.8", "libsycl-preview.so")

    for directory in library_dirs:
        if not directory:
            continue
        for library_name in library_names:
            library_path = os.path.join(directory, library_name)
            if not os.path.isfile(library_path):
                continue
            ctypes.CDLL(library_path, mode=rtld_flag)
            return library_path
    return None


def normalize_sycl_dtype(dtype):
    normalized = dtype.replace("sycl::", "")
    normalized = normalized.replace("const", "")
    normalized = " ".join(normalized.split())
    return normalized.strip()


def get_sycl_numpy_dtype(dtype):
    normalized = normalize_sycl_dtype(dtype)
    mapping = {
        "half": np.float16,
        "float": np.float32,
        "double": np.float64,
        "int8_t": np.int8,
        "uint8_t": np.uint8,
        "int": np.int32,
        "int32_t": np.int32,
        "size_t": np.int64,
    }
    if normalized not in mapping:
        raise NotImplementedError(f"Unsupported SYCL dtype: {dtype}")
    return mapping[normalized]


def get_sycl_ctype(dtype):
    normalized = normalize_sycl_dtype(dtype)
    mapping = {
        "half": ctypes.c_uint16,
        "float": ctypes.c_float,
        "double": ctypes.c_double,
        "int8_t": ctypes.c_int8,
        "uint8_t": ctypes.c_uint8,
        "int": ctypes.c_int,
        "int32_t": ctypes.c_int32,
        "size_t": ctypes.c_size_t,
    }
    if normalized not in mapping:
        raise NotImplementedError(f"Unsupported SYCL dtype: {dtype}")
    return mapping[normalized]


def run_test(file_name, test_file):
    try:
        env = None
        test_dir_parts = os.path.normpath(test_file).split(os.sep)
        if "sycl_test" in test_dir_parts:
            env = configure_sycl_environment()
        output = subprocess.run(
            ["python", test_file, "--file", file_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            check=True,
            text=True,
            timeout=400,
            env=env,
        )
        return True, output
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except subprocess.CalledProcessError as e:
        return False, e.output
