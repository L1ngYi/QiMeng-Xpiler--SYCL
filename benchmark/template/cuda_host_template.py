import re
from string import Template


def infer_grid_dim_from_kernel(kernel_code: str, thread_num: int) -> str:
    """
    根据 Kernel 代码中对内置变量的使用情况（threadIdx/blockIdx），
    启发式地推断并生成 dim3 网格（Grid）和块（Block）的定义代码。
    """
    # Derive the dimensions used by blockIdx and threadIdx.
    # 通过正则检查代码中是否使用了 x, y, z 维度的 blockIdx
    use_block_x = bool(re.search(r"\bblockIdx\.x\b", kernel_code))
    use_block_y = bool(re.search(r"\bblockIdx\.y\b", kernel_code))
    use_block_z = bool(re.search(r"\bblockIdx\.z\b", kernel_code))

    # 检查是否使用了 x, y, z 维度的 threadIdx
    use_thread_x = bool(re.search(r"\bthreadIdx\.x\b", kernel_code))
    use_thread_y = bool(re.search(r"\bthreadIdx\.y\b", kernel_code))
    use_thread_z = bool(re.search(r"\bthreadIdx\.z\b", kernel_code))

    # Set numBlocks
    # 设置网格大小：如果用到了某维度，默认设为 256（这是一个简单的启发式默认值，实际应根据数据大小动态计算）
    numBlocks_x = 256 if use_block_x else 1
    numBlocks_y = 256 if use_block_y else 1
    numBlocks_z = 256 if use_block_z else 1
    numblocks_define = (
        f"dim3 numBlocks({numBlocks_x}, {numBlocks_y}, {numBlocks_z});"
    )

    # Set the blockSize.
    # 设置线程块大小：如果用到了 x 维度，使用传入的 thread_num（通常从 launch_bounds 获取）
    blockSize_x = thread_num if use_thread_x else 1
    blockSize_y = 1024 if use_thread_y else 1
    blockSize_z = 1024 if use_thread_z else 1
    blocksize_define = (
        f"dim3 blockSize({blockSize_x}, {blockSize_y}, {blockSize_z});"
    )
    return numblocks_define, blocksize_define


def create_cuda_func(file_name, op_type="ewise"):
    """
    读取 CUDA 源代码，生成包含 Host 端调用的完整 C++ 文件。
    参数:
        file_name: 原始 .cu 文件路径
        op_type: 算子类型 (ewise, pool, matmul, etc.)，决定了内存分配和拷贝的逻辑
    """
    with open(file_name, "r") as f:
        original_function = f.read()

    # Regular expression to extract function signature
    # 使用正则提取 Kernel 函数签名
    # 捕获组 1: launch_bounds 中的线程数 (可选)
    # 捕获组 2: 函数名
    # 捕获组 3: 参数列表字符串
    function_signature_pattern = r"__global__ void\s*(?:__launch_bounds__\((\d+)(?:,\s*\d+)?\))?\s*(\w+)\(([^)]*)\)"
    match = re.search(function_signature_pattern, original_function, re.DOTALL)
    if not match:
        raise ValueError("Could not find CUDA kernel signature.")

    thread_num = match.group(1)
    kernel_name = match.group(2)
    param_list_str = match.group(3)

    if thread_num is None:
        thread_num = "1024"  # Default value

    # 清洗参数列表，移除修饰符如 __restrict__, const 等，以便提取纯变量名和类型
    params = [param.strip() for param in param_list_str.split(",")]
    params = [var.replace("__restrict__ ", "").strip() for var in params]
    params = [var.replace("const ", "").strip() for var in params]
    param_list = ", ".join(params)
    # 提取参数名列表 (用于生成调用代码)
    param_names = [
        param.split()[-1].replace("*", "").replace("__restrict__", "").strip()
        for param in params
    ]

    # 为每个参数生成对应的设备端变量名 (加上 _cuda 后缀)
    device_vars = [f"{name}_cuda" for name in param_names]

    # Memory allocation and copy operations
    # 初始化代码片段列表
    device_memory_alloc = [] # cudaMalloc 代码
    memcpy = []              # cudaMemcpy (Host->Device) 代码
    size = None
    
    # ---------------------------------------------------------
    # 根据算子类型生成特定的内存管理代码
    # ---------------------------------------------------------
    
    # 情况 1: Element-wise (逐元素操作)
    # 假设所有输入输出大小一致，由单一变量 size 控制
    if op_type == "ewise":
        size = "size"
        for param in params:
            name = param.split("*")[1]
            device_memory_alloc.append(param + "_cuda;\n")
            # 为每个参数分配 size * sizeof(float) 的显存
            device_memory_alloc.append(
                f"cudaMalloc((void**)&{name}_cuda, {size} * sizeof(float));\n"
            )

        # 除了最后一个参数（通常是输出），其余参数从 Host 拷贝到 Device
        for param in params[:-1]:
            name = param.split("*")[1]
            memcpy.append(
                f"cudaMemcpy({name}_cuda, {name}, {size} * sizeof(float), cudaMemcpyHostToDevice);\n"
            )
        # Copy back
        # 将最后一个参数（输出）从 Device 拷贝回 Host
        name = params[-1].split("*")[1]
        memcpy_back = f"cudaMemcpy({name}, {name}_cuda, {size} * sizeof(float), cudaMemcpyDeviceToHost);\n"
    
    # 情况 2: Pooling (池化操作)
    # 需要两个 size 变量：size1 (输入大小) 和 size2 (输出大小)
    elif op_type == "pool":
        size = ["size1", "size2"]
        for i, param in enumerate(params):
            name = param.split("*")[1]
            device_memory_alloc.append(param + "_cuda;\n")
            device_memory_alloc.append(
                f"cudaMalloc((void**)&{name}_cuda, {size[i]} * sizeof(float));\n"
            )

        for i, param in enumerate(params[:-1]):
            name = param.split("*")[1]
            memcpy.append(
                f"cudaMemcpy({name}_cuda, {name}, {size[i]} * sizeof(float), cudaMemcpyHostToDevice);\n"
            )
        # Copy back
        name = params[-1].split("*")[1]
        memcpy_back = f"cudaMemcpy({name}, {name}_cuda, {size[-1]} * sizeof(float), cudaMemcpyDeviceToHost);\n"
    
    # 情况 3: Matmul (矩阵乘法)
    # 需要三个 size 变量：M*K, K*N, M*N，同时支持不同数据类型（从参数定义中解析）
    elif op_type == "matmul":
        size = ["size1", "size2", "size3"]
        for i, param in enumerate(params):
            name = param.split("*")[1]
            dtype = param.split("*")[0] # 解析数据类型 (如 int8_t, float)
            device_memory_alloc.append(param + "_cuda;\n")
            device_memory_alloc.append(
                f"cudaMalloc((void**)&{name}_cuda, {size[i]} * sizeof({dtype}));\n"
            )

        for i, param in enumerate(params[:-1]):
            name = param.split("*")[1]
            dtype = param.split("*")[0]
            memcpy.append(
                f"cudaMemcpy({name}_cuda, {name}, {size[i]} * sizeof({dtype}), cudaMemcpyHostToDevice);\n"
            )
        # Copy back
        name = params[-1].split("*")[1]
        dtype = params[-1].split("*")[0]
        memcpy_back = f"cudaMemcpy({name}, {name}_cuda, size3 * sizeof({dtype}), cudaMemcpyDeviceToHost);\n"

    # 情况 4: Layer Norm (层归一化)
    # 特殊的 size 映射逻辑：输入/输出用 size1，均值/方差用 size2
    elif op_type == "layer_norm":
        size = ["size1", "size2"]
        for i, param in enumerate(params):
            name = param.split("*")[1]
            device_memory_alloc.append(param + "_cuda;\n")
            if i == 1 or i == 2: # 假设第2、3个参数是均值/方差等小数组
                device_memory_alloc.append(
                    f"cudaMalloc((void**)&{name}_cuda, size2 * sizeof(float));\n"
                )
            else:
                device_memory_alloc.append(
                    f"cudaMalloc((void**)&{name}_cuda, size1 * sizeof(float));\n"
                )
        for i, param in enumerate(params[:-1]):
            name = param.split("*")[1]
            if i == 1 or i == 2:
                memcpy.append(
                    f"cudaMemcpy({name}_cuda, {name}, size2 * sizeof(float), cudaMemcpyHostToDevice);\n"
                )
            else:
                memcpy.append(
                    f"cudaMemcpy({name}_cuda, {name}, size1 * sizeof(float), cudaMemcpyHostToDevice);\n"
                )
        # Copy back
        name = params[-1].split("*")[1]
        memcpy_back = f"cudaMemcpy({name}, {name}_cuda, size1 * sizeof(float), cudaMemcpyDeviceToHost);\n"
    
    # 情况 5: Deformable Attention (可变形注意力机制)
    # 复杂的参数列表，涉及 5 个不同的 size
    elif op_type == "deformable":
        size = ["size1", "size2", "size3", "size4", "size5"]
        print("[INFO]params: ", params)
        for i, param in enumerate(params):
            name = param.split("*")[1]
            dtype = param.split("*")[0]
            device_memory_alloc.append(param + "_cuda;\n")
            device_memory_alloc.append(
                f"cudaMalloc((void**)&{name}_cuda, {size[i]} * sizeof({dtype}));\n"
            )

        for i, param in enumerate(params[:-1]):
            name = param.split("*")[1]
            dtype = param.split("*")[0]
            memcpy.append(
                f"cudaMemcpy({name}_cuda, {name}, {size[i]} * sizeof({dtype}), cudaMemcpyHostToDevice);\n"
            )
        # Copy back
        name = params[-1].split("*")[1]
        dtype = params[-1].split("*")[0]
        memcpy_back = f"cudaMemcpy({name}, {name}_cuda, size3 * sizeof({dtype}), cudaMemcpyDeviceToHost);\n"

    # 移除原始代码中的 extern "C"，避免在生成的包装器中重复定义或语法冲突
    original_function = original_function.replace('extern "C"', "")
    # Infer grid dimensions from kernel code
    # 推断 Grid 和 Block 维度
    numblocks_define, blocksize_define = infer_grid_dim_from_kernel(
        original_function,
        thread_num,
    )
    # 生成 Host 函数签名所需的 size 参数列表字符串 (int size1, int size2...)
    if isinstance(size, list):
        size_list = ", ".join(
            arg for arg in ["int " + string for string in size]
        )
    else:
        size_list = "int size"
    # Create host function template
    # 定义完整 CUDA 代码的模板，包含必要的头文件、原始 Kernel 和 extern "C" 的包装函数
    host_func_template = Template(
        """
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <mma.h>
#include <cuda_fp16.h>
#include <stdlib.h>

using namespace nvcuda;

// Original Kernel
${original_function}

extern "C" void ${kernel_name}_kernel(${param_list}, ${size_list}) {
    ${memcpy_alloc_list}
    ${memcpy_htod}

    ${blockSize_define}
    ${numblocks_define}
    ${kernel_name}<<<numBlocks, blockSize>>>(${called_param_list});

    ${memcpy_dtoh}
    ${cuda_free}
}
"""
    )
    # 将分配列表拼接成字符串
    memcpy_alloc_list = "    ".join(alloc for alloc in device_memory_alloc)
    # 执行模板替换，生成最终代码
    new_code = host_func_template.substitute(
        kernel_name=kernel_name,
        original_function=original_function.strip(),
        param_list=param_list,
        memcpy_htod="\n    ".join(memcpy),
        thread_num=thread_num,
        numblocks_define=numblocks_define,
        blockSize_define=blocksize_define,
        called_param_list=", ".join(device_vars),
        memcpy_dtoh=memcpy_back,
        cuda_free="\n    ".join(
            [f"cudaFree({dev}_cuda);" for dev in param_names]
        ),
        size_list=size_list,
        memcpy_alloc_list=memcpy_alloc_list,
    )

    # 将生成的代码写入 _bak.cu 文件
    output_file = file_name.replace(".cu", "_bak.cu")
    with open(output_file, "w") as f:
        f.write(new_code)

    return output_file


if __name__ == "__main__":
    create_cuda_perf_func("benchmark/data/cuda_code_test/sign_5_128.cu")