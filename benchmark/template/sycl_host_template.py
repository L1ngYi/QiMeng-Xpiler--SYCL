import re
from string import Template

def create_sycl_func(file_name, op_type="ewise"):
    """
    读取生成的 SYCL Kernel 代码，并用 SYCL Runtime (Queue, USM) 包裹它，
    生成一个完整的、可被 Python ctypes 调用的 C++ 文件。
    """
    with open(file_name, "r") as f:
        original_function = f.read()

    # 1. 解析函数签名 (这一步是为了获取参数列表，以便生成 Host 调用接口)
    # 假设生成的 Kernel 签名是 void kernel_name(arg1, arg2, ..., item)
    # 注意：我们的 Loop Recovery 生成的代码可能没有 item 参数，或者有。
    # 这里我们假设生成的代码是一个普通的 C++ 函数，接受指针参数。
    
    # 正则匹配：匹配返回 void 的函数，获取函数名和参数列表
    # 忽略 static, inline 等前修饰符
    # 例子: void gemm(int8_t* A, int8_t* B, ...)
    function_signature_pattern = r"void\s+(\w+)\s*\(([^)]*)\)"
    match = re.search(function_signature_pattern, original_function)
    
    if not match:
        # 如果找不到，可能是因为代码被 extern "C" 包裹了，或者有其他修饰符
        # 尝试更宽松的匹配
        function_signature_pattern = r"\w+\s+(\w+)\s*\(([^)]*)\)"
        match = re.search(function_signature_pattern, original_function)
        if not match:
             raise ValueError(f"Could not find kernel signature in {file_name}")

    kernel_name = match.group(1)
    param_list_str = match.group(2)

    # 清洗参数列表
    params = [param.strip() for param in param_list_str.split(",")]
    # 过滤掉可能的 item 参数 (如果 MCTS 步骤中已经注入了 item)
    params = [p for p in params if "sycl::id" not in p and "sycl::nd_item" not in p and "sycl::item" not in p]
    
    # 提取参数类型和名称
    # e.g., ["int8_t* A", "float* C"]
    # param_names -> ["A", "C"]
    # param_types -> ["int8_t*", "float*"]
    param_names = []
    param_types = []
    for p in params:
        parts = p.rsplit(" ", 1) # 从右边分，防止类型里有空格 (unsigned int)
        if len(parts) == 2:
            param_types.append(parts[0].strip())
            param_names.append(parts[1].replace("*", "").strip())
    
    # 2. 生成内存管理代码 (USM Malloc & Memcpy)
    device_allocs = []
    memcpys_h2d = [] # Host to Device
    memcpys_d2h = [] # Device to Host
    frees = []
    
    # 根据 op_type 确定数据大小变量名
    size_vars = []
    if op_type == "ewise":
        size_vars = ["size"]
    elif op_type == "matmul":
        size_vars = ["size1", "size2", "size3"] # M*K, K*N, M*N
    elif op_type in ["pool", "conv"]: # 简化处理，假设也是 size1, size2
        size_vars = [f"size{i+1}" for i in range(len(params))]

    # 生成逻辑
    for i, (p_name, p_type) in enumerate(zip(param_names, param_types)):
        # 移除 const 修饰符以便 malloc
        base_type = p_type.replace("const ", "").replace("*", "").strip()
        d_name = f"d_{p_name}"
        
        # 确定这个参数的大小变量
        # 简单的启发式：如果是输出(最后一个)，通常对应最后一个 size
        # 这里需要根据 op_type 严格对应，下面是 matmul 的逻辑：
        # A(0)->size1, B(1)->size2, C(2)->size3
        current_size = size_vars[0]
        if op_type == "matmul":
            if i < len(size_vars):
                current_size = size_vars[i]
            else:
                current_size = size_vars[-1] # fallback
        elif op_type == "ewise":
             current_size = "size"

        # 1. Malloc Device
        device_allocs.append(f"{base_type}* {d_name} = sycl::malloc_device<{base_type}>({current_size}, q);")
        
        # 2. Memcpy H2D (除了最后一个参数通常是输出)
        if i < len(params) - 1:
            memcpys_h2d.append(f"q.memcpy({d_name}, {p_name}, {current_size} * sizeof({base_type}));")
        
        # 3. Memcpy D2H (只针对最后一个参数)
        if i == len(params) - 1:
            memcpys_d2h.append(f"q.memcpy({p_name}, {d_name}, {current_size} * sizeof({base_type})).wait();")

        # 4. Free
        frees.append(f"sycl::free({d_name}, q);")

    # 3. 准备参数列表供 Kernel 调用
    # Kernel 此时接收的是 Device 指针
    kernel_call_args = ", ".join([f"d_{name}" for name in param_names])
    
    # 额外的 size 参数列表 (int size1, int size2...)
    size_args_def = ", ".join([f"int {s}" for s in size_vars])

    # 4. 组装模板
    # 注意：这里我们使用 intel/llvm 的扩展或者标准 sycl
    sycl_template = Template("""
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
// 包含必要的头文件以支持 half/bfloat16 (如果有)
// #include <sycl/ext/oneapi/bfloat16.hpp>

using namespace sycl;

// 原始 Kernel 代码插入点
// 我们需要把原始代码里的 extern "C" 去掉，防止冲突，因为我们会在外面再包一层
${original_function_clean}

extern "C" {

    // Host 调用的入口函数
    void ${kernel_name}_kernel(${host_args}, ${size_args_def}) {
        try {
            // 创建 Queue (默认选择器会优先选 GPU)
            queue q(default_selector_v);
            
            // 1. 设备内存分配
            ${allocs}
            
            // 2. 数据拷贝 Host -> Device
            ${memcpy_h2d}
            q.wait();

            // 3. 启动 Kernel
            // 如果原始 Kernel 是一个接收指针的 C++ 函数，我们直接调用它
            // 如果原始 Kernel 已经被 loop recovery 变成了串行逻辑，
            // 我们这里其实是在 Host 端调用了一个 "Device Pointer" 的函数。
            // 
            // 【重要】: 这里有一个巨大的假设：
            // 你的 'original_function' 必须包含 parallel_for 逻辑 (即它自己提交任务到 queue)
            // 或者，如果 'original_function' 只是纯计算逻辑 (Loop Recovery 产物)，
            // 那么我们必须在这里构造 parallel_for。
            // 
            // 鉴于 QiMeng-Xpiler 的架构，Loop Recovery 后的代码通常是纯 C++ 循环。
            // 如果是 Tensorization 后的代码，它可能包含了特定的 Intrinsic。
            // 
            // 为了兼容性，如果 Kernel 签名里没有 queue 参数，我们假设它是一个
            // 自包含的函数 (Self-contained) 或者我们在这里不需要 queue (如果是纯 CPU 模拟)。
            // 但如果这是 SYCL 测试，代码里应该已经有了 q.submit(...)
            // 
            // 既然我们要测的是生成的 SYCL 代码，我们假设生成的代码接受 Queue 或者
            // 是一个接受指针的函数，内部自己处理。
            // 但根据之前的讨论，生成的代码可能只是裸的计算函数。
            // 
            // 为了让这个模板通用，我们假设生成的 kernel_name 函数接收的是 device 指针。
            ${kernel_name}(${kernel_call_args});
            
            q.wait();

            // 4. 数据拷贝 Device -> Host
            ${memcpy_d2h}

            // 5. 释放内存
            ${frees}

        } catch (sycl::exception const& e) {
            std::cerr << "SYCL Exception: " << e.what() << std::endl;
            exit(1);
        }
    }
}
""")
    
    # 清理原始代码中的 extern "C" 以避免嵌套错误
    original_function_clean = original_function.replace('extern "C"', "")

    new_code = sycl_template.substitute(
        original_function_clean=original_function_clean,
        kernel_name=kernel_name,
        host_args=", ".join([f"{t} {n}" for t, n in zip(param_types, param_names)]),
        size_args_def=size_args_def,
        allocs="\n            ".join(device_allocs),
        memcpy_h2d="\n            ".join(memcpys_h2d),
        kernel_call_args=kernel_call_args,
        memcpy_d2h="\n            ".join(memcpys_d2h),
        frees="\n            ".join(frees)
    )

    output_file = file_name.replace(".cpp", "_wrapped.cpp")
    with open(output_file, "w") as f:
        f.write(new_code)

    return output_file