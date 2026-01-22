import re
from string import Template

def create_sycl_func(file_name, op_type="ewise"):
    """
    读取 SYCL 源代码，生成包含 Host 端调用的完整 C++ 文件。
    """
    with open(file_name, "r") as f:
        original_function = f.read()

    # 1. 解析函数签名
    # 匹配类似: void gemm(half *A, ..., queue &q)
    # 我们假设函数返回 void，且最后一个参数可能是 queue &q (我们会在生成调用时处理)
    function_signature_pattern = r"void\s+(\w+)\(([^)]*)\)"
    match = re.search(function_signature_pattern, original_function, re.DOTALL)
    if not match:
        raise ValueError("Could not find SYCL function signature.")

    kernel_name = match.group(1)
    param_list_str = match.group(2)

    # 2. 清洗参数
    raw_params = [param.strip() for param in param_list_str.split(",")]
    
    # 分离出真正的数据参数（排除最后的 queue &q）
    data_params = []
    for p in raw_params:
        if "queue" in p:
            continue
        data_params.append(p)
        
    # 提取参数名和类型
    # data_params 例子: ["half *A", "half *B", "float *C", "int m", ...]
    param_names = []
    pointer_params = [] # 需要分配内存的指针参数
    scalar_params = []  # 标量参数 (int m 等)
    
    for p in data_params:
        # 简单提取变量名 (取最后一个空格后的部分，去掉 * 和 &)
        parts = p.split()
        var_name = parts[-1].replace("*", "").replace("&", "").strip()
        dtype = " ".join(parts[:-1]) # 类型部分
        
        param_names.append(var_name)
        
        if "*" in p:
            pointer_params.append({"name": var_name, "dtype": dtype, "full": p})
        else:
            scalar_params.append(var_name)

    # 3. 生成内存管理代码
    device_memory_alloc = [] 
    memcpy_htod = []        
    device_vars = []        # 调用 kernel 时传入的参数列表
    
    # 构造 kernel 调用时的参数列表
    # 指针参数变成 name_sycl，标量参数保持原名
    for p in data_params:
        var_name = p.split()[-1].replace("*", "").replace("&", "").strip()
        if "*" in p:
            device_vars.append(f"{var_name}_sycl")
        else:
            device_vars.append(var_name)
    
    # 既然你的 kernel 定义里有 queue &q，我们调用时必须把它加进去
    device_vars.append("q")

    # 根据 op_type 生成 Size 逻辑 (复用你 CUDA 模板的逻辑)
    size_vars = []
    if op_type == "matmul":
        size_vars = ["size1", "size2", "size3"] # M*K, K*N, M*N
        
        # 分配内存 & Host->Device 拷贝
        for i, item in enumerate(pointer_params):
            name = item["name"]
            dtype = item["dtype"] # e.g. half
            
            # 声明设备指针
            device_memory_alloc.append(f"{dtype} *{name}_sycl;")
            # Malloc Device
            device_memory_alloc.append(
                f"{name}_sycl = sycl::malloc_device<{dtype}>({size_vars[i]}, q);"
            )
            
            # Memcpy H->D (除了最后一个输出 C)
            if i < len(pointer_params) - 1:
                memcpy_htod.append(
                    f"q.memcpy({name}_sycl, {name}, {size_vars[i]} * sizeof({dtype}));"
                )

        # Memcpy D->H (只拷贝最后一个参数 C)
        last_item = pointer_params[-1]
        memcpy_dtoh = f"q.memcpy({last_item['name']}, {last_item['name']}_sycl, size3 * sizeof({last_item['dtype']}));"
    
    else:
        # 这里你可以补充 ewise 等其他逻辑，结构同上
        raise NotImplementedError("Currently only matmul is supported in this SYCL template.")

    # 4. 构造模板
    # 生成 extern "C" 的参数列表：原始指针参数 + size 参数
    # 例如: half *A, half *B, float *C, int m, int k, int n, int size1, int size2, int size3
    extern_c_params = ", ".join(data_params) + ", " + ", ".join(["int " + s for s in size_vars])

    host_func_template = Template(
"""
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

using namespace sycl;

// Original Kernel implementation (User provided)
${original_function}

extern "C" void ${kernel_name}_kernel(${extern_c_params}) {
    try {
        // 1. Create Queue
        queue q(default_selector_v);

        // 2. Device Allocation
        ${alloc_code}

        // 3. Memcpy Host -> Device
        ${memcpy_htod_code}
        q.wait();

        // 4. Call Kernel
        // We pass the device pointers and the queue
        ${kernel_name}(${called_args});
        q.wait();

        // 5. Memcpy Device -> Host
        ${memcpy_dtoh_code}
        q.wait();

        // 6. Free
        ${free_code}

    } catch (sycl::exception const &e) {
        std::cerr << "[SYCL Wrapper Error] " << e.what() << std::endl;
    }
}
"""
    )

    new_code = host_func_template.substitute(
        original_function=original_function,
        kernel_name=kernel_name,
        extern_c_params=extern_c_params,
        alloc_code="\n        ".join(device_memory_alloc),
        memcpy_htod_code="\n        ".join(memcpy_htod),
        called_args=", ".join(device_vars),
        memcpy_dtoh_code=memcpy_dtoh,
        free_code="\n        ".join([f"sycl::free({p['name']}_sycl, q);" for p in pointer_params])
    )

    # 写入 _wrapped.cpp
    output_file = file_name.replace(".cpp", "_wrapped.cpp")
    with open(output_file, "w") as f:
        f.write(new_code)

    return output_file