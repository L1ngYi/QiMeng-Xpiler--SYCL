import re
from string import Template

def _parse_param_decl(param_decl):
    normalized = (
        param_decl.replace("*", " * ").replace("&", " & ").strip()
    )
    tokens = normalized.split()
    if not tokens:
        raise ValueError("Encountered an empty SYCL parameter declaration.")

    var_name = tokens[-1]
    type_tokens = tokens[:-1]
    is_pointer = "*" in type_tokens
    dtype = " ".join(t for t in type_tokens if t not in {"*", "&"}).strip()
    storage_dtype = re.sub(r"\bconst\b", "", dtype)
    storage_dtype = re.sub(r"\s+", " ", storage_dtype).strip()

    return {
        "name": var_name,
        "dtype": re.sub(r"\s+", " ", dtype).strip(),
        "storage_dtype": storage_dtype,
        "full": param_decl.strip(),
        "is_pointer": is_pointer,
    }


def parse_sycl_function_metadata(original_function):
    """Extract the callable SYCL kernel signature from a source string."""
    function_signature_pattern = r"void\s+(\w+)\(([^)]*)\)"
    match = re.search(function_signature_pattern, original_function, re.DOTALL)
    if not match:
        raise ValueError("Could not find SYCL function signature.")

    kernel_name = match.group(1)
    param_list_str = match.group(2)
    raw_params = [param.strip() for param in param_list_str.split(",") if param.strip()]
    parsed_params = [_parse_param_decl(param) for param in raw_params]

    data_params = [param for param in parsed_params if "queue" not in param["dtype"]]
    pointer_params = [param for param in data_params if param["is_pointer"]]
    scalar_params = [param for param in data_params if not param["is_pointer"]]

    return {
        "kernel_name": kernel_name,
        "raw_params": parsed_params,
        "data_params": data_params,
        "pointer_params": pointer_params,
        "scalar_params": scalar_params,
    }


def get_sycl_function_metadata(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        original_function = f.read()
    metadata = parse_sycl_function_metadata(original_function)
    metadata["original_function"] = original_function
    return metadata


def get_sycl_shape_from_file_name(file_name):
    base_name = file_name.rsplit("/", 1)[-1]
    name_without_ext = base_name.rsplit(".", 1)[0]
    shape_tokens = name_without_ext.split("_")[1:]
    if len(shape_tokens) < 3:
        raise ValueError(
            f"Invalid matmul shape encoded in file name: {base_name}"
        )
    return [int(token) for token in shape_tokens[:3]]


def get_sycl_matmul_size_exprs(metadata, shape=None):
    pointer_params = metadata["pointer_params"]
    scalar_params = metadata["scalar_params"]
    if len(pointer_params) != 3:
        raise NotImplementedError(
            "SYCL matmul wrappers expect exactly 3 pointer params."
        )

    if len(scalar_params) == 3:
        m_name = scalar_params[0]["name"]
        k_name = scalar_params[1]["name"]
        n_name = scalar_params[2]["name"]
        return [
            f"{m_name} * {k_name}",
            f"{k_name} * {n_name}",
            f"{m_name} * {n_name}",
        ]

    if scalar_params:
        raise NotImplementedError(
            "SYCL matmul wrappers support either 0 scalar dims or exactly 3."
        )

    if shape is None:
        raise NotImplementedError(
            "SYCL matmul wrappers need either 3 scalar dims or a shape fallback."
        )

    m_dim, k_dim, n_dim = shape[:3]
    return [str(m_dim * k_dim), str(k_dim * n_dim), str(m_dim * n_dim)]


def get_sycl_invocation_args(metadata, pointer_name_map=None, queue_name="q"):
    pointer_name_map = pointer_name_map or {}
    call_args = []
    for param in metadata["raw_params"]:
        if "queue" in param["dtype"]:
            call_args.append(queue_name)
        elif param["is_pointer"]:
            call_args.append(pointer_name_map.get(param["name"], param["name"]))
        else:
            call_args.append(param["name"])
    return call_args


def create_sycl_func(file_name, op_type="ewise", shape=None):
    """
    读取 SYCL 源代码，生成包含 Host 端调用的完整 C++ 文件。
    """
    metadata = get_sycl_function_metadata(file_name)
    original_function = metadata["original_function"]
    kernel_name = metadata["kernel_name"]
    data_params = metadata["data_params"]
    pointer_params = metadata["pointer_params"]

    # 3. 生成内存管理代码
    device_memory_alloc = [] 
    memcpy_htod = []        

    # 根据 op_type 生成 Size 逻辑 (复用你 CUDA 模板的逻辑)
    if op_type == "matmul":
        if shape is None:
            shape = get_sycl_shape_from_file_name(file_name)
        size_exprs = get_sycl_matmul_size_exprs(metadata, shape=shape)
        
        # 分配内存 & Host->Device 拷贝
        for i, item in enumerate(pointer_params):
            name = item["name"]
            storage_dtype = item["storage_dtype"] # e.g. half
            
            # 声明设备指针
            device_memory_alloc.append(f"{storage_dtype} *{name}_sycl;")
            # Malloc Device
            device_memory_alloc.append(
                f"{name}_sycl = sycl::malloc_device<{storage_dtype}>({size_exprs[i]}, q);"
            )
            
            # Memcpy H->D (除了最后一个输出 C)
            if i < len(pointer_params) - 1:
                memcpy_htod.append(
                    f"q.memcpy({name}_sycl, {name}, {size_exprs[i]} * sizeof({storage_dtype}));"
                )

        # Memcpy D->H (只拷贝最后一个参数 C)
        last_item = pointer_params[-1]
        memcpy_dtoh = (
            f"q.memcpy({last_item['name']}, {last_item['name']}_sycl, "
            f"{size_exprs[-1]} * sizeof({last_item['storage_dtype']}));"
        )
    
    else:
        # 这里你可以补充 ewise 等其他逻辑，结构同上
        raise NotImplementedError("Currently only matmul is supported in this SYCL template.")

    # 4. 构造模板
    # 生成 extern "C" 的参数列表：原始指针参数 + 原始标量参数（去掉 queue）
    extern_c_params = ", ".join([param["full"] for param in data_params])
    pointer_name_map = {
        param["name"]: f"{param['name']}_sycl" for param in pointer_params
    }

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
        queue q;

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
        called_args=", ".join(
            get_sycl_invocation_args(metadata, pointer_name_map=pointer_name_map)
        ),
        memcpy_dtoh_code=memcpy_dtoh,
        free_code="\n        ".join([f"sycl::free({p['name']}_sycl, q);" for p in pointer_params])
    )

    # 写入 _wrapped.cpp
    output_file = file_name.replace(".cpp", "_wrapped.cpp")
    with open(output_file, "w") as f:
        f.write(new_code)

    return output_file
