import logging
import os
import re

from pycparser import c_ast, c_generator, parse_file


class NodeTransformer(c_ast.NodeVisitor):
    """
    AST 节点转换器类，继承自 pycparser 的 NodeVisitor。
    用于遍历 AST 并允许在遍历过程中修改节点。
    """
    def generic_visit(self, node):
        # 遍历节点的所有字段
        for field, old_value in iter_fields(node):
            # 如果字段值是一个列表（例如代码块中的语句列表）
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    # 如果列表项是 AST 节点，则递归访问它
                    if isinstance(value, c_ast.Node):
                        value = self.visit(value)
                        # 如果 visit 返回 None，说明该节点被删除
                        if value is None:
                            continue
                        # 如果返回的不是节点（可能是节点列表），则将其展开添加到新列表中
                        elif not isinstance(value, c_ast.Node):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                # 更新列表内容
                old_value[:] = new_values
            # 如果字段值是一个函数调用节点
            elif isinstance(old_value, c_ast.FuncCall):
                new_node = self.generic_visit(old_value)
                setattr(node, field, new_node)

            # 如果字段值是普通的 AST 节点
            elif isinstance(old_value, c_ast.Node):
                new_node = self.visit(old_value)
                setattr(node, field, new_node)
        return node


def iter_fields(node):
    """
    迭代 AST 节点的字段。
    由于 pycparser 的节点结构与标准库 ast 模块不同，需要特殊处理。
    """
    # this doesn't look pretty because `pycparser` decided to have structure
    # for AST node classes different from stdlib ones
    index = 0
    children = node.children()
    while index < len(children):
        name, child = children[index]
        try:
            # 尝试查找数组索引标记，处理数组成员
            bracket_index = name.index("[")
        except ValueError:
            # 如果没有中括号，直接 yield 字段名和子节点
            yield name, child
            index += 1
        else:
            # 如果有中括号，截取字段名，获取对应的列表属性
            name = name[:bracket_index]
            child = getattr(node, name)
            # 跳过该列表长度数量的子节点，因为它们已经包含在这个 list 属性中了
            index += len(child)
            yield name, child


def add_memory_prefix(code):
    """
    使用正则表达式为特定命名的 float 变量添加内存修饰符（如 __nram__）。
    主要用于处理特定 AI 加速器（如寒武纪 MLU）的内存空间标记。
    """
    # Define the memory types and their associated prefixes
    prefix_map = {
        "_Nram": "__nram__ float",
        "_Wram": "__wram__ float",
        "_nram": "__nram__ float",
        "_wram": "__wram__ float",
    }

    # Regex pattern to match the variable declarations
    # 匹配未被修饰且以特定后缀（_Nram 等）结尾的 float 变量
    pattern = re.compile(
        r"(?<!__nram__\s)(?<!__wram__\s)float\s+"
        r"(\w+_(?:Nram|Wram|nram|wram|Gdram))\b"
    )

    # Function to replace matched float declarations with the appropriate
    # prefix
    def replacer(match):
        var_name = match.group(1)  # Variable name, such as "lhs_local_Nram"
        suffix = "_" + match.group(1).split("_")[-1]  # "_Nram"
        # If it exists in the map, replace it.
        if suffix in prefix_map:
            return f"{prefix_map[suffix]} {var_name}"
        # Otherwise, keep it as is.
        return match.group(0)

    # Substitute in the code using regex
    modified_code = pattern.sub(replacer, code)
    # 将标准的 memcpy 替换为特定硬件的 __memcpy
    if "memcpy" in modified_code and "__memcpy" not in modified_code:
        modified_code = modified_code.replace("memcpy", "__memcpy")


def add_parallel_variable_prefix(code):
    """
    将通用的并行变量名（如 threadIdxx）转换为 CUDA/HIP 标准格式（如 threadIdx.x）。
    同时恢复 WMMA（Warp Matrix Multiply Accumulate）相关的类型和函数调用。
    通常在代码生成（Backend）阶段使用。
    """
    # 恢复线程索引和块索引的写法
    code = re.sub(r"threadIdxx", "threadIdx.x", code)
    code = re.sub(r"threadIdxy", "threadIdx.y", code)
    code = re.sub(r"threadIdxz", "threadIdx.z", code)
    code = re.sub(r"blockIdxx", "blockIdx.x", code)
    code = re.sub(r"blockIdxy", "blockIdx.y", code)
    code = re.sub(r"blockIdxz", "blockIdx.z", code)
    # 1) Restore the three fragment types by name
    # 恢复 WMMA fragment 类型定义，这是 pycparser 无法处理的 C++ 模板语法
    code = re.sub(
        r"\bwmma_fragment\s+a_frag\b",
        "wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a_frag",
        code,
    )
    code = re.sub(
        r"\bwmma_fragment\s+b_frag\b",
        "wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> b_frag",
        code,
    )
    code = re.sub(
        r"\bwmma_fragment\s+c_frag\b",
        "wmma::fragment<wmma::accumulator,16,16,16,float> c_frag",
        code,
    )

    # 2) Restore the function calls
    # 恢复 WMMA 函数调用，将下划线形式转回命名空间形式
    code = re.sub(r"\bwmma_fill_fragment\b", "wmma::fill_fragment", code)
    code = re.sub(r"\bwmma_load_matrix_sync\b", "wmma::load_matrix_sync", code)
    code = re.sub(r"\bwmma_mma_sync\b", "wmma::mma_sync", code)
    code = re.sub(
        r"\bwmma_store_matrix_sync\b", "wmma::store_matrix_sync", code
    )
    # 修正 store_matrix_sync 的参数，添加内存布局参数
    code = re.sub(
        r"(wmma::store_matrix_sync\s*\([^,]+,\s*[^,]+,\s*[^,]+,\s*)0(\s*\))",
        r"\1wmma::mem_row_major\2",
        code,
    )
    # 如果没有 __global__ 关键字，则添加（默认为内核函数）
    return "__global__ " + code if "__global__ " not in code else code


def remove_target_prefix(code, target=None):
    """
    移除目标特定的语法（如 CUDA/HIP 关键字），将代码标准化为 C 代码。
    这通常用于在解析 AST 之前，因为 pycparser 无法处理 __global__ 或 C++ 模板。
    """
    patterns = [
        (r'extern "C"\s+', ""),  # Remove `extern "C"`.
        (r"\b__nram__\s+", ""),  # Remove `__nram__`.
        (r"\b__wram__\s+", ""),  # Remove `__wram__`.
        (r"__global__\s+", ""),  # Remove `__global__`.
        (r"__launch_bounds__\(\d+\)\s+", ""),  # Remove `__launch_bounds__`.
        (r"\b__restrict__\b", ""),  # Remove `__restrict__`.
        (r"//.*?\n|/\*.*?\*/", "", re.S),  # Remove all C/C++ comments.
        (r"\bthreadIdxx\b", "threadIdxx"),  # Change to underscore style.
        (r"\bthreadIdxy\b", "threadIdxy"),
        (r"\bthreadIdxz\b", "threadIdxz"),
        (r"\bblockIdxx\b", "blockIdxx"),
        (r"\bblockIdxy\b", "blockIdxy"),
        (r"\bblockIdxz\b", "blockIdxz"),
        # 将 static_cast 转换为 C 风格强制转换
        (
            r"static_cast<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>\s*\(([^)]+?)\)",
            r"(\1)(\2)",
        ),
        # 将 reinterpret_cast 转换为 C 风格强制转换
        (r"reinterpret_cast<\s*([^>]+?)\s*>\s*\(([^)]+?)\)", r"(\1)(\2)"),
        # Handle the types in the wmma namespace (template syntax -> C struct type).
        # 将 C++ 模板类型的 wmma fragment 转换为普通的 C 结构体名，以便解析
        (r"wmma::fragment<[^>]+?>", "wmma_fragment"),
        # Handle wmma:: prefix function calls (such as wmma::load_matrix_sync).
        # 将 wmma:: 命名空间调用转换为下划线连接的函数名
        (r"\bwmma::(\w+)", r"wmma_\1"),
    ]

    # Traverse the pattern list and apply replacements.
    # 应用上述所有正则替换
    for pattern, replacement, *flags in patterns:
        code = re.sub(
            pattern, replacement, code, flags=flags[0] if flags else 0
        )

    # 定义需要按需添加的头文件映射
    headers = [
        {
            "header": '#include "stdint.h"',
            "trigger_keywords": ["int8_t", "int32_t"],
        },
        {
            "header": '#include "simd.h"',
            "trigger_keywords": ["__m128i", "_mm_"],
        },
        {
            "header": '#include "simd_cuda.h"',
            "trigger_keywords": [
                "wmma_fragment",
                "wmma_fill_fragment",
                "wmma_load_matrix_sync",
                "wmma_mma_sync",
                "wmma_store_matrix_sync",
            ],
        },
    ]
    lines = code.splitlines()
    existing_includes = set(
        line.strip() for line in lines if line.strip().startswith("#include")
    )

    added_headers = []
    for h in headers:
        needs = any(kw in code for kw in h["trigger_keywords"])
        has_header = h["header"] in existing_includes
        if needs and not has_header:
            added_headers.append(h["header"])

    # 添加必要的头文件
    if added_headers:
        return "\n".join(added_headers) + "\n\n" + code
    else:
        # 如果包含 half 类型，添加 stdhalf.h
        if "half" in code:
            code = '#include "stdhalf.h"' + "\n\n" + code
        return code


def get_target(code, target=None):
    """
    根据代码内容或显式参数确定目标平台（cuda, hip, cpu）及文件后缀。
    """
    # Determine the file type and set the target.
    if target is not None:
        return target, {
            "cuda": ".cu",
            "hip": ".hip",
            "cpu": ".cpp",
        }.get(target, ".cpp")
    # 简单的启发式检测：HIP
    if target == "hip" and ("__global__" in code or "threadIdx.x" in code):
        target, file_type = "hip", ".hip"
    # 启发式检测：CUDA
    elif "__global__" in code or "threadIdx.x" in code or "wmma" in code:
        target, file_type = "cuda", ".cu"
    else:
        target, file_type = "cpu", ".cpp"
    return target, file_type


def make_full_func(code, target=None):
    """
    生成完整的函数代码。
    如果是 CUDA/HIP 目标，会调用 add_parallel_variable_prefix 恢复特定的语法。
    """
    target, _ = get_target(code, target)
    if target in ["cuda", "hip"]:
        code = add_parallel_variable_prefix(code)
    return code


def parse_code_ast(code, target=None):
    """
    解析代码并生成 AST。
    流程：
    1. 移除目标特定的语法 (remove_target_prefix)。
    2. 写入临时文件。
    3. 使用 pycparser 解析（包含伪造的 libc 头文件路径）。
    """
    code = remove_target_prefix(code, target=target)
    filename = "./local_parse_test.c"
    with open(filename, "w") as f:
        f.write(code)
    # Ensure the fake libc include dir is passed as an absolute path so the
    # C preprocessor can find headers like simd_cuda.h that declare
    # `struct dim3` and `threadIdx`/`blockIdx`.
    # 获取伪造的头文件目录，帮助 pycparser 处理标准库和 CUDA 特有类型
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    fake_include = os.path.join(repo_root, "utils", "fake_libc_include")
    try:
        ast = parse_file(
            filename,
            use_cpp=True,
            cpp_path="cpp",
            cpp_args=[f"-I{fake_include}"],
        )
    except Exception as e:
        logging.error(f"parse_file failed: {e}")
        ast = None
    os.remove(filename)
    return ast


def generate_code(ast):
    """
    将 AST 节点转换回 C 代码字符串。
    只处理 AST 中的函数定义 (FuncDef)。
    """
    generator = c_generator.CGenerator()
    # Collect all function definitions in the translation unit
    # 收集所有的函数定义
    func_defs = [ext for ext in ast.ext if isinstance(ext, c_ast.FuncDef)]
    # Generate code for each function and join them with two newlines
    # 生成代码并用空行连接
    all_functions_code = "\n\n".join(
        generator.visit(func) for func in func_defs
    )
    return all_functions_code


def extract_code(content: str):
    """Extract code from LLM output.

    Handles fenced code blocks (```lang\n...```) and unfenced raw function
    definitions (e.g. starting with `extern "C"` or a typical C function
    signature). Returns the extracted code string or None if not found.
    
    从 LLM 的输出文本中提取代码。
    支持 Markdown 代码块和原始 C 函数定义。
    """
    if not content or not isinstance(content, str):
        return None

    # 1) fenced code block
    # 尝试匹配 Markdown 代码块 ```...```
    match = re.search(r"```[a-zA-Z]*\n(.*?)```", content, re.S)
    if match:
        return match.group(1).strip()

    # 2) try to find an unfenced function by locating a C-style function
    # signature and extracting until the matching closing brace.
    # 尝试通过正则匹配 C 风格函数签名来提取未被代码块包裹的代码
    func_sig = re.search(
        r'(extern\s+"C"\s+)?[A-Za-z_][\w\s\*]+\s+[A-Za-z_]\w*\s*\([^)]*\)\s*{',
        content,
    )
    if func_sig:
        start = func_sig.start()
        s = content[start:]
        brace = 0
        end_idx = None
        # 简单的括号计数法来找到函数结束的右大括号
        for i, ch in enumerate(s):
            if ch == "{":
                brace += 1
            elif ch == "}":
                brace -= 1
                if brace == 0:
                    end_idx = i
                    break
        if end_idx is not None:
            return s[: end_idx + 1].strip()
        # fallback: return everything from signature onward
        # 如果找不到闭合括号，返回签名之后的所有内容
        return s.strip()

    # 3) last resort: if it looks like code (contains braces and semicolons),
    # return it
    # 最后的手段：如果内容看起来像代码（包含花括号和分号），直接返回
    if "{" in content and ";" in content:
        return content.strip()

    return None