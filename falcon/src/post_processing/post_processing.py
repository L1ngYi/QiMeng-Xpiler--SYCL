import json
import logging
import re

from falcon.client import invoke_llm
from falcon.src.post_processing.post_processing_prompt import (
    CACHE_READ_PROMPT,
    CACHE_WRITE_PROMPT,
    DECORATION_PROMPT,
    DOUBLE_BUFFER_PROMPT,
    TENSORIZATION_PROMPT,
    THREAD_BINDING_DEMO_CUDA,
    THREAD_BINDING_DEMO_SYCL,
    THREAD_BINDING_PROMPT_CUDA,
    THREAD_BINDING_PROMPT_SYCL,
)
from falcon.src.prompt.prompt import SYSTEM_PROMPT
from falcon.util import extract_code, make_full_func


def _extract_pragma_payloads(code, pragma_name):
    payloads = []
    prefix = f"#pragma {pragma_name}("
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped.startswith(prefix):
            continue
        end_idx = stripped.rfind(")")
        if end_idx <= len(prefix) - 1:
            continue
        payloads.append(stripped[len(prefix):end_idx].strip())
    return payloads


def _split_csv(text):
    return [item.strip() for item in text.split(",") if item.strip()]


def _split_parameters(param_text):
    parts = []
    current = []
    angle_depth = 0
    paren_depth = 0
    bracket_depth = 0

    for ch in param_text:
        if ch == "<":
            angle_depth += 1
        elif ch == ">" and angle_depth > 0:
            angle_depth -= 1
        elif ch == "(":
            paren_depth += 1
        elif ch == ")" and paren_depth > 0:
            paren_depth -= 1
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]" and bracket_depth > 0:
            bracket_depth -= 1

        if (
            ch == ","
            and angle_depth == 0
            and paren_depth == 0
            and bracket_depth == 0
        ):
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue

        current.append(ch)

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_sycl_signature(code):
    signature_source = re.sub(r"//.*?$|/\*.*?\*/", "", code, flags=re.M | re.S)
    match = re.search(
        r'(?P<extern>extern\s+"C"\s+)?'
        r"(?P<ret>[A-Za-z_][\w:\s<>\*&]*?)\s+"
        r"(?P<name>\w+)\s*\((?P<params>.*?)\)\s*{",
        signature_source,
        re.S,
    )
    if match is None:
        return None

    params = _split_parameters(match.group("params"))
    param_types = {}
    scalar_params = []
    queue_name = None
    for param in params:
        name_match = re.search(r"([A-Za-z_]\w*)\s*$", param)
        if name_match is None:
            continue
        param_name = name_match.group(1)
        param_type = param[: name_match.start(1)].strip()
        param_types[param_name] = param_type
        if "queue" in param_type:
            queue_name = param_name
            continue
        if "*" not in param and "&" not in param:
            scalar_params.append(param_name)

    alias_lines = re.findall(
        r"^\s*(using\s+\w+\s*=\s*sycl::[A-Za-z_:<>]+\s*;)\s*$",
        code,
        re.M,
    )

    signature_prefix = (
        f'{match.group("extern") or ""}{match.group("ret").strip()}'
    ).strip()
    return {
        "signature_prefix": signature_prefix,
        "func_name": match.group("name"),
        "params": params,
        "param_types": param_types,
        "scalar_params": scalar_params,
        "queue_name": queue_name,
        "alias_lines": alias_lines,
    }


def _pointee_type(param_type):
    base = param_type.replace("*", " ").replace("&", " ")
    base = re.sub(r"\bconst\b", "", base)
    base = re.sub(r"\s+", " ", base).strip()
    return base or "float"


def _detect_sycl_matmul_kernel(code):
    if "parallel_for" not in code or "sum" not in code:
        return None

    signature = _parse_sycl_signature(code)
    if signature is None or signature["queue_name"] is None:
        return None

    row_match = re.search(
        r"int\s+(?P<row>\w+)\s*=\s*item\.get_(?:global_)?id\(\s*0\s*\)\s*;",
        code,
    )
    col_match = re.search(
        r"int\s+(?P<col>\w+)\s*=\s*item\.get_(?:global_)?id\(\s*1\s*\)\s*;",
        code,
    )
    if row_match is None or col_match is None:
        return None

    reduction_match = re.search(
        r"float\s+sum\s*=\s*0(?:\.0f?)?\s*;\s*"
        r"for\s*\(\s*int\s+(?P<red>\w+)\s*=\s*0\s*;\s*(?P=red)\s*<\s*(?P<kexpr>[^;]+?)\s*;\s*\+\+(?P=red)\s*\)\s*"
        r"{(?P<body>.*?)}\s*"
        r"(?P<output>\w+)\s*\[(?P<out_index>[^\]]+)\]\s*=\s*sum\s*;",
        code,
        re.S,
    )
    if reduction_match is None:
        return None

    loop_body = reduction_match.group("body")
    output_name = reduction_match.group("output")
    array_refs = re.findall(r"([A-Za-z_]\w*)\s*\[([^\]]+)\]", loop_body)
    input_names = []
    for array_name, _ in array_refs:
        if array_name == output_name or array_name in input_names:
            continue
        input_names.append(array_name)
        if len(input_names) == 2:
            break
    if len(input_names) != 2:
        return None

    dim_names = signature["scalar_params"]
    m_expr = dim_names[0] if len(dim_names) > 0 else None
    k_expr = reduction_match.group("kexpr").strip()
    n_expr = dim_names[2] if len(dim_names) > 2 else None

    cond_match = re.search(
        rf"if\s*\(\s*{row_match.group('row')}\s*<\s*(?P<m>[^&|)]+?)\s*&&\s*"
        rf"{col_match.group('col')}\s*<\s*(?P<n>[^&|)]+?)\s*\)",
        code,
    )
    if cond_match is not None:
        m_expr = cond_match.group("m").strip()
        n_expr = cond_match.group("n").strip()
    elif len(dim_names) >= 3:
        m_expr = dim_names[0]
        n_expr = dim_names[2]

    if m_expr is None or n_expr is None:
        range_match = re.search(
            r"parallel_for\s*\(\s*range<2>\s*\(\s*([^,]+)\s*,\s*([^)]+)\)",
            code,
        )
        if range_match is not None:
            m_expr = m_expr or range_match.group(1).strip()
            n_expr = n_expr or range_match.group(2).strip()

    if m_expr is None or n_expr is None:
        return None

    a_name, b_name = input_names
    param_types = signature["param_types"]
    if a_name not in param_types or b_name not in param_types:
        return None
    if output_name not in param_types:
        return None

    return {
        **signature,
        "row_var": row_match.group("row"),
        "col_var": col_match.group("col"),
        "red_var": reduction_match.group("red"),
        "m_expr": m_expr,
        "k_expr": k_expr,
        "n_expr": n_expr,
        "a_name": a_name,
        "b_name": b_name,
        "c_name": output_name,
        "a_elem_type": _pointee_type(param_types[a_name]),
        "b_elem_type": _pointee_type(param_types[b_name]),
        "c_elem_type": _pointee_type(param_types[output_name]),
    }


def _generate_sycl_tiled_matmul(info, use_fma=False):
    subgroup_attr = " [[sycl::reqd_sub_group_size(16)]]" if use_fma else ""
    inner_update = (
        "          sum = sycl::mad(\n"
        "              static_cast<float>(A_tile[local_row][kk]),\n"
        "              static_cast<float>(B_tile[kk][local_col]),\n"
        "              sum);\n"
        if use_fma
        else "          sum += static_cast<float>(A_tile[local_row][kk]) *\n"
        "                 static_cast<float>(B_tile[kk][local_col]);\n"
    )
    alias_block = ""
    if info["alias_lines"]:
        alias_block = "\n".join(info["alias_lines"]) + "\n\n"

    params_text = ", ".join(info["params"])
    return (
        f"{alias_block}"
        f"{info['signature_prefix']} {info['func_name']}({params_text}) {{\n"
        f"  constexpr int TILE_M = 16;\n"
        f"  constexpr int TILE_N = 16;\n"
        f"  constexpr int TILE_K = 16;\n"
        f"  range<2> global_size(\n"
        f"      (({info['m_expr']}) + TILE_M - 1) / TILE_M * TILE_M,\n"
        f"      (({info['n_expr']}) + TILE_N - 1) / TILE_N * TILE_N);\n"
        f"  range<2> local_size(TILE_M, TILE_N);\n\n"
        f"  {info['queue_name']}.submit([&](handler &h) {{\n"
        f"    local_accessor<{info['a_elem_type']}, 2> A_tile(\n"
        f"        range<2>(TILE_M, TILE_K), h);\n"
        f"    local_accessor<{info['b_elem_type']}, 2> B_tile(\n"
        f"        range<2>(TILE_K, TILE_N), h);\n"
        f"    h.parallel_for(\n"
        f"        nd_range<2>(global_size, local_size),\n"
        f"        [=](nd_item<2> item){subgroup_attr} {{\n"
        f"          int {info['row_var']} = item.get_global_id(0);\n"
        f"          int {info['col_var']} = item.get_global_id(1);\n"
        f"          int local_row = item.get_local_id(0);\n"
        f"          int local_col = item.get_local_id(1);\n"
        f"          float sum = 0.0f;\n\n"
        f"          for (int tile_k = 0; tile_k < {info['k_expr']}; tile_k += TILE_K) {{\n"
        f"            int a_col = tile_k + local_col;\n"
        f"            int b_row = tile_k + local_row;\n"
        f"            A_tile[local_row][local_col] =\n"
        f"                ({info['row_var']} < {info['m_expr']} && a_col < {info['k_expr']})\n"
        f"                    ? {info['a_name']}[{info['row_var']} * {info['k_expr']} + a_col]\n"
        f"                    : ({info['a_elem_type']})0;\n"
        f"            B_tile[local_row][local_col] =\n"
        f"                (b_row < {info['k_expr']} && {info['col_var']} < {info['n_expr']})\n"
        f"                    ? {info['b_name']}[b_row * {info['n_expr']} + {info['col_var']}]\n"
        f"                    : ({info['b_elem_type']})0;\n"
        f"            item.barrier(access::fence_space::local_space);\n\n"
        f"#pragma unroll\n"
        f"            for (int kk = 0; kk < TILE_K; ++kk) {{\n"
        f"{inner_update}"
        f"            }}\n"
        f"            item.barrier(access::fence_space::local_space);\n"
        f"          }}\n\n"
        f"          if ({info['row_var']} < {info['m_expr']} && {info['col_var']} < {info['n_expr']}) {{\n"
        f"            {info['c_name']}[{info['row_var']} * {info['n_expr']} + {info['col_var']}] = sum;\n"
        f"          }}\n"
        f"        }});\n"
        f"  }});\n"
        f"}}"
    )


def _promote_sycl_cached_matmul(code):
    if "local_accessor<" not in code:
        return code

    updated = code
    if "[[sycl::reqd_sub_group_size(16)]]" not in updated:
        updated = re.sub(
            r"(\[=\]\s*\(\s*nd_item<2>\s+\w+\s*\))\s*{",
            r"\1 [[sycl::reqd_sub_group_size(16)]] {",
            updated,
            count=1,
        )
    updated = re.sub(
        r"sum\s*\+=\s*static_cast<float>\(([^)]+)\)\s*\*\s*"
        r"static_cast<float>\(([^)]+)\)\s*;",
        (
            "sum = sycl::mad(\n"
            "              static_cast<float>(\\1),\n"
            "              static_cast<float>(\\2),\n"
            "              sum);"
        ),
        updated,
    )
    return updated


def _run_sycl_cache_process(code, space_maps,target_platform):
    if "local_accessor<" in code:
        return make_full_func(code, "sycl")

    info = _detect_sycl_matmul_kernel(code)
    if info is None:
        return make_full_func(code, "sycl")

    return make_full_func(_generate_sycl_tiled_matmul(info), "sycl")


def _run_sycl_tensorization(code):
    if "sycl::mad(" in code and "[[sycl::reqd_sub_group_size(16)]]" in code:
        return make_full_func(code, "sycl")

    promoted = _promote_sycl_cached_matmul(code)
    if promoted != code:
        return make_full_func(promoted, "sycl")

    info = _detect_sycl_matmul_kernel(code)
    if info is None:
        return make_full_func(code, "sycl")

    return make_full_func(_generate_sycl_tiled_matmul(info, use_fma=True), "sycl")

def run_thread_binding(code, target):
    PROMPT = """
    {SYSTEM_PROMPT}

    {THREAD_BINDING_PROMPT}

    Please return the output kernel function without any additional information.
    """

    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)
    prompt_demo = None
    THREAD_BINDING_PROMPT = None

    if target == "cuda" or target == "hip":
        prompt_demo = THREAD_BINDING_DEMO_CUDA
        THREAD_BINDING_PROMPT = THREAD_BINDING_PROMPT_CUDA
    elif target == "sycl":
        prompt_demo = THREAD_BINDING_DEMO_SYCL
        THREAD_BINDING_PROMPT = THREAD_BINDING_PROMPT_SYCL

    PROMPT = PROMPT.replace("{THREAD_BINDING_PROMPT}", THREAD_BINDING_PROMPT)
    PROMPT = PROMPT.replace("{THREAD_BINDING_DEMO}", prompt_demo)
    PROMPT = PROMPT.replace("{cpp_code}", code)

    content = invoke_llm(PROMPT)
    return extract_code(content)


def get_operation_content(code):
    return _extract_pragma_payloads(code, "operation")


def get_input_operand(pragma):
    inputs = pragma.split("input[")[1].split("]")[0]
    input_list = _split_csv(inputs)
    return input_list


def get_output_operand(pragma):
    outputs = pragma.split("output[")[1].split("]")[0]
    output_list = _split_csv(outputs)
    return output_list


def replace_operation_with_intrinsic(code, op_pragma):
    if not op_pragma:
        return code, None
    # Get the list of operations from the code
    op_list = get_operation_content(code)
    space_maps = []
    # Iterate over each operation found in the code
    for op in op_list:
        op_name = op.split("(")[0]
        pragma_pattern = re.escape(f"#pragma operation({op})")
        if op_name not in op_pragma:
            raise KeyError(f"Operation '{op_name}' not found in op_pragma.")

        # Handle input/output existence for op and op_pragma[op_name]
        if "input[" in op:
            input_operands = get_input_operand(op)
        else:
            input_operands = []
        if "output[" in op:
            output_operands = get_output_operand(op)
        else:
            output_operands = []

        pragma_val = op_pragma[op_name]
        if "input[" in pragma_val:
            input_spaces = get_input_operand(pragma_val)
        else:
            input_spaces = []
        if "output[" in pragma_val:
            output_spaces = get_output_operand(pragma_val)
        else:
            output_spaces = []

        if len(input_operands) != len(input_spaces):
            raise ValueError(
                f"Input operands and memory spaces length mismatch for operation '{op_name}' "
                f"({len(input_operands)} operands vs {len(input_spaces)} spaces)."
            )
        if len(output_operands) != len(output_spaces):
            raise ValueError(
                f"Output operands and memory spaces length mismatch for operation '{op_name}' "
                f"({len(output_operands)} operands vs {len(output_spaces)} spaces)."
            )
        input_map = {
            operand: space
            for operand, space in zip(input_operands, input_spaces)
        }
        output_map = {
            operand: space
            for operand, space in zip(output_operands, output_spaces)
        }
        space_map = {"input": input_map, "output": output_map}
        code = re.sub(pragma_pattern, pragma_val, code)
        space_maps.append(space_map)
    return code, space_maps


def get_intrinsic_content(code):
    return _extract_pragma_payloads(code, "intrinsic")


def get_input_memory_spaces(pragma):
    inputs = pragma.split("input[")[1].split("]")[0]
    input_list = _split_csv(inputs)
    return input_list


def get_output_memory_spaces(pragma):
    outputs = pragma.split("output[")[1].split("]")[0]
    output_list = _split_csv(outputs)
    return output_list


def generate_cache_read_prompt(buffer, space, code):
    PROMPT = """
    {SYSTEM_PROMPT}

    {CACHE_READ_PROMPT}

    {CACHE_READ_DEMO}
    Please return the output kernel function without any additional information.
    """
    space_map = {"nram": "__nram__", "wram": "__wram__"}
    NAMESPACE = space_map[space.lower()]

    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)
    PROMPT = PROMPT.replace("{CACHE_READ_PROMPT}", CACHE_READ_PROMPT)
    PROMPT = PROMPT.replace("{buffer}", buffer)
    PROMPT = PROMPT.replace("{CACHE_NAME}", space)
    PROMPT = PROMPT.replace("{CODE}", code)
    PROMPT = PROMPT.replace("{NAMESPACE}", NAMESPACE)
    return PROMPT


def generate_cache_write_prompt(buffer, space, code):
    assert space, "memory space cannot be empty"
    PROMPT = """
    {SYSTEM_PROMPT}
    {CACHE_WRITE_PROMPT}
    {CACHE_WRITE_DEMO}
    Please return the output kernel function without any additional information.
    """
    NAMESPACE = "__nram__"
    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)
    PROMPT = PROMPT.replace("{CACHE_WRITE_PROMPT}", CACHE_WRITE_PROMPT)
    PROMPT = PROMPT.replace("{buffer}", buffer)
    PROMPT = PROMPT.replace("{CACHE_NAME}", space)
    PROMPT = PROMPT.replace("{CODE}", code)
    PROMPT = PROMPT.replace("{NAMESPACE}", NAMESPACE)
    return PROMPT


def run_cache_process(code, space_maps, target):
    # Get the list of intrinsics from the code
    intrinsic_list = get_intrinsic_content(code)
    # Ensure the intrinsic lists and spaces have matching lengths
    if len(intrinsic_list) != len(space_maps):
        raise ValueError(
            f"intrinsics and memory spaces length mismatch for operation"
            f"({len(intrinsic_list)} intrinsics vs {len(space_maps)} spaces)."
        )
    # Iterate over each intrinsic found in the code
    for _, space_map in zip(intrinsic_list, space_maps):
        for key, value in space_map["input"].items():
            logging.info(f"Start cache read: buffer={key}, space={value}")
            cache_read_prompt = generate_cache_read_prompt(key, value, code)
            content = invoke_llm(cache_read_prompt)
            code = extract_code(content)
        for key, value in space_map["output"].items():
            logging.info(f"Start cache write: buffer={key}, space={value}")
            cache_write_prompt = generate_cache_write_prompt(key, value, code)
            content = invoke_llm(cache_write_prompt)
            code = extract_code(content)
    return make_full_func(code, target)


def tensorization(op, code, document):
    PROMPT = """
    {SYSTEM_PROMPT}

    Here is the introduction of Tensorization: {TENSORIZATION_PROMPT}
    Please tensorize the sequential code of {op} below the #pragma operation in {code}
    accordingt to the introduction of tensorized intrinsic.
    {document}
    Please return the output kernel function without any additional information.
    """

    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)
    PROMPT = PROMPT.replace("{TENSORIZATION_PROMPT}", TENSORIZATION_PROMPT)
    PROMPT = PROMPT.replace("{document}", document)
    PROMPT = PROMPT.replace("{code}", code)
    PROMPT = PROMPT.replace("{op}", op)

    content = invoke_llm(PROMPT)
    return extract_code(content)


def get_operation_words(pragma_line):
    return [
        operation.split("(", 1)[0].strip()
        for operation in get_operation_content(pragma_line)
    ]


def run_tensorization(code, target):
    op_list = get_operation_words(code)
    if target in ["cuda", "hip"]:
        if "matmul" not in op_list:
            return code
    return code


def run_code_decoration(code):
    PROMPT = DECORATION_PROMPT.replace("{cpp_code}", code)
    content = invoke_llm(PROMPT)
    decorated_code = extract_code(content)
    return decorated_code if decorated_code else code


def double_buffer(code):
    PROMPT = """
    {SYSTEM_PROMPT}

    Here is the introduction of double buffer: {DOUBLE_BUFFER_PROMPT}
    Please optimize the code snippet below #pragma with double buffer pipeline.

    {code}


    accordingt to the introduction of double buffer.

    {DOUBLE_BUFFER_DEMO}
    Please return the output kernel function without any additional information.
    """

    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)
    PROMPT = PROMPT.replace("{DOUBLE_BUFFER_PROMPT}", DOUBLE_BUFFER_PROMPT)
    PROMPT = PROMPT.replace("{code}", code)
    content = invoke_llm(PROMPT)
    code = extract_code(content)
    return code


def run_double_buffer(code, target):
    code = double_buffer(code)
    return code


def post_processing_pipeline(code, target):
    """This function transforms the given code by performing two main transformations:
        1. Convert parallel loop variables (e.g., OpenMP, cuda) into standard C for loops.
        2. Convert SIMD tensor operations into scalar for-loop based calculations.
    :param func_content: The content of the function (code) to be transformed.

    :return: Transformed code after applying the two transformations."""
    code = run_thread_binding(code, target)

    if target in ["DLBOOST"]:
        code = run_code_decoration(code)
        op_pragma = {}
        code, space_maps = replace_operation_with_intrinsic(code, op_pragma)
        code = run_cache_process(code, space_maps, target)
        code = run_code_decoration(code)
        code = run_tensorization(code, target)
    return code


if __name__ == "__main__":
    pass
