import json
import logging

from falcon.src.loop_transformation.loop_transformation import (
    run_apply_split,
    run_loop_fusion,
    run_split_annotation,
)
from falcon.src.post_processing.post_processing import (
    replace_operation_with_intrinsic,
    run_cache_process,
    run_code_decoration,
    run_tensorization,
    run_thread_binding,
)

logging.basicConfig(level=logging.INFO)


def run_transcompile_code(file_name, source, target):
    with open(file_name, "r") as f:
        device_code = f.read()
        f.close()
    # loop transformation
    fusion_code = run_loop_fusion(device_code)

    code = run_split_annotation(fusion_code)

    split_code = run_apply_split(code)

    # postprocessing
    final_code = run_thread_binding(split_code, target)
    code = run_code_decoration(final_code)

    op_pragma = json.load(
        open("./falcon/documents/operation_bang_C_instruction_map.json", "r")
    )
    code, space_maps = replace_operation_with_intrinsic(code, op_pragma)
    cache_code = run_cache_process(code, space_maps, target)

    code = run_code_decoration(cache_code)
    final_code = run_tensorization(code, target)
    return final_code


if __name__ == "__main__":
    file_name = "benchmark/data/cpp_code_test/gemm_32_128_128.cpp"
    code = run_transcompile_code(file_name, source="cpp", target="mlu")
    print(code)
