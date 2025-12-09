import json
import random

from falcon.src.loop_transformation.loop_transformation import (
    run_apply_split,
    run_loop_contraction,
    run_loop_fusion,
    run_loop_reorder,
    run_split_annotation,
    run_stmt_split,
)
from falcon.src.post_processing.post_processing import (
    replace_operation_with_intrinsic,
    run_cache_process,
    run_code_decoration,
    run_double_buffer,
    run_tensorization,
    run_thread_binding,
)
from falcon.src.pre_processing.preprocessing import (
    run_detensorization,
    run_loop_recovery,
)


def loop_recovery(file_name, code, source_platform, target_platform):
    final_code = run_loop_recovery(code, source_platform)
    return final_code


def stmt_split(file_name, code, source_platform, target_platform):
    return run_stmt_split(code)


def detensorization(file_name, code, source_platform, target_platform):
    final_code = run_detensorization(code, source_platform)
    return final_code


def loop_fusion(file_name, code, source_platform, target_platform):
    final_code = run_loop_fusion(code)
    return final_code


def loop_reorder(file_name, code, source_platform, target_platform):
    final_code = run_loop_reorder(code)
    return final_code


def loop_split(file_name, code, source_platform, target_platform):
    code = run_split_annotation(code)
    final_code = run_apply_split(code)
    return final_code


def loop_contraction(file_name, code, source_platform, target_platform):
    final_code = run_loop_contraction(code, None)
    return final_code


def auto_bind(file_name, code, source_platform, target_platform):
    if target_platform not in ["cuda", "hip"]:
        return code
    final_code = run_thread_binding(code, target_platform)
    return final_code


def auto_cache(file_name, code, source_platform, target_platform):
    code = run_code_decoration(code)
    op_pragma = {}
    code, space_maps = replace_operation_with_intrinsic(code, op_pragma)
    # If no need to cache, just return origin code
    if space_maps is None:
        return code

    cache_code = run_cache_process(code, space_maps, target_platform)
    return cache_code


def auto_tensorization(file_name, code, source_platform, target_platform):
    code = run_code_decoration(code)
    final_code = run_tensorization(code, target_platform)
    return final_code


def auto_pipeline(file_name, code, source_platform, target_platform):
    return code


actions = [
    loop_recovery,
    stmt_split,
    detensorization,
    loop_fusion,
    loop_reorder,
    loop_split,
    loop_contraction,
    auto_bind,
    auto_cache,
    auto_tensorization,
    auto_pipeline,
]

if __name__ == "__main__":
    pass
