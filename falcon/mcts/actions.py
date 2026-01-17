import json

# 导入基于规则（AST/SMT）的转换实现，作为保底方案
from falcon.smt.auto_cache import ast_auto_cache
from falcon.smt.const_inline import constant_inline
from falcon.smt.loop_transformation.loop_contraction import (
    ast_loop_contraction,
)
from falcon.smt.loop_transformation.loop_fusion import ast_loop_fusion
from falcon.smt.loop_transformation.loop_recovery import ast_loop_recovery
from falcon.smt.loop_transformation.loop_reorder import ast_loop_reorder
from falcon.smt.loop_transformation.loop_split import ast_loop_split
from falcon.smt.software_pipeline import smt_double_buffer
from falcon.smt.stmt_split import ast_stmt_split
from falcon.smt.tensorization.detensorization import ast_detensorization
from falcon.smt.tensorization.tensorization import ast_tensorization
from falcon.smt.thread_binding import ast_thread_binding

# 导入基于 LLM（大模型）的转换实现，作为优先方案
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
from falcon.unit_test import unit_test


def loop_recovery(file_name, code, source_platform, target_platform):
    """
    动作：循环恢复
    功能：将特定硬件（如CUDA）的并行代码还原为标准的串行C++循环代码 (IR)。
    策略：优先使用LLM进行语义理解和恢复，失败则使用AST规则。
    """
    try:
        # 尝试使用 LLM 进行恢复
        final_code = run_loop_recovery(code, source_platform)
        # 运行单元测试验证正确性
        if not unit_test(file_name, final_code)[0]:
            raise RuntimeError("loop recovery error")
    except Exception:
        # LLM 失败或校验不通过，回退到 AST 规则恢复
        final_code = ast_loop_recovery(code, source_platform)
    return final_code


def stmt_split(file_name, code, source_platform, target_platform):
    """
    动作：语句拆分
    功能：将复杂的单行语句拆分为多个简单语句，以便后续优化。
    """
    try:
        final_code = run_stmt_split(code)
        if not unit_test(file_name, final_code)[0]:
            raise RuntimeError("stmt split error")
    except Exception:
        final_code = ast_stmt_split(code, source_platform)
    return final_code


def detensorization(file_name, code, source_platform, target_platform):
    """
    动作：去张量化
    功能：将硬件特定的内建函数（如 wmma 及其它 intrinsics）还原为通用的循环计算逻辑。
    """
    try:
        final_code = run_detensorization(code, source_platform)
        if not unit_test(file_name, final_code)[0]:
            raise RuntimeError("detensorization error")
    except Exception:
        final_code = ast_detensorization(code, source_platform)
    return final_code


def loop_fusion(file_name, code, source_platform, target_platform):
    """
    动作：循环融合
    功能：合并相邻的循环以减少循环开销或提高数据局部性。
    """
    try:
        final_code = run_loop_fusion(code)
        if not unit_test(file_name, final_code)[0]:
            raise RuntimeError("loop fusion error")
    except Exception:
        final_code = ast_loop_fusion(code)
    return final_code


def loop_reorder(file_name, code, source_platform, target_platform):
    """
    动作：循环重排序
    功能：改变嵌套循环的顺序（如交换内外层循环），以优化缓存命中率。
    """
    try:
        final_code = run_loop_reorder(code)
        if not unit_test(file_name, final_code)[0]:
            raise RuntimeError("loop_reorder error")
    except Exception:
        final_code = ast_loop_reorder(code)
    return final_code


def loop_split(file_name, code, source_platform, target_platform):
    """
    动作：循环拆分
    功能：将循环拆分为多个部分（如 Strip-mining 或 Tiling）。
    注意：LLM 流程分为两步：1. 标记拆分点 (Annotate) 2. 执行拆分 (Apply)。
    """
    try:
        code = run_split_annotation(code)
        final_code = run_apply_split(code)
        if not unit_test(file_name, final_code)[0]:
            raise RuntimeError("loop_split error")
    except Exception:
        final_code = ast_loop_split(code)
    return final_code


def loop_contraction(file_name, code, source_platform, target_platform):
    """
    动作：循环收缩
    功能：将高维循环展平或合并，通常用于简化结构。
    """
    try:
        final_code = run_loop_contraction(code, None)
        if not unit_test(file_name, final_code)[0]:
            raise RuntimeError("loop_contraction error")
    except Exception:
        final_code = ast_loop_contraction(code)
    return final_code


def auto_bind(file_name, code, source_platform, target_platform):
    """
    动作：自动线程绑定
    功能：将循环迭代变量映射到 GPU 的线程索引 (threadIdx, blockIdx)。
    注意：仅适用于 CUDA/HIP (未来需添加 SYCL 支持)。
    """
    if target_platform not in ["cuda", "hip"]:
        return code
    try:
        final_code = run_thread_binding(code, target_platform)
        if not unit_test(file_name, final_code)[0]:
            raise RuntimeError("auto_bind error")
    except Exception:
        final_code = ast_thread_binding(code, target_platform)
    return final_code


def auto_cache(file_name, code, source_platform, target_platform):
    """
    动作：自动缓存优化
    功能：识别数据访问模式，将全局内存数据预取到 Shared Memory 或 Local Memory。
    """
    # 先进行常量内联和代码装饰预处理
    code = constant_inline(code)
    code = run_code_decoration(code)
    op_pragma = {}
    # 将标准操作替换为 intrinsic 形式，并分析内存空间映射
    code, space_maps = replace_operation_with_intrinsic(code, op_pragma)
    # If no need to cache, just return origin code
    if space_maps is None:
        return code
    try:
        cache_code = run_cache_process(code, space_maps, target_platform)
        if not unit_test(file_name, cache_code):
            raise RuntimeError("auto_cache error")
    except Exception:
        cache_code = ast_auto_cache(code, space_maps)
    return cache_code


def auto_tensorization(file_name, code, source_platform, target_platform):
    """
    动作：自动张量化
    功能：将通用循环逻辑映射回特定硬件的加速指令（如 Tensor Cores 的 wmma）。
    """
    try:
        code = run_code_decoration(code)
        final_code = run_tensorization(code, target_platform)
        if not unit_test(file_name, final_code)[0]:
            raise RuntimeError("auto_tensorization error")
    except Exception:
        final_code = ast_tensorization(code, target_platform)
    return final_code


def auto_pipeline(file_name, code, source_platform, target_platform):
    """
    动作：软件流水线
    功能：(预留接口) 实现双缓冲或软件流水线优化，目前直接返回原代码。
    """
    return code


# 定义 MCTS 的动作空间列表
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
    # 本地测试代码：模拟 MCTS 执行某个动作
    file_name = "benchmark/data/cuda_code_test/add_4_4_4_64.cu"
    from falcon.mcts.utils import open_file

    code = open_file(file_name)
    # 简单的预处理，截取 extern 之前的部分
    code = code.split("extern")[0]
    
    # 测试第一个动作 (action_id=0 即 loop_recovery)
    for action_id in [0]:
        action = actions[action_id]
        code = action(
            file_name,
            code,
            "cuda",
            "cpu",
        )
        print("[INFO]****trans code: ", code)
    from falcon.util import get_target

    target, file_type = get_target(code)
    # 根据结果决定写入的文件后缀
    if target == "cuda":
        new_file = "./tmp/add_4_4_4_64.cu"
    else:
        new_file = "./tmp/add_4_4_4_64.cpp"
    print("[INFO]**********code: ", code)
    with open(new_file, "w", encoding="utf-8") as f:
        f.write(code)
    
    # 计算目标分数（验证性能）
    from falcon.mcts.transcompile import objective

    score = objective(new_file, target)
    print(score)