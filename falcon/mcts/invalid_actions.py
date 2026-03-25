from pycparser import c_ast

from falcon.mcts.actions import actions as ActionSpace
from falcon.util import parse_code_ast


class CallNodeTransformer(c_ast.NodeVisitor):
    def __init__(self):
        self.func_call = False

    def visit_FuncCall(self, node):
        self.func_call = True


def visit_func_call(code, target=None):
    ast = parse_code_ast(code, target=target)
    # Count the number of loop layers.
    loop_visitor = CallNodeTransformer()
    loop_visitor.visit(ast)
    return loop_visitor.func_call


class CompoundNodeTransformer(c_ast.NodeVisitor):
    def __init__(self):
        # Used to mark whether a compound statement is encountered.
        self.has_compound_stmt = False

    def visit_Compound(self, node):
        # Check if the compound statement contains multiple statements.
        if len(node.block_items) > 1:
            self.has_compound_stmt = True
        self.generic_visit(node)


def visit_compound_stmt(code, target=None):
    ast = parse_code_ast(code, target=target)
    compound_visitor = CompoundNodeTransformer()
    compound_visitor.visit(ast)
    return compound_visitor.has_compound_stmt

def get_invalid_actions(code, source_platform, target_platform):
    # 初始化 Mask：0 表示允许，1 表示禁止
    # ActionSpace 的顺序对应 actions.py 里的列表：
    # [0:loop_recovery, 1:stmt_split, 2:detensorization, 3:loop_fusion,
    #  4:loop_reorder, 5:loop_split, 6:loop_contraction, 7:auto_bind,
    #  8:auto_cache, 9:auto_tensorization, 10:auto_pipeline, 11:sycl_bind]
    invalid_mask = [0] * len(ActionSpace)
    #print(f"[DEBUG-Platform] source_platform: '{source_platform}', target_platform: '{target_platform}'", flush=True)
    # ================= SYCL source: SYCL → CPU 逻辑 =================
    if source_platform == "sycl":
        # 简单的启发式检查：看代码里还有没有 SYCL 的特征关键字
        is_raw_sycl = "parallel_for" in code or "q.submit" in code or "handler" in code

        if is_raw_sycl:
            # 情况 1: 代码还是原始的 SYCL 代码
            # 策略: 只允许 Action 0 (loop_recovery)，禁止其他所有动作
            invalid_mask = [1] * len(ActionSpace)  # 先全禁
            invalid_mask[0] = 0                    # 独放 Action 0
        else:
            # 情况 2: 代码已经是 Loop Recovery 后的 C++ 代码了
            # 策略: 禁止再次 Loop Recovery，同时也禁止其他优化(暂时)
            invalid_mask = [1] * len(ActionSpace)

        # 直接返回，避免 pycparser 解析 SYCL 代码报错
        return invalid_mask
    # =================================================================

    # ================= CPU → SYCL 逻辑 (新增) ========================
    if target_platform == "sycl":
        is_raw_sycl = "parallel_for" in code or "q.submit" in code or "handler" in code
        if is_raw_sycl:
            invalid_mask = [1] * len(ActionSpace)  # 先全禁
        else:
            invalid_mask = [1] * len(ActionSpace)  # 先全禁
            invalid_mask[7] = 0  # 只允许 auto_bind 
            
        return invalid_mask
    # =================================================================

    # --- 以下是原有的 CUDA/CPU 逻辑 ---

    if source_platform == "cpu":
        invalid_mask[0] = 1

    # 注意：如果代码是 SYCL，程序在上面就已经 return 了，不会执行到这里
    # 从而避免了 pycparser 解析 C++ 报错的问题
    if not visit_func_call(code, source_platform):
        invalid_mask[2] = 1

    if not visit_compound_stmt(code, source_platform):
        invalid_mask[1] = 1

    if target_platform == "cpu":
        # GPU-specific actions are irrelevant when targeting CPU
        invalid_mask[7] = 1   # auto_bind
        invalid_mask[8] = 1   # auto_cache
        if len(invalid_mask) > 10:
            invalid_mask[10] = 1  # auto_pipeline

    if target_platform != "sycl":
        # sycl_bind only makes sense when targeting SYCL
        invalid_mask[10] = 1

    if (
        "coreId" not in code
        and "threadIdx." not in code
        and "blockIdx.x" not in code
    ):
        invalid_mask[0] = 1

    return invalid_mask


if __name__ == "__main__":
    code = """
    int square(int x) {
        return x * x;
    }
    """
    result = visit_func_call(code)
    print(result)

    code = """
    int main() {
        int a = 3;
        int b = square(a);  // <--- 函数调用
        return b;
    }
    """
    result = visit_func_call(code)
    print(result)

    code = """
    int main() {
        int a = 3;
        square(a);  // <--- 函数调用
        return a;
    }
    """
    result = visit_func_call(code)
    print(result)
