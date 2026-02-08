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
    # [0:loop_recovery, 1:stmt_split, 2:detensorization, 3:loop_fusion, ...]
    invalid_mask = [0] * len(ActionSpace)

    # ================= SYCL 特殊处理逻辑 (新增) =================
    if source_platform == "sycl":
        # 简单的启发式检查：看代码里还有没有 SYCL 的特征关键字
        is_raw_sycl = "parallel_for" in code or "q.submit" in code or "handler" in code

        if is_raw_sycl:
            # 情况 1: 代码还是原始的 SYCL 代码
            # 策略: 只允许 Action 0 (loop_recovery)，禁止其他所有动作
            invalid_mask = [1] * len(ActionSpace) # 先全禁
            invalid_mask[0] = 0                   # 独放 Action 0
        else:
            # 情况 2: 代码已经是 Loop Recovery 后的 C++ 代码了
            # 策略: 禁止再次 Loop Recovery，同时也禁止其他优化(暂时)
            # 这样 MCTS 树在这里就会因为没有有效动作而趋于停止，或者只评估当前结果
            invalid_mask = [1] * len(ActionSpace)
            
            # 如果你未来实现了 CPU 的 OpenMP 优化 (Action X)，可以在这里把 invalid_mask[X] = 0

        # 【重要】直接返回！
        # 绝对不要继续往下走，因为下面的 visit_func_call 会调用 pycparser
        # pycparser 解析 SYCL 代码必挂。
        return invalid_mask
    # ==========================================================

    # --- 以下是原有的 CUDA/CPU 逻辑 (保持不变) ---

    if source_platform == "cpu":
        invalid_mask[0] = 1

    # 注意：如果代码是 SYCL，程序在上面就已经 return 了，不会执行到这里
    # 从而避免了 pycparser 解析 C++ 报错的问题
    if not visit_func_call(code, source_platform):
        invalid_mask[2] = 1

    if not visit_compound_stmt(code, source_platform):
        invalid_mask[1] = 1

    if target_platform == "cpu":
        # 假设 Action 列表里后几个是 GPU 专用操作 (如 bind, cache, tensorize)
        # 根据 actions.py 的定义：
        # 7: auto_bind, 8: auto_cache, 9: auto_tensorization (假设索引)
        # 这里需要确保索引号是对的，原代码写的 7,8,10
        invalid_mask[7] = 1
        invalid_mask[8] = 1
        # invalid_mask[9] = 1 # auto_tensorization
        if len(invalid_mask) > 10:
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
