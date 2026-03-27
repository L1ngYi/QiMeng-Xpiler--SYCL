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
    invalid_mask = [0] * len(ActionSpace)
    if source_platform == "sycl" or target_platform == "sycl":
        invalid_mask = [1] * len(ActionSpace)  # 先全禁
        is_raw_sycl = "parallel_for" in code or "q.submit" in code or "handler" in code
        is_cuda = "blockIdx." in code or "threadIdx." in code or "__global__" in code
        has_tensor = "wmma" in code or "wmma_fragment" in code
        if source_platform == "sycl":
            if target_platform == "sycl":
                # SYCL->SYCL 只允许 auto_cache 和 auto_tensorization
             if "local_accessor" not in code:
                invalid_mask[8] = 0  # 允许 Cache
            if "sycl::mad" not in code and "reqd_sub_group_size" not in code and "joint_matrix" not in code:
                invalid_mask[9] = 0  # 允许 Tensorize
                return invalid_mask 
            if is_raw_sycl:
                invalid_mask[0] = 0  # loop recovery 循环降维
                #invalid_mask[2] = 0 #detensorization 去张量化 未适配，暂时禁用
            else:
                if target_platform != "cpu":
                    if is_cuda:
                        invalid_mask[9] = 0  # auto_tensorization 自动张量化
                        invalid_mask[8] = 0  # auto_cache 自动缓存  
                    else:
                        invalid_mask[7] = 0  # auto_bind 自动绑定
                else:
                    invalid_mask[1] = 0  # stmt_split 语句分解
                    invalid_mask[3] = 0  # loop_fusion 循环融合
                    invalid_mask[4] = 0  # loop_reorder 循环重排
                    invalid_mask[5] = 0  # loop_split 循环分割
                    invalid_mask[6] = 0  # loop_contraction 循环收缩

        if target_platform == "sycl":
            if is_cuda:
                if has_tensor:
                    invalid_mask[2] = 0 #detensorization 去张量化
                else:
                    invalid_mask[0] = 0  # loop recovery 循环降维
            elif is_raw_sycl:
                invalid_mask[8] = 0  # auto_cache
                invalid_mask[9] = 0  # auto_tensorization
            else:
                invalid_mask[7] = 0  # auto_bind 自动绑定
                invalid_mask[1] = 0  # stmt_split 语句分解
                invalid_mask[3] = 0  # loop_fusion 循环融合
                invalid_mask[4] = 0  # loop_reorder 循环重排
                invalid_mask[5] = 0  # loop_split 循环分割
                invalid_mask[6] = 0  # loop_contraction 循环收缩
                
        return invalid_mask

    # --- 以下是原有的 CUDA/CPU 逻辑 ---

    if source_platform == "cpu":
        invalid_mask[0] = 1

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
