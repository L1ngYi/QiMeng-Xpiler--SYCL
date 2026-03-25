import re

from pycparser import c_ast, c_generator
from falcon.util import NodeTransformer, generate_code, parse_code_ast
import sys
import clang.cindex
from clang.cindex import CursorKind, TokenKind
import tempfile
import os
from falcon.simplification import simplify_code
from falcon.smt.const_inline import constant_inline
from falcon.stmt_simplification import ast_stmt_simplification
from falcon.util import (
    NodeTransformer,
    generate_code,
    make_full_func,
    parse_code_ast,
    remove_target_prefix,
)

ParaVar = {
    "threadIdx.x": 1024,
    "blockIdx.x": 256,
    "coreId": 4,
    "clusterId": 4,
    "threadIdx.y": 1024,
    "blockIdx.y": 256,
    "threadIdx.z": 1024,
    "blockIdx.z": 256,
}

cuda_paravar = [
    "threadIdx.x",
    "threadIdx.y",
    "threadIdx.z",
    "blockIdx.x",
    "blockIdx.y",
    "blockIdx.z",
]


def update_dim(cuda_code):
    """The re module in Python is used to write a regular expression that
    matches the number inside the parentheses."""
    match = re.search(r"__launch_bounds__\((\d+)\)", cuda_code)
    if match:
        # Print the matching values
        launch_bounds_value = int(match.group(1))
        ParaVar["threadIdx.x"] = launch_bounds_value
    return ParaVar

class SyclKernelExtractor:
    def __init__(self, file_path, source_code):
        self.file_path = file_path
        self.source_code = source_code
        # 【关键修复】：将源码转成字节流，专门应付底层返回的 byte offset
        self.source_code_bytes = source_code.encode('utf-8')
        self.index = clang.cindex.Index.create()
        self.tu = self.index.parse(
            file_path,
            args=['-std=c++17', '-x', 'c++', '-w', '-fparse-all-comments']
        )

    def _get_text(self, cursor):
        start = cursor.extent.start.offset
        end = cursor.extent.end.offset
        # 【关键修复】：在 byte 数组上切片，然后安全解码！免疫所有中文注释错位！
        return self.source_code_bytes[start:end].decode('utf-8', 'ignore')

    def find_kernels(self):
        kernels = []
        self._recursive_find_submit(self.tu.cursor, kernels)
        return kernels

    def _recursive_find_submit(self, cursor, results):
        #if cursor.kind in [CursorKind.CALL_EXPR, CursorKind.UNEXPOSED_EXPR, CursorKind.CXX_MEMBER_CALL_EXPR]:
        if cursor.kind in [CursorKind.CALL_EXPR, CursorKind.UNEXPOSED_EXPR]:
            text = self._get_text(cursor)
            # 定位 q.submit() 调用
            if '.submit' in text.split('(')[0] or 'submit' in text.split('(')[0]:
                sub_kernels = []
                self._recursive_find_parallel_for(cursor, sub_kernels)
                if sub_kernels:
                    for k in sub_kernels:
                        k['submit_node'] = cursor
                        results.append(k)
                    return
        
        for child in cursor.get_children():
            self._recursive_find_submit(child, results)

    def _recursive_find_parallel_for(self, cursor, results):
        #if cursor.kind in [CursorKind.CALL_EXPR, CursorKind.UNEXPOSED_EXPR, CursorKind.CXX_MEMBER_CALL_EXPR]:
        if cursor.kind in [CursorKind.CALL_EXPR, CursorKind.UNEXPOSED_EXPR]:
            text = self._get_text(cursor)
            if 'parallel_for' in text.split('(')[0]:
                try:
                    kernel_info = self._analyze_kernel_invocation(cursor)
                    if kernel_info:
                        results.append(kernel_info)
                except Exception as e:
                    print(f"[Warn] 解析 Kernel 失败: {e}")
                return
        
        for child in cursor.get_children():
            self._recursive_find_parallel_for(child, results)

    def _analyze_kernel_invocation(self, call_node):
        call_text = self._get_text(call_node)
        lambda_node = None
        range_node = None
        
        for child in call_node.get_children():
            clean_type = child.type.spelling.replace("const", "").replace("&", "").strip()
            if "range" in clean_type or "nd_range" in clean_type:
                range_node = child
            elif child.kind == CursorKind.LAMBDA_EXPR or "lambda" in clean_type:
                lambda_node = self._drill_down_to_lambda(child)
            elif child.kind == CursorKind.UNEXPOSED_EXPR:
                if "range" in self._get_text(child):
                    range_node = child
                elif "{" in self._get_text(child) and "[" in self._get_text(child):
                    lambda_node = child

        if not range_node or not lambda_node:
            return None

        dims, bounds = self._parse_range(range_node)
        kernel_data = self._analyze_lambda(lambda_node, dims)
        
        return {
            "dims": dims,
            "bounds": bounds,
            "body": kernel_data['body'],
            "index_var": kernel_data['index_var']
        }

    def _drill_down_to_lambda(self, node):
        if node.kind == CursorKind.LAMBDA_EXPR: return node
        if "{" in self._get_text(node):
            for child in node.get_children():
                if child.kind == CursorKind.LAMBDA_EXPR: return child
            return node
        return node

    def _parse_range(self, range_node):
        text = self._get_text(range_node)
        dims = 1
        if "<2>" in text: dims = 2
        elif "<3>" in text: dims = 3
        
        # 智能上下文回溯：解决变量传递问题 (如 global_size(m, n))
        range_map = {}
        for match in re.finditer(r'range<\d+>\s+(\w+)\s*\(([^)]+)\)', self.source_code):
            var_name = match.group(1)
            args = [arg.strip() for arg in match.group(2).split(',')]
            range_map[var_name] = args
            
        bounds = []
        direct_match = re.search(r'(?:nd_)?range<\d+>\s*\(([^)]+)\)', text)
        if direct_match:
            args = [arg.strip() for arg in direct_match.group(1).split(',')]
            # 如果参数是一个已知变量名，展开它
            if args[0] in range_map:
                bounds = range_map[args[0]]
            else:
                bounds = args
        
        if not bounds:
            bounds = ['N', 'M', 'K'][:dims]
            
        return dims, bounds

    def _analyze_lambda(self, lambda_node, dims):
        index_var = "item"
        lambda_text = self._get_text(lambda_node)
        idx_match = re.search(r'\(\s*(?:nd_)?item<\d+>\s+(\w+)\s*\)', lambda_text)
        if idx_match:
            index_var = idx_match.group(1)
        
        body_node = None
        for child in lambda_node.get_children():
            if child.kind == CursorKind.COMPOUND_STMT:
                body_node = child
        
        if not body_node:
            start_idx = lambda_text.find('{')
            end_idx = lambda_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                raw_body = lambda_text[start_idx:end_idx+1]
            else:
                raise ValueError("Lambda body not found")
        else:
            raw_body = self._get_text(body_node)

        trans_body = self._rewrite_body_tokens(raw_body, index_var, dims)

        return {
            "index_var": index_var,
            "body": trans_body
        }

    def _rewrite_body_tokens(self, body_text, index_var, dims):
        # 使用不易冲突的专属变量名
        loop_vars = ["__sycl_i", "__sycl_j", "__sycl_k"]
        
        for i in range(dims):
            # 替换 item.get_global_id(0) -> __sycl_i
            pattern = rf'{index_var}\s*\.\s*get_global_id\s*\(\s*{i}\s*\)'
            body_text = re.sub(pattern, loop_vars[i], body_text)
            
            # 替换 item[0] -> __sycl_i
            pattern_idx = rf'{index_var}\s*\[\s*{i}\s*\]'
            body_text = re.sub(pattern_idx, loop_vars[i], body_text)
        
        return body_text.strip().lstrip("{").rstrip("}")

    def generate_loops_only(self, kernel_info):
        dims = kernel_info['dims']
        bounds = kernel_info['bounds']
        
        loops = ""
        indent = "    "
        loop_vars = ["__sycl_i", "__sycl_j", "__sycl_k"]
        
        for d in range(dims):
            limit = bounds[d] if d < len(bounds) else "N"
            var = loop_vars[d]
            loops += f"{indent * (d+1)}for (int {var} = 0; {var} < {limit}; ++{var}) {{\n"
        
        body_lines = kernel_info['body'].splitlines()
        formatted_body = "\n".join([f"{indent * (dims+1)}{line.strip()}" for line in body_lines if line.strip()])
        
        closing = ""
        for d in reversed(range(dims)):
             closing += f"{indent * (d+1)}}}\n"

        return f"{loops}{formatted_body}\n{closing}"

class LoopRecoveryVisitor(NodeTransformer):
    def __init__(self, variable_map):
        self.variable_map = variable_map

    def visit_FuncDef(self, node):
        self.visit(node.body)
        body_node = node.body
        for var, ext in self.variable_map.items():
            init_node = c_ast.Decl(
                name=var.replace(".", ""),
                quals=[],
                align=[],
                storage=[],
                funcspec=[],
                type=c_ast.TypeDecl(
                    declname=var.replace(".", ""),
                    quals=[],
                    align=None,
                    type=c_ast.IdentifierType(["int"]),
                ),
                init=c_ast.Constant("int", "0"),
                bitsize=None,
            )
            cond_node = c_ast.BinaryOp(
                "<",
                c_ast.ID(var.replace(".", "")),
                c_ast.Constant("int", ext),
            )
            next_node = c_ast.UnaryOp("++", c_ast.ID(var.replace(".", "")))

            inner_loop = c_ast.For(
                init=init_node, cond=cond_node, next=next_node, stmt=body_node
            )
            body_node = c_ast.Compound(block_items=[inner_loop])

        node = c_ast.FuncDef(
            decl=node.decl, param_decls=node.param_decls, body=body_node
        )
        return node

    def visit_StructRef(self, node):
        if node.name.name in ["threadIdx", "blockIdx"]:
            name = node.name.name
            filed = node.field.name
            return c_ast.ID(name=name + filed)
        elif node.name.name in ["blockDim"]:
            name = node.name.name + node.field.name
            return c_ast.Constant("int", value=str(1024))
        else:
            return self.generic_visit(node)


def ast_sycl_loop_recovery(code):
    import tempfile
    import os
    import re
    
    try:
        mock_sycl = """
#include <stdint.h>
namespace sycl {
    template <int dimensions = 1> struct range { range(int, int=1, int=1){} };
    template <int dimensions = 1> struct nd_range { nd_range(range<dimensions>, range<dimensions>){} };
    template <int dimensions = 1> struct id { int operator[](int) const; };
    template <int dimensions = 1> struct item { int get_global_id(int) const; };
    template <int dimensions = 1> struct nd_item { int get_global_id(int) const; };
    struct handler {
        template<typename T, typename F> void parallel_for(T, F) {}
        template<typename T, typename F> void parallel_for(nd_range<2>, F) {}
    };
    struct queue {
        template<typename F> void submit(F) {}
        void wait() {}
    };
    typedef uint16_t half;
}
"""
        code_without_include = re.sub(r'#include\s*<sycl/sycl\.hpp>', '', code)
        mocked_code = mock_sycl + "\n" + code_without_include
        
        fd, path = tempfile.mkstemp(suffix=".cpp")
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as tmp:
                tmp.write(mocked_code)
            
            # 这里保留了我们解决 UTF-8 偏移量的关键修复！
            extractor = SyclKernelExtractor(path, mocked_code) 
            kernels = extractor.find_kernels()
            
            if not kernels:
                raise RuntimeError("SYCL AST Parsing failed: No kernels found in source.")
                
            final_code = mocked_code
            for k in kernels:
                loops_code = extractor.generate_loops_only(k)
                submit_text = extractor._get_text(k['submit_node'])
                final_code = final_code.replace(submit_text, loops_code)
                
            # 清理和纯化
            final_code = final_code.replace(mock_sycl + "\n", "")
            final_code = final_code.replace("using namespace sycl;", "")
            final_code = final_code.replace("using half = sycl::half;", "typedef uint16_t half;")
            final_code = re.sub(r',\s*(?:sycl::)?queue\s*[*&]?\s*\w+', '', final_code)
            final_code = re.sub(r'(?:sycl::)?queue\s*[*&]?\s*\w+\s*,', '', final_code)
            final_code = re.sub(r'^(\s*)(?:sycl::)?(?:nd_)?range<.*?;', r'\1// [Removed by AST] \g<0>', final_code, flags=re.MULTILINE)
            
            if "<stdint.h>" not in final_code and "<cstdint>" not in final_code:
                final_code = "#include <stdint.h>\n" + final_code
            if "<vector>" not in final_code:
                final_code = "#include <vector>\n" + final_code
                
            return final_code
            
        finally:
            if os.path.exists(path):
                os.remove(path)
                
    except Exception as e:
        print(f"[Error] SYCL AST Logic Failed: {e}")
        raise e


def ast_loop_recovery(code, target="cuda"):
    if target == "sycl":
        return ast_sycl_loop_recovery(code)
    
    ParaVar = update_dim(code)
    code = remove_target_prefix(code)
    builtin_map = {}
    if target == "cuda" or target == "hip":
        for builtin_var in cuda_paravar:
            if builtin_var in code:
                builtin_map[builtin_var] = ParaVar[builtin_var]

    # insert the parallel loop
    ast = parse_code_ast(code)
    visitor = LoopRecoveryVisitor(builtin_map)
    visitor.visit(ast)
    code = generate_code(ast)
    code = simplify_code(code)
    code = constant_inline(code)
    code = ast_stmt_simplification(code)
    if target == "cuda" or target == "hip":
        code = code.replace("threadIdx.x", "threadIdxx")
        code = code.replace("threadIdx.y", "threadIdxy")
        code = code.replace("threadIdx.z", "threadIdxz")
        code = code.replace("blockIdx.x", "blockIdxx")
        code = code.replace("blockIdx.y", "blockIdxy")
        code = code.replace("blockIdx.z", "blockIdxz")
        return "__global__ " + code if "__global__ " not in code else code


if __name__ == "__main__":
    cuda_code = """
    void add(float*  A, float*  B, float*  T_add) {
        if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 2309) {
            T_add[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (A[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + B[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
        }
    }
    """
    converted_code = ast_loop_recovery(cuda_code, "cuda")
    print(converted_code)

    cuda_code = """
    extern "C" __global__ void __launch_bounds__(960) add(float* __restrict__ A, float* __restrict__ B, float* __restrict__ T_add) {
        T_add[((int)threadIdx.x)] = (A[((int)threadIdx.x)] + B[((int)threadIdx.x)]);
    }
    """
    converted_code = ast_loop_recovery(cuda_code, "cuda")
    print(converted_code)

    cuda_code = """
    __global__ void gemm(float *A, float *B, float *C) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        int col = blockIdx.y * blockDim.y + threadIdx.y;

        if (row < 32 && col < 128) {
            float sum = 0.0f;
            for (int i = 0; i < 128; i++) {
                sum += A[row * 128 + i] * B[i * 128 + col];
            }
            C[row * 128 + col] = sum;
        }
    }
    """
    converted_code = ast_loop_recovery(cuda_code, "cuda")
    print(converted_code)
