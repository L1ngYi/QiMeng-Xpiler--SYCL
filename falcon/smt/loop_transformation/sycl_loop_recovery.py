import sys
import clang.cindex
from clang.cindex import CursorKind, TokenKind

# 配置 libclang 路径（根据你的环境修改）
# clang.cindex.Config.set_library_path("/usr/lib/llvm-14/lib")

class SyclKernelExtractor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.index = clang.cindex.Index.create()
        # 解析时必须开启 C++17 或更高，并保留宏
        self.tu = self.index.parse(
            file_path,
            args=['-std=c++17', '-x', 'c++', '-fsyntax-only']
        )
        self.source_code = self._read_source()

    def _read_source(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _get_text(self, cursor):
        """准确提取 AST 节点对应的源码文本"""
        start = cursor.extent.start.offset
        end = cursor.extent.end.offset
        return self.source_code[start:end]

    def find_kernels(self):
        """入口：查找所有 parallel_for 并提取信息"""
        kernels = []
        self._recursive_find_parallel_for(self.tu.cursor, kernels)
        return kernels

    def _recursive_find_parallel_for(self, cursor, results):
        # 匹配函数调用且名字是 parallel_for
        if cursor.kind == CursorKind.CALL_EXPR and cursor.spelling == 'parallel_for':
            try:
                kernel_info = self._analyze_kernel_invocation(cursor)
                if kernel_info:
                    results.append(kernel_info)
            except Exception as e:
                print(f"[Warn] 解析 Kernel 失败: {e}")
        
        for child in cursor.get_children():
            self._recursive_find_parallel_for(child, results)

    def _analyze_kernel_invocation(self, call_node):
        """分析 parallel_for 的参数，不依赖固定位置"""
        args = list(call_node.get_arguments())
        
        range_node = None
        lambda_node = None
        
        # 1. 智能参数识别
        for arg in args:
            arg_type = arg.type.spelling
            # 去除 const/ref 修饰以便匹配
            clean_type = arg_type.replace("const", "").replace("&", "").strip()
            
            # 识别 Range / ND_Range
            if "range" in clean_type and range_node is None:
                range_node = arg
            
            # 识别 Lambda (AST 中通常显示为 lambda at line:col 或 class (lambda))
            # 或者 UnexposedExpr 包裹的 Lambda
            if lambda_node is None:
                if arg.kind == CursorKind.LAMBDA_EXPR:
                    lambda_node = arg
                elif "lambda" in clean_type or "(lambda at" in clean_type:
                    # 有时候 Lambda 被包裹在构造函数或转换中，需要钻取
                    lambda_node = self._drill_down_to_lambda(arg)

        if not range_node or not lambda_node:
            print(f"[Debug] 无法识别参数: Range={range_node}, Lambda={lambda_node}")
            return None

        # 2. 提取维度和边界
        dims, bounds = self._parse_range(range_node)
        
        # 3. 分析 Lambda (参数、捕获、体)
        kernel_data = self._analyze_lambda(lambda_node, dims)
        
        return {
            "dims": dims,
            "bounds": bounds,
            "captures": kernel_data['captures'],
            "body": kernel_data['body'],
            "index_var": kernel_data['index_var']
        }

    def _drill_down_to_lambda(self, node):
        """递归向下寻找真正的 LambdaExpr 节点"""
        if node.kind == CursorKind.LAMBDA_EXPR:
            return node
        for child in node.get_children():
            res = self._drill_down_to_lambda(child)
            if res: return res
        return node

    def _parse_range(self, range_node):
        """解析 range<N>(d1, d2...)"""
        # 获取 range 的模板参数 N (维度)
        # 这里用文本分析作为后备，因为 libclang 对模板参数支持一般
        text = self._get_text(range_node)
        
        # 尝试推断维度
        dims = 1
        if "range<2>" in text or "nd_range<2>" in text: dims = 2
        elif "range<3>" in text or "nd_range<3>" in text: dims = 3
        
        # 提取构造函数的参数 (循环边界)
        # 遍历 range_node 的子节点，找到 INTEGER_LITERAL 或 DECL_REF_EXPR
        bounds = []
        for child in range_node.get_children():
            # 过滤掉类型引用等杂项，只看表达式
            if child.kind in [CursorKind.INTEGER_LITERAL, CursorKind.DECL_REF_EXPR, CursorKind.UNEXPOSED_EXPR]:
                bounds.append(self._get_text(child))
        
        # 如果参数不够，补全（针对 nd_range，可能需要取前一半作为 global_range）
        if "nd_range" in text and len(bounds) >= dims * 2:
            bounds = bounds[:dims] # 取 Global Range
            
        return dims, bounds

    def _analyze_lambda(self, lambda_node, dims):
        """深入分析 Lambda 内部"""
        # 1. 获取 Kernel 索引变量名 (item / idx)
        # Lambda 的子节点中，第一个 PARM_DECL 通常是索引
        index_var = "item"
        children = list(lambda_node.get_children())
        body_node = None
        
        for child in children:
            if child.kind == CursorKind.PARM_DECL:
                index_var = child.spelling
            elif child.kind == CursorKind.COMPOUND_STMT:
                body_node = child

        if not body_node:
            raise ValueError("Lambda body not found")

        # 2. 真正的捕获分析 (Def-Use Chain)
        captures = []
        # 获取 Lambda 的起始位置
        lambda_start_offset = lambda_node.extent.start.offset
        
        def visit_for_captures(node):
            if node.kind == CursorKind.DECL_REF_EXPR:
                ref_node = node.referenced
                if ref_node:
                    # 如果变量定义在 Lambda 之前，则是捕获变量
                    # 且排除掉全局变量（通常不需要作为参数传递，或者需要特殊处理）
                    def_offset = ref_node.extent.start.offset
                    if def_offset < lambda_start_offset:
                        # 排除函数名引用、全局常量等
                        if ref_node.kind in [CursorKind.VAR_DECL, CursorKind.PARM_DECL]:
                            # 获取类型
                            type_name = ref_node.type.spelling
                            captures.append((node.spelling, type_name))
            
            for child in node.get_children():
                visit_for_captures(child)

        visit_for_captures(body_node)
        # 去重
        unique_captures = list(set(captures))

        # 3. Body 转换 (Token 级替换)
        # 我们需要将 body 中的 item.get_global_id(0) 替换为循环变量
        trans_body = self._rewrite_body_tokens(body_node, index_var, dims)

        return {
            "index_var": index_var,
            "captures": unique_captures,
            "body": trans_body
        }

    def _rewrite_body_tokens(self, body_node, index_var, dims):
        """基于 Token 的安全代码重写"""
        tokens = list(self.tu.get_tokens(extent=body_node.extent))
        output_parts = []
        i = 0
        
        # 定义目标循环变量名
        loop_vars = ["i", "j", "k"] # 对应 dim 0, 1, 2

        while i < len(tokens):
            t = tokens[i]
            replaced = False
            
            # 检测模式: index_var . get_global_id ( N )
            # 例如: item.get_global_id(0)
            if t.spelling == index_var:
                # 向前看 5 个 token
                if i + 5 < len(tokens):
                    t1 = tokens[i+1] # .
                    t2 = tokens[i+2] # get_global_id / get_id
                    t3 = tokens[i+3] # (
                    t4 = tokens[i+4] # 0/1/2
                    t5 = tokens[i+5] # )
                    
                    if t1.spelling == '.' and 'get' in t2.spelling and t3.spelling == '(':
                        # 确定维度
                        dim_idx = 0
                        if t4.kind == TokenKind.LITERAL:
                            dim_idx = int(t4.spelling)
                        
                        # 执行替换：换成对应的循环变量 i, j, k
                        if dim_idx < len(loop_vars):
                            output_parts.append(loop_vars[dim_idx])
                            i += 6 # 跳过这 6 个 token
                            replaced = True
            
            # 检测模式: 直接使用 idx[0] (如果是 id<N> 类型)
            if not replaced and t.spelling == index_var:
                 if i + 3 < len(tokens) and tokens[i+1].spelling == '[':
                      # item[0] -> i
                      dim_token = tokens[i+2]
                      if dim_token.kind == TokenKind.LITERAL:
                           dim = int(dim_token.spelling)
                           output_parts.append(loop_vars[dim])
                           i += 4
                           replaced = True
            
            if not replaced:
                # 简单处理：如果是 } 或 { 后面加换行，其他加空格
                # 实际工程可以用 SourceLocation 精确还原空格
                output_parts.append(t.spelling)
                output_parts.append(" ")
                i += 1
        
        # 简单的后处理，去掉首尾花括号
        return "".join(output_parts).strip().lstrip("{").rstrip("}")

    def generate_cpp_code(self, kernel_info):
        """生成最终 C++ 代码"""
        dims = kernel_info['dims']
        bounds = kernel_info['bounds']
        captures = kernel_info['captures']
        
        # 1. 生成参数列表
        args_str = ", ".join([f"{typ} {name}" for name, typ in captures])
        
        # 2. 生成循环头
        loops = ""
        indent = "    "
        loop_vars = ["i", "j", "k"]
        
        for d in range(dims):
            # 注意：SYCL range<2>(R, C) 通常 dim 0 是 Row，dim 1 是 Col
            # 这里的 bounds 需要对应。假设 bounds 顺序正确。
            limit = bounds[d] if d < len(bounds) else "N"
            var = loop_vars[d]
            loops += f"{indent * (d+1)}for (int {var} = 0; {var} < {limit}; ++{var}) {{\n"
        
        # 3. 填充 Body
        body_lines = kernel_info['body'].splitlines()
        formatted_body = "\n".join([f"{indent * (dims+1)}{line.strip()}" for line in body_lines if line.strip()])
        
        # 4. 闭合循环
        closing = ""
        for d in reversed(range(dims)):
             closing += f"{indent * (d+1)}}}\n"

        code = f"""
void kernel_recovered({args_str}) {{
{loops}
{formatted_body}
{closing}
}}
"""
        return code




import tempfile
import os

def ast_sycl_loop_recovery(code):
    """
    [新增] SYCL AST 恢复的适配器入口。
    适配 actions.py 的接口：输入字符串 -> 输出字符串。
    """
    # 1. 创建临时文件保存 code 字符串
    # 后缀必须是 .cpp，否则 Clang 可能会拒绝解析或无法识别 C++ 语法
    fd, path = tempfile.mkstemp(suffix=".cpp")
    try:
        # 将内存中的代码写入临时文件
        with os.fdopen(fd, 'w', encoding='utf-8') as tmp:
            tmp.write(code)
        
        # 2. 调用核心提取器
        extractor = SyclKernelExtractor(path)
        kernels = extractor.find_kernels()
        
        if not kernels:
            # 如果没找到 Kernel，抛出异常以触发外层的错误处理（或返回原代码）
            raise RuntimeError("SYCL AST Parsing failed: No kernels found in source.")
            
        # 3. 生成代码
        # 通常 Benchmark 文件里只有一个核心 Kernel，直接提取。
        # 如果有多个，拼接返回。
        recovered_code_parts = []
        for k in kernels:
            recovered_code_parts.append(extractor.generate_cpp_code(k))
            
        return "\n".join(recovered_code_parts)

    except Exception as e:
        print(f"[Error] SYCL AST Logic Failed: {e}")
        # 重新抛出异常，让上层决定是中断还是怎么处理
        raise e
    finally:
        # 4. 清理临时文件
        if os.path.exists(path):
            os.remove(path)

# --- 使用示例 ---
if __name__ == "__main__":
    # 创建一个模拟文件
    with open("test_kernel.cpp", "w") as f:
        f.write("""
        #include <sycl/sycl.hpp>
        using namespace sycl;
        
        void complicated_kernel(queue &q, float *A, float *B, int H, int W) {
            q.submit([&](handler &h) {
                // 定义 accessor，测试类型提取
                accessor acc_A(A, h, read_only);
                
                h.parallel_for(range<3>(H, W, 16), [=](item<3> item) {
                    int r = item.get_global_id(0);
                    int c = item.get_global_id(1);
                    // 外部变量捕获测试
                    if (r < H && c < W) {
                        B[r * W + c] = acc_A[r * W + c] * 2.0f;
                    }
                });
            });
        }
        """)

    extractor = SyclKernelExtractor("test_kernel.cpp")
    kernels = extractor.find_kernels()
    
    for k in kernels:
        print("=== Recovered Kernel ===")
        print(extractor.generate_cpp_code(k))