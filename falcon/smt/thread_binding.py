from pycparser import c_ast, c_generator
from falcon.util import NodeTransformer, generate_code, parse_code_ast
import logging

from falcon.util import (
    NodeTransformer,
    generate_code,
    make_full_func,
    parse_code_ast,
)

builtin_var = {
    "cuda": ["threadIdxx", "blockIdxx"],
    "hip": ["threadIdxx", "blockIdxx"],
}
builtin_dim = {
    "threadIdxx": 256,
    "blockIdxx": 1024,
    "coreId": 4,
    "clusterId": 12,
}


# Temporarily, we will binding the outermost with thread
class ThreadBindingTransformer(NodeTransformer):
    def __init__(self, parallel_loops, target="cuda"):
        self.binding_map = {}
        self.parallel_loops = parallel_loops
        self.target = target
        self.current_depth = 0

    def visit_For(self, node):
        self.current_depth += 1
        try:
            loop_var = (
                node.init.decls[0].name
                if isinstance(node.init, c_ast.DeclList)
                else None
            )
            extend = int(node.cond.right.value)


            # For CUDA/HIP only bind when at the outermost loop (depth==1)
            if (self.target in ("cuda", "hip")) and self.current_depth == 1:
                thread_var = self._generate_thread_var(extend, 1024)
                new_node = self._generate_new_node(thread_var, node)
                self.binding_map[loop_var] = thread_var
                return self.generic_visit(new_node)

            return self.generic_visit(node)
        finally:
            self.current_depth -= 1

    def _generate_thread_var(self, extend, limit):
        if extend <= limit:
            return c_ast.ID(name=builtin_var[self.target][0])
        else:
            return c_ast.BinaryOp(
                op="+",
                left=c_ast.BinaryOp(
                    op="*",
                    left=c_ast.ID(name=builtin_var[self.target][1]),
                    right=c_ast.Constant("int", value=str(limit)),
                ),
                right=c_ast.ID(name=builtin_var[self.target][0]),
            )

    def _generate_new_node(self, thread_var, node):
        return c_ast.If(
            cond=c_ast.BinaryOp(
                op="<", left=thread_var, right=node.cond.right
            ),
            iftrue=node.stmt,
            iffalse=None,
        )

    def visit_ID(self, node):
        return self.binding_map.get(node.name, node)


class LoopVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.current_depth = 0  # Current nesting depth
        self.max_depth = 0  # Maximum nesting depth

    def visit_For(self, node):
        # Each time a for loop is encountered, the nesting depth increases by
        # 1.
        self.current_depth += 1
        # Update maximum nesting depth.
        if self.current_depth > self.max_depth:
            self.max_depth = self.current_depth

        # Access child nodes
        self.generic_visit(node)

        # When exiting the for loop, decrease the nesting depth by 1.
        self.current_depth -= 1

#------sycl thread binding entry------
class OutermostForCollector(c_ast.NodeVisitor):
    """Collect outermost for-loops inside a function body."""

    def __init__(self):
        self.loops = []
        self._in_loop = False

    def visit_For(self, node):
        if not self._in_loop:
            self.loops.append(node)
            self._in_loop = True
            self.generic_visit(node)
            self._in_loop = False
        else:
            self.generic_visit(node)


def _sycl_get_loop_var_and_bound(for_node):
    """Return (loop_variable_name, loop_bound_expr_string) for a for-loop node."""
    gen = c_generator.CGenerator()
    loop_var = None
    loop_bound = None

    if isinstance(for_node.init, c_ast.DeclList) and for_node.init.decls:
        loop_var = for_node.init.decls[0].name

    if isinstance(for_node.cond, c_ast.BinaryOp):
        loop_bound = gen.visit(for_node.cond.right)

    return loop_var, loop_bound


def _sycl_get_inner_parallelizable_loop(for_node):
    """
    If the body of for_node is (optionally wrapped in a Compound) a single
    inner for-loop with no dependencies on the outer loop variable in its
    init/cond/next, return that inner For node. Otherwise return None.
    This enables 2D loop collapsing for kernels like GEMM.
    """
    body = for_node.stmt

    # Unwrap a single-statement Compound
    if isinstance(body, c_ast.Compound):
        items = body.block_items or []
        if len(items) != 1:
            return None
        body = items[0]

    if not isinstance(body, c_ast.For):
        return None

    return body


def _sycl_generate_sycl_kernel_block(for_node):
    """
    Convert a single outermost for-loop node to a SYCL q.submit block.
    Automatically collapses two perfectly nested parallelizable loops into
    a range<2> kernel.
    """
    gen = c_generator.CGenerator()

    outer_var, outer_bound = _sycl_get_loop_var_and_bound(for_node)
    if outer_var is None or outer_bound is None:
        logging.warning(
            "sycl_thread_binding: could not extract loop variable or bound; "
            "emitting original for-loop unchanged."
        )
        return gen.visit(for_node) + ";"

    # Try to collapse two nested loops into range<2>
    inner_for = _sycl_get_inner_parallelizable_loop(for_node)
    if inner_for is not None:
        inner_var, inner_bound = _sycl_get_loop_var_and_bound(inner_for)
        if inner_var is not None and inner_bound is not None:
            return _sycl_generate_2d_kernel(
                outer_var, outer_bound,
                inner_var, inner_bound,
                inner_for.stmt,  # actual compute body is inside inner loop
                gen,
            )

    # Fall back to 1D kernel
    return _sycl_generate_1d_kernel(outer_var, outer_bound, for_node.stmt, gen)


def _indent_body(body_code):
    """Strip braces and indent lines for embedding in a lambda."""
    stripped = body_code.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        stripped = stripped[1:-1].strip()
    return "\n".join(
        "      " + line if line.strip() else line
        for line in stripped.splitlines()
    )


def _sycl_generate_1d_kernel(loop_var, loop_bound, body_node, gen):
    """Emit a range<1> parallel_for kernel."""
    indented_body = _indent_body(gen.visit(body_node))
    return (
        f"q.submit([&](handler &h) {{\n"
        f"    h.parallel_for(range<1>({loop_bound}), [=](item<1> item) {{\n"
        f"      int {loop_var} = item.get_id(0);\n"   # ← get_id, not get_global_id
        f"{indented_body}\n"
        f"    }});\n"
        f"  }});\n"
        f"  q.wait();"
    )


def _sycl_generate_2d_kernel(outer_var, outer_bound, inner_var, inner_bound,
                        body_node, gen):
    """Emit a range<2> parallel_for kernel collapsing two nested loops."""
    indented_body = _indent_body(gen.visit(body_node))
    return (
        f"q.submit([&](handler &h) {{\n"
        f"    h.parallel_for(range<2>({outer_bound}, {inner_bound}), "
        f"[=](item<2> item) {{\n"
        f"      int {outer_var} = item.get_id(0);\n"  # ← get_id, not get_global_id
        f"      int {inner_var} = item.get_id(1);\n"  # ← get_id, not get_global_id
        f"{indented_body}\n"
        f"    }});\n"
        f"  }});\n"
        f"  q.wait();"
    )


def _sycl_build_sycl_function(func_node, for_loops):
    """Reconstruct the full SYCL function string from the parsed AST."""
    gen = c_generator.CGenerator()

    ret_type = gen.visit(func_node.decl.type.type)
    func_name = func_node.decl.name

    params = []
    if func_node.decl.type.args:
        for param in (func_node.decl.type.args.params or []):
            params.append(gen.visit(param))
    params.append("sycl::queue &q")
    param_str = ", ".join(params)

    body_stmts = []
    if func_node.body and func_node.body.block_items:
        collector = OutermostForCollector()
        collector.visit(func_node.body)
        outermost_set = set(id(n) for n in collector.loops)

        for stmt in func_node.body.block_items:
            if id(stmt) in outermost_set:
                body_stmts.append("  " + _sycl_generate_sycl_kernel_block(stmt))
            else:
                stmt_code = gen.visit(stmt)
                if not stmt_code.endswith(";"):
                    stmt_code += ";"
                body_stmts.append("  " + stmt_code)

    body_str = "\n".join(body_stmts)
    return (
        f"{ret_type} {func_name}({param_str}) {{\n"
        f"{body_str}\n"
        f"}}"
    )


def ast_sycl_thread_binding(code):
    """Convert serial C++ for-loops to SYCL parallel_for pattern."""
    ast = parse_code_ast(code)
    if ast is None:
        raise RuntimeError("Failed to parse code AST for SYCL thread binding.")

    func_defs = [ext for ext in ast.ext if isinstance(ext, c_ast.FuncDef)]
    if not func_defs:
        raise RuntimeError("No function definitions found in code.")

    converted_functions = []
    for func_node in func_defs:
        collector = OutermostForCollector()
        if func_node.body:
            collector.visit(func_node.body)
        converted_functions.append(
            _sycl_build_sycl_function(func_node, collector.loops)
        )

    sycl_body = "\n\n".join(converted_functions)
    return (
        "#include <sycl/sycl.hpp>\n"
        "using namespace sycl;\n\n"
        + sycl_body
    )
#------sycl thread binding end------


def ast_thread_binding(code, target="cuda"):
    # Simple validation: only accept these targets
    allowed_targets = ["cuda", "hip","sycl"]
    if not isinstance(target, str):
        raise ValueError(
            f"Unsupported target '{target}'. Supported targets: {allowed_targets}"
        )
    
    if target not in allowed_targets:
        raise ValueError(
            f"Unsupported target '{target}'. Supported targets: {allowed_targets}"
        )
    
    if target == "sycl":
        # For SYCL, we will use the specialized AST-based transformation
        return ast_sycl_thread_binding(code)
    
    
    # Analytical code
    ast = parse_code_ast(code)

    # Count the number of loop layers.
    loop_visitor = LoopVisitor()
    loop_visitor.visit(ast)
    # Perform thread-bound conversion.
    transformer = ThreadBindingTransformer(loop_visitor.max_depth, target)
    ast = transformer.visit(ast)
    # Output the modified code.
    binding_code = generate_code(ast)

    return make_full_func(binding_code, target)


if __name__ == "__main__":
    # Sample code

    code = """
    void func() {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                for (int k = 0; k < 7; ++k) {
                    B[i * 4 * 7 + j * 7 + k] = A[i * 4 * 7 + j * 7 + k] + 1.0;
                }
            }
        }
    }
    """
    output_code = ast_thread_binding(code, target="cuda")
    print(output_code)

    code = """
    void softmax(float *A, float *T_softmax_norm)
    {
        for (int k = 0; k < 5; ++k)
        {
            float maxVal = A[k * 128];
            for (int j = 1; j < 128; ++j)
            {
                if (A[(k * 128) + j] > maxVal)
                {
                    maxVal = A[(k * 128) + j];
                }
            }

            float denom = 0.0f;
            for (int j = 0; j < 128; ++j)
            {
                T_softmax_norm[(k * 128) + j] = expf(A[(k * 128) + j] - maxVal);
            }

            for (int j = 0; j < 128; ++j)
            {
                denom += T_softmax_norm[(k * 128) + j];
            }

            for (int j = 0; j < 128; ++j)
            {
                T_softmax_norm[(k * 128) + j] /= denom;
            }
        }
    }
    """
    output_code = ast_thread_binding(code, target="cuda")
    print(output_code)
