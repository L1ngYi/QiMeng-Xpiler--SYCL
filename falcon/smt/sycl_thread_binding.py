from pycparser import c_ast, c_generator
from falcon.util import NodeTransformer, generate_code, parse_code_ast
import logging


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


def _get_loop_var_and_bound(for_node):
    """Return (loop_variable_name, loop_bound_expr_string) for a for-loop node."""
    gen = c_generator.CGenerator()
    loop_var = None
    loop_bound = None

    if isinstance(for_node.init, c_ast.DeclList) and for_node.init.decls:
        loop_var = for_node.init.decls[0].name

    if isinstance(for_node.cond, c_ast.BinaryOp):
        loop_bound = gen.visit(for_node.cond.right)

    return loop_var, loop_bound


def _get_inner_parallelizable_loop(for_node):
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


def _generate_sycl_kernel_block(for_node):
    """
    Convert a single outermost for-loop node to a SYCL q.submit block.
    Automatically collapses two perfectly nested parallelizable loops into
    a range<2> kernel.
    """
    gen = c_generator.CGenerator()

    outer_var, outer_bound = _get_loop_var_and_bound(for_node)
    if outer_var is None or outer_bound is None:
        logging.warning(
            "sycl_thread_binding: could not extract loop variable or bound; "
            "emitting original for-loop unchanged."
        )
        return gen.visit(for_node) + ";"

    # Try to collapse two nested loops into range<2>
    inner_for = _get_inner_parallelizable_loop(for_node)
    if inner_for is not None:
        inner_var, inner_bound = _get_loop_var_and_bound(inner_for)
        if inner_var is not None and inner_bound is not None:
            return _generate_2d_kernel(
                outer_var, outer_bound,
                inner_var, inner_bound,
                inner_for.stmt,  # actual compute body is inside inner loop
                gen,
            )

    # Fall back to 1D kernel
    return _generate_1d_kernel(outer_var, outer_bound, for_node.stmt, gen)


def _indent_body(body_code):
    """Strip braces and indent lines for embedding in a lambda."""
    stripped = body_code.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        stripped = stripped[1:-1].strip()
    return "\n".join(
        "      " + line if line.strip() else line
        for line in stripped.splitlines()
    )


def _generate_1d_kernel(loop_var, loop_bound, body_node, gen):
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


def _generate_2d_kernel(outer_var, outer_bound, inner_var, inner_bound,
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


def _build_sycl_function(func_node, for_loops):
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
                body_stmts.append("  " + _generate_sycl_kernel_block(stmt))
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
            _build_sycl_function(func_node, collector.loops)
        )

    sycl_body = "\n\n".join(converted_functions)
    return (
        "#include <sycl/sycl.hpp>\n"
        "using namespace sycl;\n\n"
        + sycl_body
    )


if __name__ == "__main__":
    # Test 1D
    code_1d = """
    extern "C" void add(float *input1, float *input2, float *output) {
      int size = 64;
      for (int i = 0; i < size; i++) {
        output[i] = input1[i] + input2[i];
      }
    }
    """
    print("=== 1D ===")
    print(ast_sycl_thread_binding(code_1d))

    # Test 2D (your GEMM)
    code_2d = """
    extern "C" void gemm(float *A, float *B, float *result) {
      for (int j = 0; j < 32; j++) {
        for (int k = 0; k < 128; k++) {
          result[j * 128 + k] = 0;
          for (int l = 0; l < 32; l++) {
            result[j * 128 + k] += A[j * 32 + l] * B[l * 128 + k];
          }
        }
      }
    }
    """
    print("\n=== 2D GEMM ===")
    print(ast_sycl_thread_binding(code_2d))