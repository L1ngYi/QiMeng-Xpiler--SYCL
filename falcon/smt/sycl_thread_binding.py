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


def _generate_sycl_kernel_block(for_node, dim_idx=0):
    """Convert a single outermost for-loop node to a SYCL q.submit block."""
    gen = c_generator.CGenerator()
    loop_var, loop_bound = _get_loop_var_and_bound(for_node)

    if loop_var is None or loop_bound is None:
        # Cannot determine loop info — emit original loop unchanged
        logging.warning(
            "sycl_thread_binding: could not extract loop variable or bound; "
            "emitting original for-loop unchanged."
        )
        return gen.visit(for_node) + ";"

    # Generate the raw body code (without replacing the loop variable)
    body_code = gen.visit(for_node.stmt)

    # Strip surrounding braces that CGenerator adds for Compound statements
    stripped = body_code.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        inner_body = stripped[1:-1].strip()
    else:
        inner_body = stripped

    # Indent inner body lines for readability
    indented_body = "\n".join(
        "      " + line if line.strip() else line
        for line in inner_body.splitlines()
    )

    # Declare the loop variable via item.get_global_id so the original body
    # can reference it by its original name unchanged.
    return (
        f"q.submit([&](handler &h) {{\n"
        f"    h.parallel_for(range<1>({loop_bound}), [=](item<1> item) {{\n"
        f"      int {loop_var} = item.get_global_id({dim_idx});\n"
        f"{indented_body}\n"
        f"    }});\n"
        f"  }});\n"
        f"  q.wait();"
    )


def _build_sycl_function(func_node, for_loops):
    """Reconstruct the full SYCL function string from the parsed AST."""
    gen = c_generator.CGenerator()

    # ---- Function return type ----
    ret_type = gen.visit(func_node.decl.type.type)

    # ---- Function name ----
    func_name = func_node.decl.name

    # ---- Original parameters (strip extern "C" and __global__ etc.) ----
    params = []
    if func_node.decl.type.args:
        for param in (func_node.decl.type.args.params or []):
            params.append(gen.visit(param))
    # Append the SYCL queue parameter
    params.append("sycl::queue &q")
    param_str = ", ".join(params)

    # ---- Function body ----
    # We iterate over the body statements; for-loops are converted, others kept.
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
    """Convert serial C++ for-loops to SYCL parallel_for pattern.

    Parses *code* with pycparser, finds the outermost for-loops inside each
    function, and replaces them with ``q.submit`` / ``h.parallel_for`` /
    ``item.get_global_id`` patterns.  The function signature is extended with
    a ``sycl::queue &q`` parameter.  A SYCL header and ``using namespace sycl``
    are prepended to the output.
    """
    ast = parse_code_ast(code)
    if ast is None:
        raise RuntimeError("Failed to parse code AST for SYCL thread binding.")

    func_defs = [ext for ext in ast.ext if isinstance(ext, c_ast.FuncDef)]
    if not func_defs:
        raise RuntimeError("No function definitions found in code.")

    # Convert each function definition to SYCL
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
    code = """
    extern "C" void add(float *input1, float *input2, float *output) {
      int size = 64;
      for (int i = 0; i < size; i++) {
        output[i] = input1[i] + input2[i];
      }
    }
    """
    print(ast_sycl_thread_binding(code))
