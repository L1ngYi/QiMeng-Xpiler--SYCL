import json

from pycparser import c_ast, c_parser

from falcon.buffer_inline import ast_buffer_inline
from falcon.simplification import simplify_code
from falcon.stmt_simplification import ast_stmt_simplification
from falcon.util import (
    NodeTransformer,
    generate_code,
    make_full_func,
    parse_code_ast,
)

cuda_file_name = "falcon/documents/cuda_op_tensorization.json"
hip_file_name = "falcon/documents/hip_op_tensorization.json"
cpu_file_name = "falcon/documents/cpu_op_tensorization.json"


class Detensorizer(NodeTransformer):
    def __init__(self, func_defs):
        self.func_defs = func_defs
        self.parser = c_parser.CParser()
        self.parameter_mappings = {}

    def visit_FuncCall(self, node):
        if node.name.name in self.func_defs:
            func_def = self.func_defs[node.name.name]
            seq_def = self.parser.parse(func_def)
            if not isinstance(seq_def, c_ast.FileAST):
                raise ValueError("Sequential code must be a function")

            # Construct a map between the function call's  arguments and
            # callee's arguments
            seq_def_args = seq_def.ext[0].decl.type.args.params
            seq_def_name = [arg_id.name for arg_id in seq_def_args]
            self.parameter_mappings = {
                arg: param for arg, param in zip(seq_def_name, node.args.exprs)
            }
            body = seq_def.ext[0].body
            return self.visit(body.block_items[0])
        else:
            return self.generic_visit(node)

    def visit_ID(self, node):
        if node.name in self.parameter_mappings:
            return self.parameter_mappings[node.name]
        return node


def ast_detensorization(code, target):
    """Transform C code using an SMT solver to optimize loop constructs.

    This function parses the provided C code into an Abstract Syntax Tree (AST) and applies
    a transformation to split loops based on the given loop index and factor. The transformation
    is guided by an SMT solver to ensure the generated code is logically equivalent to the
    original but potentially more optimized.

    Parameters:
    - code (str): A string containing the C code to be transformed.
    - file_name (str): The definition of intrinsics.

    Returns:
    - str: The transformed C code as a string.

    Todo:
    - Implement additional error checking for the input parameters.
    - Extend the visitor to handle more complex loop structures.
    """
    ast = parse_code_ast(code)

    if target == "cuda":
        # If the code doesn't contain any CUDA tensor-core / WMMA markers,
        # skip detensorization to avoid unnecessary work.
        if not any(
            tok in code
            for tok in ("wmma", "__wmma", "wmma::", "wmma_fragment", "wmma_")
        ):
            return code
        else:
            with open(cuda_file_name) as json_file:
                func_defs = json.load(json_file)
            visitor = Detensorizer(func_defs)
            visitor.visit(ast)
            code = generate_code(ast)

    elif target == "hip":
        with open(hip_file_name) as json_file:
            func_defs = json.load(json_file)
        visitor = Detensorizer(func_defs)
        visitor.visit(ast)
        code = generate_code(ast)

    elif target == "cpu":
        with open(cpu_file_name) as json_file:
            func_defs = json.load(json_file)
        visitor = Detensorizer(func_defs)
        visitor.visit(ast)
        code = generate_code(ast)

    else:
        raise RuntimeError("unsuppored target.")
    code = simplify_code(code)
    code = ast_stmt_simplification(code)
    code = ast_buffer_inline(code)
    code = make_full_func(code, target)
    return code


if __name__ == "__main__":

    cuda_code = """
    __global__ void WMMAF16TensorCore(half *A, half *B, float *C, float *D)
    {
        int ix = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
        int iy = (blockIdx.y * blockDim.y + threadIdx.y);

        wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, M, N, K, float> ab_frag;
        wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;

        wmma::fill_fragment(ab_frag, 0.0f);
        int a_col, a_row, b_col, b_row, c_col, c_row;
        a_row = ix * M;
        b_row = iy * N;
        for (int k=0; k<32; k+=K) {
            a_col = b_col = k;

            if (a_row < 32 && a_col < 32 && b_row < 32 && b_col < 32) {
                // Load the inputs
                wmma::load_matrix_sync(a_frag, A + a_col + a_row * 32, 32);
                wmma::load_matrix_sync(b_frag, B + b_col + b_col * 32, 32);

                // Perform the matrix multiplication
                wmma::mma_sync(ab_frag, a_frag, b_frag, ab_frag);
            }
        }

        c_col = b_row;
        c_row = a_row;
        if (c_row < 32 && c_col < 32) {
    wmma::load_matrix_sync(c_frag, C + c_col + c_row * 32, 32,
    wmma::mem_row_major);

            for (int i = 0; i < c_frag.num_elements; i++) {
                c_frag.x[i] = ab_frag.x[i] + c_frag.x[i];
            }

            // Store the output
            wmma::store_matrix_sync(D + c_col + c_row * 32, c_frag, 32, wmma::mem_row_major);
        }
    }
    """
    converted_code = ast_detensorization(cuda_code, "cuda")
    print(converted_code)

    cuda_code = """
    typedef float half;
    typedef struct {
        float x[8];
    } wmma_fragment;
    void wmma_load_matrix_sync(wmma_fragment f, float *ptr, int stride);
    void kernel(half *A, half *B, float *C, float *D)
    {
    wmma_load_matrix_sync(a_frag, A + ix, 32);
    }
    """

    converted_code = ast_detensorization(cuda_code, "cuda")
    print(converted_code)
