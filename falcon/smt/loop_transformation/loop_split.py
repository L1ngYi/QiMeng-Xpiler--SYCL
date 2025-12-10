import re

from pycparser import c_ast, c_generator

from falcon.util import parse_code_ast


class SplitForLoopVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.factor = None
        self.axis_name = None
        self.org_extent = None

    def visit_Compound(self, node):
        """Scans block items for #pragma loop_split and applies transformation
        to the subsequent for-loop."""
        new_block_items = []
        skip_next = False

        if node.block_items:
            for index, subnode in enumerate(node.block_items):
                if skip_next:
                    skip_next = False
                    continue

                # Check for #pragma loop_split
                if (
                    isinstance(subnode, c_ast.Pragma)
                    and "loop_split" in subnode.string
                ):
                    # Extract factor using regex (handles "loop_split(2)" or
                    # "loop_split(factor=2)")
                    match = re.search(
                        r"\((\d+)\)", subnode.string
                    ) or re.search(r"factor=(\d+)", subnode.string)

                    if match:
                        self.factor = int(match.group(1))

                        # Check if next node is a For loop
                        if index + 1 < len(node.block_items):
                            next_node = node.block_items[index + 1]
                            if isinstance(next_node, c_ast.For):
                                # Extract loop variable name (assuming: int i =
                                # 0;)
                                self.axis_name = next_node.init.decls[0].name

                                # Perform the split
                                split_loop = self.split_for_loop(next_node)
                                new_block_items.append(split_loop)

                                # Skip the original loop in the next iteration
                                skip_next = True
                                continue

                # If no split happened, keep the node
                new_block_items.append(subnode)

            node.block_items = new_block_items

        # Continue visiting children
        self.generic_visit(node)

    def split_for_loop(self, node):
        """Transforms a single loop into a tiled nested loop structure."""
        # 1. Calculate Ranges
        # Note: robust code should handle cases where extent isn't perfectly
        # divisible
        self.org_extent = int(node.cond.right.value)
        inner_range = self.org_extent // self.factor

        # Define variable names
        i_in = f"{self.axis_name}_in"
        i_out = f"{self.axis_name}_out"

        # 2. Create Inner Loop
        # Init: int i_in = 0;
        init_in = c_ast.Decl(
            name=i_in,
            quals=[],
            align=[],
            storage=[],
            funcspec=[],
            type=c_ast.TypeDecl(
                declname=i_in,
                quals=[],
                align=None,
                type=c_ast.IdentifierType(["int"]),
            ),
            init=c_ast.Constant("int", "0"),
            bitsize=None,
        )
        # Cond: i_in < inner_range;
        cond_in = c_ast.BinaryOp(
            "<", c_ast.ID(i_in), c_ast.Constant("int", str(inner_range))
        )
        # Next: i_in++
        next_in = c_ast.UnaryOp("p++", c_ast.ID(i_in))

        # 3. Create the Index Reconstruction (The Fix)
        # We inject: int i = (i_out * inner_range) + i_in;
        # This replaces the need for the visit_ID string hack.
        calc_expr = c_ast.BinaryOp(
            "+",
            c_ast.BinaryOp(
                "*", c_ast.ID(i_out), c_ast.Constant("int", str(inner_range))
            ),
            c_ast.ID(i_in),
        )

        decl_original_var = c_ast.Decl(
            name=self.axis_name,
            quals=[],
            align=[],
            storage=[],
            funcspec=[],
            type=c_ast.TypeDecl(
                declname=self.axis_name,
                quals=[],
                align=None,
                type=c_ast.IdentifierType(["int"]),
            ),
            init=calc_expr,
            bitsize=None,
        )

        # Construct Inner Loop Body: [Declaration of i] + [Original Body]
        inner_body_items = [decl_original_var]

        if isinstance(node.stmt, c_ast.Compound):
            inner_body_items.extend(node.stmt.block_items)
        else:
            inner_body_items.append(node.stmt)

        inner_compound = c_ast.Compound(block_items=inner_body_items)
        inner_for = c_ast.For(init_in, cond_in, next_in, inner_compound)

        # 4. Create Outer Loop
        # Init: int i_out = 0;
        init_out = c_ast.Decl(
            name=i_out,
            quals=[],
            align=[],
            storage=[],
            funcspec=[],
            type=c_ast.TypeDecl(
                declname=i_out,
                quals=[],
                align=None,
                type=c_ast.IdentifierType(["int"]),
            ),
            init=c_ast.Constant("int", "0"),
            bitsize=None,
        )
        # Cond: i_out < factor;
        cond_out = c_ast.BinaryOp(
            "<", c_ast.ID(i_out), c_ast.Constant("int", str(self.factor))
        )
        # Next: i_out++
        next_out = c_ast.UnaryOp("p++", c_ast.ID(i_out))

        # Wrap inner loop
        outer_compound = c_ast.Compound(block_items=[inner_for])
        outer_for = c_ast.For(init_out, cond_out, next_out, outer_compound)

        return outer_for


def ast_loop_split(code):
    ast = parse_code_ast(code)
    visitor = SplitForLoopVisitor()
    visitor.visit(ast)

    generator = c_generator.CGenerator()
    return generator.visit(ast)


if __name__ == "__main__":
    code = """
    int factorial(int result) {
        # pragma loop_split(2)
        for (int i=0; i < 10; i++) {
            result += i;
        }
        return result;
    }
    """
    final_code = ast_loop_split(code)
    print(final_code)

    code = """
    void add_kernel(float * A, float * B, float * T_add) {
        for (int i=0; i < 256; i++) {
            # pragma loop_split(4)
            for (int j=0; j < 1024; j++) {
                T_add[((i * 1024) + j)] = (A[((i * 1024) + j)] +
                       B[((i * 1024) + j)]);
            }
        }
    }
    """
    final_node = ast_loop_split(code)
    print(final_node)

    code = """
    void softmax(float * A, float * T_softmax_norm)
    {
    for (int threadIdxx=0; threadIdxx < 5; ++threadIdxx)
    {
        float maxVal = A[threadIdxx * 128];

        # pragma loop_split(factor=4)
        for (int i=1; i < 128; ++i)
        {
        if (A[(threadIdxx * 128) + i] > maxVal)
        {
            maxVal = A[(threadIdxx * 128) + i];
        }
        }

        float denom = 0.0f;

        # pragma loop_split(factor=4)
        for (int i=0; i < 128; ++i)
        {
        T_softmax_norm[(threadIdxx * 128) +
                        i] = expf(A[(threadIdxx * 128) + i] - maxVal);
        }

        # pragma loop_split(factor=4)
        for (int i=0; i < 128; ++i)
        {
        denom += T_softmax_norm[(threadIdxx * 128) + i];
        }

        # pragma loop_split(factor=4)
        for (int i=0; i < 128; ++i)
        {
        T_softmax_norm[(threadIdxx * 128) + i] /= denom;
        }
    }
    }
    """
    final_node = ast_loop_split(code)
    print(final_node)
