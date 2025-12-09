from string import Template

from pycparser import c_ast, c_generator, c_parser

from falcon.util import (
    NodeTransformer,
    add_memory_prefix,
    generate_code,
    parse_code_ast,
    remove_target_prefix,
)


class PragmaVisitor(NodeTransformer):
    def __init__(self):
        self.inst = []

    def visit_Compound(self, node):
        # Obtain `block_items`
        blocks = node.block_items
        if not blocks:
            return node

        new_block_items = []
        skip_next = False

        # Iterate through `block_items`, searching for the combination of
        # `Pragma` and `for`.
        for index, subnode in enumerate(blocks):
            if skip_next:
                # Skip the next node (for loop) as it has already been
                # processed.
                skip_next = False
                continue

            # Check if it is `#pragma software_pipeline`
            if (
                isinstance(subnode, c_ast.Pragma)
                and subnode.string == "software_pipeline"
            ):
                if index + 1 < len(blocks) and isinstance(
                    blocks[index + 1], c_ast.For
                ):
                    pipeline_for = blocks[index + 1]

                    ext = pipeline_for.cond.right.value

                    new_call = None

                    # Add the replacement call.
                    new_block_items.append(new_call)

                    # Set to skip the next `for` loop.
                    skip_next = True
                else:
                    # If `for` is not found, continue adding the current node.
                    new_block_items.append(subnode)
            else:
                # If it's neither `#pragma` nor `for`, directly add the node.
                new_block_items.append(subnode)

        # Replace `block_items`
        node.block_items = new_block_items
        return node


class SoftwarePipelineInserter(NodeTransformer):
    """A pycparser AST transformer that inserts a `#pragma software_pipeline`
    immediately before every `for` loop in a C function body."""

    def visit_Compound(self, node):
        # If there are no statements, return as is
        if not node.block_items:
            return node

        new_items = []
        for stmt in node.block_items:
            # If the statement is a for-loop, insert a pragma first
            if isinstance(stmt, c_ast.For):
                # Inspect loop body
                body = stmt.stmt
                if isinstance(body, c_ast.Compound):
                    generator = c_generator.CGenerator()
                    body_code = generator.visit(body)
                # Append (and further transform) the for-loop
                new_items.append(self.visit(stmt))
            else:
                # Recursively visit other statements
                new_items.append(self.visit(stmt))

        node.block_items = new_items
        return node


def apply_software_pipeline(source_code: str) -> str:
    """Parse the given C source code, apply the SoftwarePipelineInserter pass,
    and return the transformed code as a string."""
    # Parse into AST
    parser = c_parser.CParser()
    ast = parser.parse(source_code)

    # Transform
    transformer = SoftwarePipelineInserter()
    transformed = transformer.visit(ast)

    # Generate C code
    generator = c_generator.CGenerator()
    return generator.visit(transformed)


def smt_double_buffer(source_code):
    code = remove_target_prefix(source_code)
    code = apply_software_pipeline(code)
    ast = parse_code_ast(code)
    visitor = PragmaVisitor()
    visitor.visit(ast)
    if not visitor.inst:
        return source_code
    output_code = op_template[visitor.inst].substitute(inst=visitor.inst)
    modify_code = generate_code(ast)
    return add_memory_prefix(output_code + modify_code)


if __name__ == "__main__":
    pass
