import os
import re
import tempfile

from falcon.util import make_full_func


MOCK_SYCL_HEADER = """
#include <stdint.h>
namespace sycl {
    template <int dimensions = 1> struct range { range(int, int=1, int=1){} };
    template <int dimensions = 1> struct nd_range { nd_range(range<dimensions>, range<dimensions>){} };
    template <int dimensions = 1> struct item {
        int get_id(int) const;
        int get_global_id(int) const;
    };
    template <int dimensions = 1> struct nd_item {
        int get_global_id(int) const;
        int get_local_id(int) const;
    };
    struct handler {
        template <typename T, typename F> void parallel_for(T, F) {}
    };
    struct queue {
        template <typename F> void submit(F) {}
        void wait() {}
    };
    typedef uint16_t half;
}
"""


def _split_csv(text):
    return [item.strip() for item in text.split(",") if item.strip()]


class SyclCacheAstExtractor:
    def __init__(self, file_path, original_code, parsed_code):
        import clang.cindex

        self.clang = clang.cindex
        self.CursorKind = clang.cindex.CursorKind
        self.file_path = file_path
        self.original_code = original_code
        self.parsed_code = parsed_code
        self.parsed_code_bytes = parsed_code.encode("utf-8")
        self.index = clang.cindex.Index.create()
        self.tu = self.index.parse(
            file_path,
            args=["-std=c++17", "-x", "c++", "-w", "-fparse-all-comments"],
        )

    def _get_text(self, cursor):
        start = cursor.extent.start.offset
        end = cursor.extent.end.offset
        return self.parsed_code_bytes[start:end].decode("utf-8", "ignore")

    def extract(self):
        return self._find_function(self.tu.cursor)

    def _find_function(self, cursor):
        for child in cursor.get_children():
            if child.kind == self.CursorKind.FUNCTION_DECL and child.is_definition():
                info = self._analyze_function(child)
                if info is not None:
                    return info
            info = self._find_function(child)
            if info is not None:
                return info
        return None

    def _analyze_function(self, func_cursor):
        func_text = self._get_text(func_cursor)
        if "parallel_for" not in func_text or "#pragma operation(matmul" not in func_text:
            return None

        kernel_call = self._find_parallel_for_call(func_cursor)
        if kernel_call is None:
            return None

        kernel_info = self._analyze_parallel_for_call(kernel_call)
        if kernel_info is None:
            return None

        func_info = self._extract_function_signature(func_cursor, func_text)
        if func_info is None:
            return None

        input_names = kernel_info["inputs"][:2]
        output_name = kernel_info["outputs"][0]
        if input_names[0] not in func_info["param_types"]:
            return None
        if input_names[1] not in func_info["param_types"]:
            return None
        if output_name not in func_info["param_types"]:
            return None

        return {
            **func_info,
            "row_var": kernel_info["row_var"],
            "col_var": kernel_info["col_var"],
            "red_var": kernel_info["red_var"],
            "m_expr": kernel_info["m_expr"],
            "k_expr": kernel_info["k_expr"],
            "n_expr": kernel_info["n_expr"],
            "a_name": input_names[0],
            "b_name": input_names[1],
            "c_name": output_name,
            "a_elem_type": self._pointee_type(func_info["param_types"][input_names[0]]),
            "b_elem_type": self._pointee_type(func_info["param_types"][input_names[1]]),
            "c_elem_type": self._pointee_type(func_info["param_types"][output_name]),
        }

    def _find_parallel_for_call(self, cursor):
        for child in cursor.get_children():
            if child.kind in [self.CursorKind.CALL_EXPR, self.CursorKind.UNEXPOSED_EXPR]:
                text = self._get_text(child)
                head = text.split("(", 1)[0]
                if "parallel_for" in head and "#pragma operation(matmul" in text:
                    return child
            nested = self._find_parallel_for_call(child)
            if nested is not None:
                return nested
        return None

    def _analyze_parallel_for_call(self, call_node):
        range_node = None
        lambda_node = None
        for child in call_node.get_children():
            clean_type = child.type.spelling.replace("const", "").replace("&", "").strip()
            child_text = self._get_text(child)
            if "range" in clean_type or "nd_range" in clean_type:
                range_node = child
            elif child.kind == self.CursorKind.LAMBDA_EXPR or "lambda" in clean_type:
                lambda_node = self._drill_down_to_lambda(child)
            elif child.kind == self.CursorKind.UNEXPOSED_EXPR:
                if "range" in child_text:
                    range_node = child
                elif "{" in child_text and "[" in child_text:
                    lambda_node = child

        if range_node is None or lambda_node is None:
            return None

        dims, bounds = self._parse_range(range_node)
        if dims != 2:
            return None

        lambda_text, index_var = self._extract_lambda_text(lambda_node)
        pragma_info = self._extract_matmul_pragma(lambda_text)
        if pragma_info is None:
            return None

        row_var, col_var = self._extract_index_vars(lambda_text, index_var)
        red_var, k_expr = self._extract_reduction_info(
            lambda_text, pragma_info["inputs"][:2]
        )
        if k_expr is None:
            return None

        m_expr, n_expr = self._extract_output_bounds(
            lambda_text,
            index_var,
            row_var,
            col_var,
            bounds,
        )

        return {
            "inputs": pragma_info["inputs"],
            "outputs": pragma_info["outputs"],
            "row_var": row_var,
            "col_var": col_var,
            "red_var": red_var,
            "m_expr": m_expr,
            "k_expr": k_expr,
            "n_expr": n_expr,
        }

    def _drill_down_to_lambda(self, node):
        if node.kind == self.CursorKind.LAMBDA_EXPR:
            return node
        if "{" in self._get_text(node):
            for child in node.get_children():
                if child.kind == self.CursorKind.LAMBDA_EXPR:
                    return child
            return node
        return node

    def _parse_range(self, range_node):
        text = self._get_text(range_node)
        dims = 1
        if "<2>" in text:
            dims = 2
        elif "<3>" in text:
            dims = 3

        range_map = {}
        for match in re.finditer(
            r"(?:sycl::)?range<\d+>\s+(\w+)\s*\(([^)]+)\)",
            self.parsed_code,
        ):
            range_map[match.group(1)] = _split_csv(match.group(2))

        bounds = []
        direct_match = re.search(r"(?:nd_)?range<\d+>\s*\(([^)]+)\)", text)
        if direct_match:
            args = _split_csv(direct_match.group(1))
            if args and args[0] in range_map:
                bounds = range_map[args[0]]
            else:
                bounds = args

        if not bounds:
            bounds = ["M", "N", "K"][:dims]
        return dims, bounds

    def _extract_lambda_text(self, lambda_node):
        index_var = "item"
        lambda_text = self._get_text(lambda_node)
        idx_match = re.search(
            r"\(\s*(?:const\s+)?(?:sycl::)?(?:nd_)?item<\d+>\s+(\w+)\s*\)",
            lambda_text,
        )
        if idx_match is not None:
            index_var = idx_match.group(1)
        return lambda_text, index_var

    def _extract_matmul_pragma(self, text):
        pragma_match = re.search(
            r"#\s*pragma\s+operation\s*\(\s*matmul\s*\(\s*"
            r"input\[\s*(?P<inputs>[^\]]+)\s*\]\s*,\s*"
            r"output\[\s*(?P<outputs>[^\]]+)\s*\]\s*\)\s*\)",
            text,
            re.S,
        )
        if pragma_match is None:
            return None
        inputs = _split_csv(pragma_match.group("inputs"))
        outputs = _split_csv(pragma_match.group("outputs"))
        if len(inputs) < 2 or not outputs:
            return None
        return {"inputs": inputs, "outputs": outputs}

    def _extract_index_vars(self, text, index_var):
        row_var = self._extract_index_var(text, index_var, 0, "row")
        col_var = self._extract_index_var(text, index_var, 1, "col")
        return row_var, col_var

    def _extract_index_var(self, text, index_var, dim, fallback_name):
        pattern = (
            rf"(?:const\s+)?[A-Za-z_][\w:<>]*\s+(?P<var>\w+)\s*=\s*"
            rf"(?:{re.escape(index_var)}\.(?:get_id|get_global_id)\(\s*{dim}\s*\)|"
            rf"{re.escape(index_var)}\[\s*{dim}\s*\])\s*;"
        )
        match = re.search(pattern, text)
        if match is not None:
            return match.group("var")
        return fallback_name

    def _extract_reduction_info(self, text, inputs):
        pragma_loop_match = re.search(
            r"#\s*pragma\s+operation\s*\(\s*matmul\(.*?\)\s*\)\s*"
            r"for\s*\(\s*(?:const\s+)?[A-Za-z_][\w:<>]*\s+(?P<red>\w+)\s*=\s*0\s*;\s*"
            r"(?P=red)\s*<\s*(?P<kexpr>[^;]+?)\s*;\s*(?:\+\+(?P=red)|(?P=red)\+\+)\s*\)",
            text,
            re.S,
        )
        if pragma_loop_match is not None:
            return pragma_loop_match.group("red"), pragma_loop_match.group("kexpr").strip()

        loop_iter = re.finditer(
            r"for\s*\(\s*(?:const\s+)?[A-Za-z_][\w:<>]*\s+(?P<red>\w+)\s*=\s*0\s*;\s*"
            r"(?P=red)\s*<\s*(?P<kexpr>[^;]+?)\s*;\s*(?:\+\+(?P=red)|(?P=red)\+\+)\s*\)\s*"
            r"{(?P<body>.*?)}",
            text,
            re.S,
        )
        for candidate in loop_iter:
            body = candidate.group("body")
            if all(re.search(rf"\b{re.escape(name)}\s*\[", body) for name in inputs):
                return candidate.group("red"), candidate.group("kexpr").strip()
        return "k", None

    def _extract_output_bounds(self, text, index_var, row_var, col_var, bounds):
        m_expr = bounds[0] if len(bounds) > 0 else "M"
        n_expr = bounds[1] if len(bounds) > 1 else "N"
        row_pattern = (
            rf"(?:{re.escape(row_var)}|"
            rf"{re.escape(index_var)}\.(?:get_id|get_global_id)\(\s*0\s*\)|"
            rf"{re.escape(index_var)}\[\s*0\s*\])"
        )
        col_pattern = (
            rf"(?:{re.escape(col_var)}|"
            rf"{re.escape(index_var)}\.(?:get_id|get_global_id)\(\s*1\s*\)|"
            rf"{re.escape(index_var)}\[\s*1\s*\])"
        )
        cond_match = re.search(
            rf"if\s*\(\s*{row_pattern}\s*<\s*(?P<m>[^&|)]+?)\s*&&\s*"
            rf"{col_pattern}\s*<\s*(?P<n>[^&|)]+?)\s*\)",
            text,
        )
        if cond_match is not None:
            return cond_match.group("m").strip(), cond_match.group("n").strip()
        return m_expr, n_expr

    def _extract_function_signature(self, func_cursor, func_text):
        func_head = func_text.split("{", 1)[0]
        prefix_match = re.search(
            rf"(?P<prefix>.*?)\b{re.escape(func_cursor.spelling)}\s*\(",
            func_head,
            re.S,
        )
        signature_prefix = (
            prefix_match.group("prefix").strip()
            if prefix_match is not None
            else func_cursor.result_type.spelling.strip()
        )
        params = []
        param_types = {}
        scalar_params = []
        queue_name = None
        for arg in func_cursor.get_arguments():
            param_name = arg.spelling.strip()
            param_type = arg.type.spelling.strip()
            param_text = self._get_text(arg).strip()
            if not param_text:
                param_text = f"{param_type} {param_name}".strip()
            params.append(param_text)
            param_types[param_name] = param_type
            if "queue" in param_type:
                queue_name = param_name
            elif "*" not in param_type and "&" not in param_type:
                scalar_params.append(param_name)

        alias_lines = re.findall(
            r"^\s*(using\s+\w+\s*=\s*sycl::[A-Za-z_:<>]+\s*;)\s*$",
            self.original_code,
            re.M,
        )
        if queue_name is None:
            return None

        return {
            "signature_prefix": signature_prefix,
            "func_name": func_cursor.spelling,
            "params": params,
            "param_types": param_types,
            "scalar_params": scalar_params,
            "queue_name": queue_name,
            "alias_lines": alias_lines,
        }

    def _pointee_type(self, param_type):
        base = param_type.replace("*", " ").replace("&", " ")
        base = re.sub(r"\bconst\b", "", base)
        base = re.sub(r"\s+", " ", base).strip()
        return base or "float"


def ast_sycl_auto_cache(code):
    if "local_accessor<" in code or "#pragma operation(matmul" not in code:
        return make_full_func(code, "sycl")

    from falcon.src.post_processing.post_processing import _generate_sycl_tiled_matmul

    code_without_include = re.sub(r'#include\s*<sycl/sycl\.hpp>', "", code)
    parsed_code = MOCK_SYCL_HEADER + "\n" + code_without_include

    fd, path = tempfile.mkstemp(suffix=".cpp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            tmp.write(parsed_code)

        extractor = SyclCacheAstExtractor(path, code, parsed_code)
        info = extractor.extract()
        if info is None:
            return make_full_func(code, "sycl")

        return make_full_func(_generate_sycl_tiled_matmul(info), "sycl")
    finally:
        if os.path.exists(path):
            os.remove(path)
