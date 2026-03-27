import logging
import os
import re
import tempfile
from dataclasses import dataclass
from typing import List, Optional

from falcon.util import make_full_func


MOCK_SYCL_TENSORIZATION_HEADER = """
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
        void barrier(int) const;
    };
    template <typename T, int dimensions = 1> struct local_accessor {
        local_accessor(range<dimensions>, struct handler &) {}
        T *operator[](int) const;
    };
    namespace access {
        enum fence_space { local_space };
    }
    struct handler {
        template <typename T, typename F> void parallel_for(T, F) {}
    };
    struct queue {
        template <typename F> void submit(F) {}
        void wait() {}
    };
    template <typename T> T mad(T, T, T);
    typedef uint16_t half;
}
"""


def _split_csv(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


@dataclass
class SyclFunctionInfo:
    signature_prefix: str
    func_name: str
    params: List[str]
    param_types: dict
    scalar_params: List[str]
    queue_name: str
    alias_lines: List[str]


@dataclass
class SyclTensorizationPlan:
    mode: str
    function_info: SyclFunctionInfo
    row_var: str
    col_var: str
    red_var: str
    m_expr: str
    k_expr: str
    n_expr: str
    input_names: Optional[List[str]] = None
    output_name: Optional[str] = None

    def can_generate_tiled_kernel(self) -> bool:
        return (
            self.input_names is not None
            and len(self.input_names) >= 2
            and self.output_name is not None
        )

    def to_kernel_info(self) -> dict:
        if not self.can_generate_tiled_kernel():
            raise ValueError("Tensorization plan does not contain enough data.")

        a_name, b_name = self.input_names[:2]
        output_name = self.output_name
        param_types = self.function_info.param_types
        return {
            "signature_prefix": self.function_info.signature_prefix,
            "func_name": self.function_info.func_name,
            "params": self.function_info.params,
            "param_types": param_types,
            "scalar_params": self.function_info.scalar_params,
            "queue_name": self.function_info.queue_name,
            "alias_lines": self.function_info.alias_lines,
            "row_var": self.row_var,
            "col_var": self.col_var,
            "red_var": self.red_var,
            "m_expr": self.m_expr,
            "k_expr": self.k_expr,
            "n_expr": self.n_expr,
            "a_name": a_name,
            "b_name": b_name,
            "c_name": output_name,
            "a_elem_type": _pointee_type(param_types[a_name]),
            "b_elem_type": _pointee_type(param_types[b_name]),
            "c_elem_type": _pointee_type(param_types[output_name]),
        }


def _pointee_type(param_type: str) -> str:
    base = param_type.replace("*", " ").replace("&", " ")
    base = re.sub(r"\bconst\b", "", base)
    base = re.sub(r"\s+", " ", base).strip()
    return base or "float"


class SyclTensorizationAstExtractor:
    def __init__(self, file_path: str, original_code: str, parsed_code: str):
        import clang.cindex

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

    def _get_text(self, cursor) -> str:
        start = cursor.extent.start.offset
        end = cursor.extent.end.offset
        return self.parsed_code_bytes[start:end].decode("utf-8", "ignore")

    def extract_plan(self) -> Optional[SyclTensorizationPlan]:
        cached_plans: List[SyclTensorizationPlan] = []
        plain_plans: List[SyclTensorizationPlan] = []

        for func_cursor in self._iter_function_defs(self.tu.cursor):
            func_text = self._get_text(func_cursor)
            if "parallel_for" not in func_text:
                continue

            function_info = self._extract_function_info(func_cursor, func_text)
            if function_info is None:
                continue

            for call_cursor in self._collect_parallel_for_calls(func_cursor):
                plan = self._analyze_parallel_for_call(
                    call_cursor, function_info, func_text
                )
                if plan is None:
                    continue
                if plan.mode == "cached":
                    cached_plans.append(plan)
                elif plan.mode == "plain":
                    plain_plans.append(plan)

        if len(cached_plans) == 1:
            return cached_plans[0]
        if len(cached_plans) > 1:
            logging.warning(
                "SYCL tensorization AST fallback found multiple cached candidates; "
                "skip to avoid ambiguous rewrite."
            )
            return None

        if len(plain_plans) == 1:
            return plain_plans[0]
        if len(plain_plans) > 1:
            logging.warning(
                "SYCL tensorization AST fallback found multiple plain candidates; "
                "skip to avoid ambiguous rewrite."
            )
        return None

    def _iter_function_defs(self, cursor):
        for child in cursor.get_children():
            if child.kind == self.CursorKind.FUNCTION_DECL and child.is_definition():
                yield child
            yield from self._iter_function_defs(child)

    def _collect_parallel_for_calls(self, cursor):
        calls = []
        self._recursive_collect_parallel_for(cursor, calls)
        return calls

    def _recursive_collect_parallel_for(self, cursor, results):
        for child in cursor.get_children():
            if child.kind in [self.CursorKind.CALL_EXPR, self.CursorKind.UNEXPOSED_EXPR]:
                text = self._get_text(child)
                if "parallel_for" in text.split("(", 1)[0]:
                    results.append(child)
                    continue
            self._recursive_collect_parallel_for(child, results)

    def _analyze_parallel_for_call(
        self,
        call_cursor,
        function_info: SyclFunctionInfo,
        func_text: str,
    ) -> Optional[SyclTensorizationPlan]:
        range_node = None
        lambda_node = None
        for child in call_cursor.get_children():
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
        row_var, col_var = self._extract_index_vars(lambda_text, index_var)
        reduction = self._extract_reduction_loop(lambda_text)
        if reduction is None:
            return None

        red_var, k_expr, reduction_body = reduction
        m_expr, n_expr = self._extract_output_bounds(
            lambda_text,
            index_var,
            row_var,
            col_var,
            bounds,
        )

        if self._looks_like_cached_candidate(func_text, lambda_text, reduction_body):
            cached_operands = self._extract_matmul_pragma(lambda_text)
            return SyclTensorizationPlan(
                mode="cached",
                function_info=function_info,
                row_var=row_var,
                col_var=col_var,
                red_var=red_var,
                m_expr=m_expr,
                k_expr=k_expr,
                n_expr=n_expr,
                input_names=cached_operands[0] if cached_operands is not None else None,
                output_name=cached_operands[1] if cached_operands is not None else None,
            )

        operands = self._extract_plain_operands(lambda_text, reduction_body)
        if operands is None:
            return None

        input_names, output_name = operands
        if any(name not in function_info.param_types for name in input_names[:2]):
            return None
        if output_name not in function_info.param_types:
            return None

        return SyclTensorizationPlan(
            mode="plain",
            function_info=function_info,
            row_var=row_var,
            col_var=col_var,
            red_var=red_var,
            m_expr=m_expr,
            k_expr=k_expr,
            n_expr=n_expr,
            input_names=input_names[:2],
            output_name=output_name,
        )

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
        if direct_match is not None:
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

    def _extract_index_vars(self, text: str, index_var: str):
        row_var = self._extract_index_var(text, index_var, 0, "row")
        col_var = self._extract_index_var(text, index_var, 1, "col")
        return row_var, col_var

    def _extract_index_var(
        self, text: str, index_var: str, dim: int, fallback_name: str
    ) -> str:
        pattern = (
            rf"(?:const\s+)?[A-Za-z_][\w:<>]*\s+(?P<var>\w+)\s*=\s*"
            rf"(?:{re.escape(index_var)}\.(?:get_id|get_global_id)\(\s*{dim}\s*\)|"
            rf"{re.escape(index_var)}\[\s*{dim}\s*\])\s*;"
        )
        match = re.search(pattern, text)
        if match is not None:
            return match.group("var")
        return fallback_name

    def _extract_reduction_loop(self, text: str):
        loop_iter = re.finditer(
            r"for\s*\(\s*(?:const\s+)?[A-Za-z_][\w:<>]*\s+(?P<red>\w+)\s*=\s*0\s*;\s*"
            r"(?P=red)\s*<\s*(?P<kexpr>[^;]+?)\s*;\s*(?:\+\+(?P=red)|(?P=red)\+\+)\s*\)\s*"
            r"{(?P<body>.*?)}",
            text,
            re.S,
        )
        for candidate in loop_iter:
            body = candidate.group("body")
            if "*" in body or "sycl::mad(" in body:
                return (
                    candidate.group("red"),
                    candidate.group("kexpr").strip(),
                    body,
                )
        return None

    def _extract_output_bounds(
        self,
        text: str,
        index_var: str,
        row_var: str,
        col_var: str,
        bounds: List[str],
    ):
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

    def _looks_like_cached_candidate(
        self, func_text: str, lambda_text: str, reduction_body: str
    ) -> bool:
        if "local_accessor<" not in func_text:
            return False
        if "barrier(access::fence_space::local_space)" not in lambda_text:
            return False
        if "sum" not in lambda_text and "acc" not in lambda_text:
            return False
        if "*" not in reduction_body and "sycl::mad(" not in reduction_body:
            return False
        tile_names = re.findall(r"local_accessor<[^>]+>\s+(\w+)", func_text)
        tile_refs = [name for name in tile_names if re.search(rf"\b{name}\s*\[", reduction_body)]
        if len(tile_refs) < 2:
            return False
        if re.search(r"\w+\s*\[[^\]]+\]\s*=\s*\w+\s*;", lambda_text) is None:
            return False
        return True

    def _extract_plain_operands(
        self, lambda_text: str, reduction_body: str
    ) -> Optional[tuple]:
        pragma_operands = self._extract_matmul_pragma(lambda_text)
        if pragma_operands is not None:
            return pragma_operands

        output_name = None
        post_assign = re.search(
            r"([A-Za-z_]\w*)\s*\[[^\]]+\]\s*=\s*[A-Za-z_]\w*\s*;",
            lambda_text,
        )
        if post_assign is not None:
            output_name = post_assign.group(1)
        else:
            inplace_assign = re.search(
                r"([A-Za-z_]\w*)\s*\[[^\]]+\]\s*(?:\+?=)",
                reduction_body,
            )
            if inplace_assign is not None:
                output_name = inplace_assign.group(1)

        if output_name is None:
            return None

        input_names = []
        for array_name, _ in re.findall(r"([A-Za-z_]\w*)\s*\[([^\]]+)\]", reduction_body):
            if array_name == output_name or array_name in input_names:
                continue
            if array_name.endswith("_tile"):
                continue
            input_names.append(array_name)
            if len(input_names) == 2:
                break

        if len(input_names) != 2:
            return None
        return input_names, output_name

    def _extract_matmul_pragma(self, text: str) -> Optional[tuple]:
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
        return inputs, outputs[0]

    def _extract_function_info(
        self, func_cursor, func_text: str
    ) -> Optional[SyclFunctionInfo]:
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

        if queue_name is None:
            return None

        alias_lines = re.findall(
            r"^\s*(using\s+\w+\s*=\s*sycl::[A-Za-z_:<>]+\s*;)\s*$",
            self.original_code,
            re.M,
        )
        return SyclFunctionInfo(
            signature_prefix=signature_prefix,
            func_name=func_cursor.spelling,
            params=params,
            param_types=param_types,
            scalar_params=scalar_params,
            queue_name=queue_name,
            alias_lines=alias_lines,
        )


def ast_sycl_tensorization(code: str) -> str:
    from falcon.src.post_processing.post_processing import (
        _generate_sycl_tiled_matmul,
        _is_sycl_tensorized,
        _promote_sycl_cached_matmul,
    )

    if _is_sycl_tensorized(code):
        return make_full_func(code, "sycl")

    code_without_include = re.sub(r"#include\s*<sycl/sycl\.hpp>", "", code)
    parsed_code = MOCK_SYCL_TENSORIZATION_HEADER + "\n" + code_without_include

    fd, path = tempfile.mkstemp(suffix=".cpp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            tmp.write(parsed_code)

        extractor = SyclTensorizationAstExtractor(path, code, parsed_code)
        plan = extractor.extract_plan()
        if plan is None:
            return make_full_func(code, "sycl")

        if plan.mode == "cached":
            promoted = _promote_sycl_cached_matmul(code)
            if _is_sycl_tensorized(promoted):
                return make_full_func(promoted, "sycl")
            if plan.can_generate_tiled_kernel():
                return make_full_func(
                    _generate_sycl_tiled_matmul(
                        plan.to_kernel_info(), use_fma=True
                    ),
                    "sycl",
                )
            logging.warning(
                "SYCL tensorization AST fallback validated a cached kernel but "
                "promotion did not produce a tensorized result."
            )
            return make_full_func(code, "sycl")

        return make_full_func(
            _generate_sycl_tiled_matmul(plan.to_kernel_info(), use_fma=True),
            "sycl",
        )
    finally:
        if os.path.exists(path):
            os.remove(path)
