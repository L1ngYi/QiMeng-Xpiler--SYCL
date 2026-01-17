import json
import re

# 导入 AST (抽象语法树) 相关的缓冲区内联工具
from falcon.buffer_inline import ast_buffer_inline
# 导入 LLM 调用客户端
from falcon.client import invoke_llm
# 导入代码简化工具
from falcon.simplification import simplify_code
# 导入预定义的 Prompt 模板和 Demo 示例
# 注意：LOOP_RECOVERY 旨在将并行代码（如 CUDA）还原为串行循环
from falcon.src.pre_processing.preprocessing_prompt import (
    LOOP_RECOVERY_DEMO_CUDA,
    LOOP_RECOVERY_PROMPT_CUDA,
)
from falcon.src.prompt.prompt import SYSTEM_PROMPT
# 导入基于 AST 的语句简化工具
from falcon.stmt_simplification import ast_stmt_simplification
# 导入代码提取（从 Markdown 块提取）和函数包装工具
from falcon.util import extract_code, make_full_func


def run_loop_recovery(code, target):
    """
    运行“循环恢复”流程：利用 LLM 将包含并行语义（如 threadIdx, blockIdx）的 CUDA/HIP 代码
    还原为标准的、串行的 C++ 嵌套循环形式。
    这通常是代码转译的第一步，先回到通用 IR (C++)，再进行后续优化。
    """
    # 构建 Prompt 模板
    # 注意：这里的 {TENSORIZATION_PROMPT} 实际上是个命名混淆（可能是复制粘贴错误），
    # 实际上下面代码填充的是 LOOP_RECOVERY_PROMPT_CUDA (循环恢复的提示词)。
    PROMPT = """
    {SYSTEM_PROMPT}

    {TENSORIZATION_PROMPT}

    Example:
    {LOOP_RECOVERY_DEMO}

    Input cuda Code:
    {code}
    Output C++ Code:

    Please return the output kernel function without any additional information.
    """

    # 填充系统级提示词
    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)
    prompt_des = None
    # 根据目标平台选择对应的任务描述 Prompt
    if target == "cuda" or target == "hip":
        prompt_des = LOOP_RECOVERY_PROMPT_CUDA
    elif target == "sycl":
        #TODO: 添加 SYCL 的 Loop Recovery Prompt
        #prompt_des = LOOP_RECOVERY_PROMPT_SYCL
        pass

    prompt_demo = None
    # 根据目标平台选择对应的 Few-Shot 示例
    if target == "cuda" or target == "hip":
        prompt_demo = LOOP_RECOVERY_DEMO_CUDA
    elif target == "sycl":
        #TODO: 添加 SYCL 的 Loop Recovery Demo
        #prompt_demo = LOOP_RECOVERY_DEMO_SYCL
        pass
    


    # 替换模板中的占位符
    # 再次注意：变量名虽然叫 TENSORIZATION_PROMPT，但填入的是 prompt_des (即 Loop Recovery Prompt)
    PROMPT = PROMPT.replace("{TENSORIZATION_PROMPT}", prompt_des)
    PROMPT = PROMPT.replace("{LOOP_RECOVERY_DEMO}", prompt_demo)
    PROMPT = PROMPT.replace("{code}", code)
    
    # 调用 LLM 获取结果
    content = invoke_llm(PROMPT)
    # 从 LLM 返回的文本中提取代码块（去除解释性文字）
    code_content = extract_code(content)
    
    if code_content:
        # 简单的文本替换，规范化变量命名
        code_content = code_content.replace("coreId", "core_id")
        code_content = code_content.replace("clusterId", "cluster_id")
        # 确保代码是一个完整的函数定义
        code_content = make_full_func(code_content, target)
        return code_content
    return None


def detensorization(op, code, document):
    """
    执行“去张量化”：将特定的硬件加速指令（如 mma_sync）还原为普通的循环实现。
    
    参数:
      op: 指令名称 (如 'mma_sync')
      code: 当前代码
      document: 该指令的官方文档描述 (从 JSON 加载)
    """
    PROMPT = """
    {SYSTEM_PROMPT}
    Please transform the instruction {op} in following code into sequential for loop.

    {code}

    accordingt to the description of instruction {op} as follows:

    {document}

    Please return the output kernel function without any additional information.
    """

    # 构建 Prompt：要求 LLM 根据文档说明 (document) 将指令 (op) 翻译回循环
    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)
    PROMPT = PROMPT.replace("{document}", document)
    PROMPT = PROMPT.replace("{code}", code)
    PROMPT = PROMPT.replace("{op}", op)

    # 调用 LLM 并提取代码
    content = invoke_llm(PROMPT)
    code_content = extract_code(content)
    return code_content


def extract_cuda_instructions(code):
    """
    从代码体中提取所有函数调用名称，用于识别潜在的 CUDA 固有指令（Intrinsics）。
    """
    # 简单粗暴地切分字符串，获取函数体部分（假设第一个 '{' 之后是函数体）
    body = code.split("{", 1)[1]

    # 2) Find all identifier( patterns in the body
    # 使用正则表达式寻找所有 "函数名(" 的模式
    all_calls = re.findall(r"\b([A-Za-z_]\w*)\s*\(", body)

    # 3) Filter out C/C++ keywords and types
    # 定义排除列表：排除标准的 C/C++ 关键字，保留可能的 API 调用
    exclude = {
        "for",
        "if",
        "while",
        "switch",
        "return",
        "int",
        "float",
        "half",
        "void",
    }
    func_calls = [name for name in all_calls if name not in exclude]

    # 4) Dedupe and sort
    # 去重并排序
    unique_funcs = sorted(set(func_calls))
    return unique_funcs


def run_detensorization(code, target):
    """
    运行“去张量化”主流程：扫描代码中的特定指令，查阅文档，并逐个替换为普通循环。
    """
    if target == "cuda":
        # 加载 CUDA 指令与其文档的映射表 (JSON 文件)
        # 注意：这里使用了相对路径，如果运行目录不对可能会报错
        op_dict = json.load(
            open("./falcon/documents/cuda_op_tensorization.json", "r")
        )

        # 提取代码中所有的函数调用
        instructions = extract_cuda_instructions(code)
        if instructions is not None:
            for inst in instructions:
                # 这里的逻辑是：如果提取到的指令在我们的 JSON 文档库里（op_dict），
                # 就调用 detensorization 把它翻译掉。
                # 注意：这里隐式依赖 extract_cuda_instructions 的结果包含在 op_dict 的 key 中，
                # 如果 inst 不在 op_dict 中，这里可能会抛出 KeyError (代码未做 try-except 保护)。
                code = detensorization(inst, code, op_dict[inst])

    # 在所有 LLM 处理完成后，进行一系列基于 AST 的确定性简化和清理
    code = simplify_code(code)                 # 常规代码简化
    code = ast_stmt_simplification(code)       # AST 级别的语句简化
    code = ast_buffer_inline(code)             # 缓冲区内联
    code = make_full_func(code, target)        # 确保函数结构完整
    return code


def pre_processing_pipeline(code, target):
    """This function transforms the given code by performing two main transformations:
        1. Convert parallel loop variables (e.g., OpenMP, cuda) into standard C for loops.
        2. Convert SIMD tensor operations into scalar for-loop based calculations.
    :param func_content: The content of the function (code) to be transformed.
    :return: Transformed code after applying the two transformations."""
    
    # 第一步：运行循环恢复 (Loop Recovery)
    # 将 threadIdx 等并行原语转回 C 循环
    code = run_loop_recovery(code, target)
    
    # ！！！关键逻辑断点！！！
    # 虽然文档字符串（docstring）声称进行了“two main transformations”，
    # 且上方已经定义了 `run_detensorization` 函数，
    # 但在此处并没有被调用。这意味着“去张量化”步骤在实际管线中被跳过了。
    # 这种缺失印证了这是一份未完成的原型代码。
    
    return code