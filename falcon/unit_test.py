import logging
import os
import shutil
import subprocess

from falcon.util import get_target

# Configure the log
# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,  # Set the log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Set the log format.
)

# 定义算子名称到测试脚本路径的映射
# {target} 占位符稍后会根据硬件平台（如 cuda, dlboost 等）进行格式化替换
test_file_map = {
    "deformable": "benchmark/evaluation/{target}_test/test_deformable_attention.py",
    "layernorm": "benchmark/evaluation/{target}_test/test_layer_norm_cuda.py",
    "mha": "benchmark/evaluation/{target}_test/test_mha_cuda.py",
    "rmsnorm": "benchmark/evaluation/{target}_test/test_rms_norm_cuda.py",
    "gemm": "benchmark/evaluation/{target}_test/test_gemm.py",
    "gemv": "benchmark/evaluation/{target}_test/test_gemv.py",
    "bmm": "benchmark/evaluation/{target}_test/test_bmm.py",
    "conv1d": "benchmark/evaluation/{target}_test/test_conv1d.py",
    "conv2d": "benchmark/evaluation/{target}_test/test_conv2d.py",
    "conv2dnchw": "benchmark/evaluation/{target}_test/test_conv2d.py",
    "depthwiseconv": "benchmark/evaluation/{target}_test/test_depthwiseconv.py",
    "add": "benchmark/evaluation/{target}_test/test_add.py",
    "sign": "benchmark/evaluation/{target}_test/test_sign.py",
    "avgpool": "benchmark/evaluation/{target}_test/test_avgpool.py",
    "maxpool": "benchmark/evaluation/{target}_test/test_maxpool.py",
    "minpool": "benchmark/evaluation/{target}_test/test_minpool.py",
    "sumpool": "benchmark/evaluation/{target}_test/test_sumpool.py",
    "relu": "benchmark/evaluation/{target}_test/test_relu.py",
    "sigmoid": "benchmark/evaluation/{target}_test/test_sigmoid.py",
    "gelu": "benchmark/evaluation/{target}_test/test_gelu.py",
    "softmax": "benchmark/evaluation/{target}_test/test_softmax.py",
}


def run_test(file_name, test_file):
    """
    执行具体的测试脚本。
    
    参数:
    file_name: 待测试的源码文件路径（通常是生成的临时文件）。
    test_file: 测试驱动脚本路径（来自 test_file_map）。
    """
    try:
        # 使用 subprocess 启动新的 Python 进程运行测试脚本
        # --file 参数将待测文件路径传递给测试脚本
        output = subprocess.run(
            ["python", test_file, "--file", file_name],
            stdout=subprocess.PIPE,     # 捕获标准输出
            stderr=subprocess.STDOUT,   # 将标准错误重定向到标准输出
            encoding="utf-8",           # 输出编码格式
            check=True,                 # 如果进程返回非零退出码，抛出 CalledProcessError
            text=True,
            timeout=400,                # 设置超时时间为 400 秒
        )
        return True, output
    except subprocess.TimeoutExpired:
        # 如果测试运行超时，返回失败
        return False, "timeout"
    except subprocess.CalledProcessError as e:
        # 如果测试脚本报错（退出码非0），返回失败及错误输出
        return False, e.output


def unit_test(file_name, code):
    """
    单元测试主入口函数。
    
    参数:
    file_name: 原始文件名（用于推断算子类型）。
    code: 需要测试的代码字符串。
    """
    if code is None:
        return False

    # Create a temporary directory
    # 创建临时目录用于存放生成的代码文件
    tmp_dir = "tmps"
    os.makedirs(tmp_dir, exist_ok=True)

    # Remove the extension.
    # 移除原始文件的扩展名
    filename_no_ext, _ = os.path.splitext(file_name)
    # Determine the file type and set the target.
    # 根据代码内容判断目标平台（cuda/cpu/hip）和文件后缀
    target, file_type = get_target(code)
    # "Generate target file name"
    # 生成新的文件名
    filename = filename_no_ext + file_type
    # Extract the operation name and generate the test file path.
    # 从文件名中提取算子名称（假设文件名格式为 opname_xxx），用于查找对应的测试脚本
    op_name = os.path.basename(filename_no_ext).split("_")[0]

    if target == "cpu":
        # 如果是 CPU 代码，确保有 extern "C" 以便 ctypes 调用，避免 C++ 命名修饰问题
        code = 'extern "C" ' + code if "extern" not in code else code
    
    # 拼接临时文件的完整路径
    tmp_file_name = os.path.join(tmp_dir, os.path.basename(filename))
    # 将代码写入临时文件
    with open(tmp_file_name, mode="w") as f:
        f.write(code)
    
    # 针对 CPU 目标进行特殊处理，这里将其映射为 'dlboost'（可能是内部 CPU 编译/运行后端的名称）
    if target == "cpu":
        target = "dlboost"
    
    # 从映射表中获取对应的测试脚本路径，并填入 target
    test_file = test_file_map.get(op_name, "").format(target=target)

    # Run the test.
    # 执行测试
    success, output = run_test(tmp_file_name, test_file)
    logging.info(output)
    
    # 清理临时目录
    shutil.rmtree(tmp_dir)
    return success, output