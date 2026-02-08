import logging
import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import mctx  # DeepMind 的蒙特卡洛树搜索 (MCTS) 库，用于强化学习
import tiktoken  # OpenAI 的分词器，用于将代码转换为向量表示
from absl import app, flags
from jax import jit, lax, vmap

# 导入自定义的性能测试模块，用于在不同硬件(CUDA, CPU, HIP)上跑分
from benchmark.perf import perf_cuda, perf_dlboost, perf_hip
# 导入先验概率生成模块，用于引导 MCTS 搜索
from falcon.mcts.action_logit import generate_prior_from_src
# 导入动作空间，定义了所有可能的代码变换操作
from falcon.mcts.actions import actions as ActionSpace
# 导入无效动作检测逻辑
from falcon.mcts.invalid_actions import get_invalid_actions
from falcon.mcts.utils import open_file
from falcon.util import get_target

# Configure the log
logging.basicConfig(
    level=logging.INFO,  # Set the log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Set the log format.
)

# TODO(michael): replace with shape calculation
# 计算 GFLOPS 的基准常数，用于归一化奖励值 (Reward)
# 64 * 1280 * 2 大概对应某种矩阵运算的浮点操作数
GFLOPS = 64 * 1280 * 2 / 1e9
A_Length = len(ActionSpace)  # 动作空间的大小


# 初始化 GPT-4o 的编码器，用于将源代码文本转化为数值 Embedding
encoder = tiktoken.encoding_for_model("gpt-4o")
FLAGS = flags.FLAGS
# 定义命令行参数
flags.DEFINE_integer("seed", 42, "Random seed.")  # 随机种子
flags.DEFINE_integer("num_simulations", 512, "Number of simulations.")  # MCTS 每次搜索的模拟次数
flags.DEFINE_integer(
    "max_num_considered_actions",
    13,
    "The maximum number of actions expanded at the root.",  # 根节点最大扩展动作数
)
flags.DEFINE_integer("max_depth", 13, "The maximum search depth.")  # 搜索树的最大深度
flags.DEFINE_string("source", "cpu", "Source platform identifier.")  # 源平台
flags.DEFINE_string("target", "cuda", "Destination platform identifier.")  # 目标平台
flags.DEFINE_string(
    "file_name",
    "benchmark/data/cpp_code_test/bmm_4_128_128_128.cpp",
    "Path to the input kernel file.",  # 待优化的 C++ 内核文件路径
)

# --- 关键配置 ---
# 禁用 JAX 的 JIT 编译。
# 因为代码中的 step 函数涉及文件 IO、调用外部编译器和运行 benchmark，
# 这些都是副作用(Side Effects)，在纯 JIT 模式下无法正常工作。
jax.config.update("jax_disable_jit", True)
jax.config.update("jax_enable_x64", True)  # 启用 64 位精度
jax.disable_jit()  # 再次显式禁用 JIT
BS = 1  # Batch Size (批大小)


def objective(file_name, target):
    """We design an objective function.

    If compile and runtime error happens, then the score is zero.
    """
    # 目标函数：计算代码的性能得分
    try:
        time_ms = 1000000
        # 根据目标平台调用对应的性能测试工具，获取运行时间 (毫秒)
        if target == "cuda":
            time_ms = perf_cuda.benchmark(file_name)
        elif target == "cpu":
            time_ms = perf_dlboost.benchmark(file_name)
        elif target == "hip":
            time_ms = perf_hip.benchmark(file_name)
        
        # 计算性能得分：GFLOPS / 时间 (数值越大越好)
        return GFLOPS / (time_ms / 1e3)
    except Exception as e:
        # 如果编译或运行失败，记录日志并返回 0 分
        logging.info(e)
        return 0.0


class FalconGo:
    """
    FalconGo 环境类。
    模拟强化学习环境，负责维护当前代码状态、执行动作(代码变换)并返回奖励。
    """
    def __init__(
        self,
        file_name,
        op_name,
        source_platform,
        target_platform,
        action_len=A_Length,
        optimizer_len=A_Length,
        timeout=None,
    ):
        self.timeout = timeout
        self.file_name = file_name
        self.op_name = op_name
        self.source_platform = source_platform
        self.target_platform = target_platform
        self.action_len = action_len
        self.optimizer_len = optimizer_len
        self.best_reward = 0.0001 # 记录历史最佳奖励
        self.best_optimizer_ids = None
        self.iteration = 0
        self.best_actions = None
        # 输出目录：基于源平台和目标平台命名
        self.output_dir = os.path.join(
            f"{self.source_platform}_{self.target_platform}"
        )
        # Ensure the directory exists.
        os.makedirs(self.output_dir, exist_ok=True)

    def perform_action(self, actions):
        """Applies a sequence of scheduling actions to the original code.

        This function:
        1. Loads the original source code
        2. Applies each scheduling action in sequence
        3. Compiles the transformed code
        4. Evaluates its performance

        The function returns both the transformed code and a performance score.
        A score of 0 indicates compilation failure.

        Args:
            actions: List of scheduling actions to apply sequentially

        Returns:
            tuple: (transformed_code, performance_score)
            - transformed_code (str): Source code after applying all actions
            - performance_score (float): Performance metric (0 if compilation fails)
        """
        # 1. 读取原始代码
        code = open_file(self.file_name)
        # 如果是 GPU 平台，移除 extern 声明（特定于具体实现的清理逻辑）
        code = (
            code.split("extern")[0]
            if self.source_platform in ["cuda", "hip"]
            else code
        )
        # 2. 依次应用动作序列中的每一个变换
        for action in actions:
            code = action(
                self.file_name,
                code,
                self.source_platform,
                self.target_platform,
            )
        
        # 3. 准备编译和测试
        target, file_type = get_target(code, self.target_platform)
        os.makedirs("tmp", exist_ok=True)
        # 提取文件名并生成临时文件路径
        base_name = os.path.basename(self.file_name)
        name_no_ext, _ = os.path.splitext(base_name)
        new_file = os.path.join("tmp", name_no_ext + file_type)
        
        # 将变换后的代码写入临时文件
        with open(new_file, "w", encoding="utf-8") as f:
            f.write(code)
            
        # 4. 运行 benchmark 获取分数
        score = objective(new_file, target)
        if target != self.target_platform:
            score = 0
        return code, score

    @jit
    def step(self, action_id, env_state):
        """
        环境的一步交互 (Step Function)。
        由于外层禁用了 JIT，这里虽然有 @jit 装饰器，但实际是以 Eager Mode 运行的，
        允许包含文件 IO 和打印等副作用。
        """
        self.iteration += 1
        # 解包环境状态：Embedding，历史轨迹，当前深度，奖励记录
        embedding_state, trajectory, depth, rewards = env_state
        # 更新轨迹：在当前深度记录采取的动作 ID
        trajectory = trajectory.at[depth].set(action_id)
        
        # 获取当前路径上的所有动作 ID
        cur_action_ids = lax.dynamic_slice(
            trajectory, (0,), (depth.val[0] + 1,)
        )
        # 将 JAX 数组转换为 Python 列表，以便后续处理
        cur_action_list = jax.device_get(cur_action_ids.val[0]).tolist()
        # 将动作 ID 映射回实际的函数对象
        cur_actions = [ActionSpace[_i] for _i in cur_action_list]

        try:
            # 执行动作序列，获取转换后的代码和奖励
            code, reward = self.perform_action(cur_actions)
            
            # 如果发现了更好的奖励，保存结果
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_actions = cur_action_list
                # save the file into success transcompiled folder
                # Extract base name and replace extension
                base_name = os.path.basename(self.file_name)
                name_no_ext, _ = os.path.splitext(base_name)
                target, file_type = get_target(code, self.target_platform)
                new_file = os.path.join(
                    self.output_dir, name_no_ext + file_type
                )
                # 保存最佳代码到输出目录
                with open(new_file, "w", encoding="utf-8") as f:
                    f.write(code)
        except Exception:
            # 异常处理：代码生成或编译失败，给予巨大的负奖励
            code = ""
            reward = -10000
            print(f"Invalid action: {cur_action_ids.val[0].tolist()}")

        # 打印当前步的搜索信息
        print(
            f"Step: {self.iteration}\t"
            f"Action: {cur_action_ids.val[0].tolist()}\t"
            f"Reward: {reward:.4f}\t"
            f"Best Reward: {self.best_reward:.4f}\t"
            f"Best action: {self.best_actions}\t",
            flush=True,
        )

        # 更新奖励历史（这里似乎有一个特定的均值更新逻辑）
        for depth_index, var in enumerate(cur_action_list):
            new_value = (rewards[0, depth_index, var] + reward) / 2
            rewards = rewards.at[0, depth_index, var].set(new_value)

        # Treminated if we reach the goal or the reward is zero
        # 终止条件：达到最大优化深度 或者 遇到严重错误(-10000)
        condition1 = depth > self.optimizer_len
        condition2 = reward == -10000

        terminal = jax.lax.cond(
            condition1,
            lambda _: True,  # If condition1 is True
            lambda _: condition2,  # If condition1 is False, return condition2
            operand=None,
        )
        # 准备下一个状态
        next_env_state = (
            embedding_state,
            trajectory,
            depth + 1,
            rewards,
        )

        # 返回元组：(新状态, 观测(代码embedding), 奖励, 是否终止, Info)
        return (
            next_env_state,
            encoder.encode(code),
            self.best_reward,
            terminal,
            None,
        )

    @partial(jit, static_argnums=(0,))
    def reset(self, key):
        """重置环境状态"""
        code = open_file(self.file_name)
        code = (
            code.split("extern")[0]
            if self.source_platform in ["cuda", "hip"]
            else code
        )
        # 初始状态：原始代码的 Embedding
        embedding_state = jnp.array(encoder.encode(code))
        trajectory = jnp.zeros(self.optimizer_len, dtype=int)
        depth = 0
        rewards = jnp.zeros(
            (1, self.optimizer_len, self.num_actions), dtype=jnp.float32
        )
        return embedding_state, trajectory, depth, rewards

    @partial(jit, static_argnums=(0,))
    def get_observation(self, env_state):
        # 从环境状态中提取观测值
        optimize_grid, trajectory, depth = env_state
        return optimize_grid

    @property
    def num_actions(self):
        return self.action_len


def build_env(file_name, source_platform="cpu", target_platform="cuda"):
    """构建 FalconGo 环境的工厂函数"""
    action_len = len(ActionSpace)
    base_name = os.path.basename(file_name)
    op_name = base_name.split("_")[0] #根据文件名提取操作名称，例如 gemm_32_32_128.cu -> gemm
    optimizer_len = 14 # 硬编码的最大优化步数
    tvm_env = FalconGo(
        file_name,
        op_name,
        source_platform,
        target_platform,
        action_len=action_len,
        optimizer_len=optimizer_len,
    )
    return tvm_env


def get_recurrent_fn(env):
    """
    构造传给 mctx 的循环函数 (Recurrent Function)。
    定义了搜索树中节点如何转移：状态 + 动作 -> 下一个状态 + 奖励。
    """
    batch_step = vmap(env.step) # 向量化 step 函数以支持 batch

    def recurrent_fn(params, key, actions, env_state):
        key, subkey = jax.random.split(key)
        # 执行环境的一步，获取新状态和奖励
        new_env_state, obs, max_reward, terminals, _ = batch_step(
            actions, env_state
        )
        embedding_state, trajectory, depth, rewards = new_env_state
        
        # 更新轨迹信息
        trajectory = trajectory.at[depth].set(actions)
        depth_val = int(jax.device_get(depth)[0])
        cur_action_ids = lax.dynamic_slice(trajectory, (0, 0), (1, depth_val))
        jax.device_get(cur_action_ids)[0].tolist()

        # 将 Embedding 解码回代码字符串，用于后续的有效性检查和先验计算
        code_embedding = [int(arr) for arr in embedding_state[0]]
        code = encoder.decode(code_embedding)
        
        # 计算当前代码状态下的无效动作 Mask (防止搜索无效路径)
        invalid_mask = jnp.array(
            get_invalid_actions(code, env.source_platform, env.target_platform)
        ).reshape(1, -1)
        
        # 获取奖励
        reward = rewards[0, 0, depth - 1, actions]

        # 计算动作的先验概率 (Prior Logits)
        # 这有助于引导树搜索优先探索更有希望的分支
        prior_logits = jnp.array(
            generate_prior_from_src(
                code, env.source_platform, env.target_platform
            )
        ).reshape(1, -1)

        # 返回 mctx 需要的 RecurrentFnOutput 对象
        return (
            mctx.RecurrentFnOutput(
                reward=reward,
                discount=jnp.where(terminals, 0, 1).astype(jnp.float32), # 终止状态 discount 为 0
                prior_logits=prior_logits,
                value=reward, # 这里简单地将 value 设为 reward，通常应该由 Value Network 预测
            ),
            new_env_state,
        )

    return recurrent_fn


def _run_demo(env, rng_key):
    """Runs a search algorithm on a toy environment."""
    """运行 MCTS 搜索演示"""
    batch_reset = vmap(env.reset)

    key, subkey = jax.random.split(rng_key)
    subkeys = jax.random.split(subkey, num=BS)

    # 初始化环境状态
    states_init = batch_reset(subkeys)
    key, logits_rng = jax.random.split(key)
    rng_key, logits_rng, q_rng, search_rng = jax.random.split(key, 4)
    
    # 获取初始代码并进行必要清理
    code = open_file(env.file_name)
    code = (
        code.split("extern")[0]
        if env.source_platform in ["cuda", "hip"]
        else code
    )
    # 计算初始状态的无效动作 Mask
    invalid_actions = jnp.array(
        get_invalid_actions(code, env.source_platform, env.target_platform)
    ).reshape(1, -1)
    
    # prior_logits = jnp.ones((1, A_Length)) / A_Length
    # 计算初始状态的动作先验概率
    prior_logits = jnp.array(
        generate_prior_from_src(code, env.source_platform, env.target_platform)
    ).reshape(1, -1)
    
    # 定义搜索树的根节点
    root = mctx.RootFnOutput(
        prior_logits=prior_logits,  # jnp.full([batch_size, num_actions],
        value=jnp.zeros([BS]), # 初始价值设为 0
        # The embedding will hold the state index.
        embedding=states_init, # 嵌入状态即环境状态
    )

    recurrent_fn = get_recurrent_fn(env)
    # Running the search.
    # 调用 mctx 的 Gumbel MuZero 策略进行搜索
    # 这是一个结合了 MCTS 和学习模型的强大的规划算法
    policy_output = mctx.gumbel_muzero_policy(
        params=states_init,
        rng_key=search_rng,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=FLAGS.num_simulations, # 模拟次数
        invalid_actions=invalid_actions, # 屏蔽无效动作
        max_depth=env.optimizer_len, # 最大搜索深度
        max_num_considered_actions=FLAGS.max_num_considered_actions, # 根节点最大动作数
    )
    return policy_output


def main(argv):
    rng_key = jax.random.PRNGKey(FLAGS.seed)
    # 构建环境
    falcon_env = build_env(FLAGS.file_name, FLAGS.source, FLAGS.target)

    start_time = time.time()
    # 执行搜索
    policy_output = _run_demo(falcon_env, rng_key)
    batch_index = 0
    # 获取搜索结果中选定的最佳动作
    selected_action = policy_output.action[batch_index]
    # 获取对应的 Q 值
    policy_output.search_tree.summary().qvalues[batch_index, selected_action]
    # To estimate the value of the root state, use the Q-value of the selected
    # action. The Q-value is not affected by the exploration at the root
    # node.
    # 要估计根状态的值，请使用所选动作的 Q 值。Q 值不受根节点探索的影响。
    end_time = time.time()
    print(f"[INFO]searching time: {end_time - start_time} s")


if __name__ == "__main__":
    app.run(main)