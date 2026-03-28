from falcon.mcts.actions import actions as ActionSpace


def _is_sycl_cached(code):
    return "local_accessor<" in code


def _is_sycl_tensorized(code):
    return "sycl::mad(" in code and (
        "[[sycl::reqd_sub_group_size(" in code
        or "reqd_sub_group_size(" in code
        or "joint_matrix" in code
        or "sub_group" in code
    )


def generate_prior_from_src(code, src_target, dst_target):
    """根据源代码中出现的特定关键词，为各个转换 pass 分配优先级。

    参数:
      code: 字符串，源代码内容。
      src_target: 源平台类型。
      dst_target: 目标平台类型。

    返回:
      logit_prior: 包含 (action, priority) 元组的列表，其中 priority 为 "high" 或 "default"。
    """
    logit_prior = [0.2] * len(ActionSpace)

    if src_target == "cuda" and "thread" in code:
        logit_prior[0] = 0.5

    if src_target == "hip" and "thread" in code:
        logit_prior[0] = 0.5

    if src_target == "cuda" and "mma_sync" in code:
        logit_prior[2] = 0.5

    if src_target == "hip" and "amdgcn" in code:
        logit_prior[2] = 0.5

    if src_target == "cpu" and "dpbusd" in code:
        logit_prior[2] = 0.5

    if dst_target == "cuda" or dst_target == "hip" and "thread" not in code:
        logit_prior[7] = 0.4

    if dst_target == "sycl" and "parallel_for" not in code and "q.submit" not in code:
        logit_prior[7] = 0.4
    elif dst_target == "sycl":
        sycl_cached = _is_sycl_cached(code)
        sycl_tensorized = _is_sycl_tensorized(code)

        # Once cache has already been introduced, bias the search slightly
        # toward tensorization to encourage 8 -> 9 transitions.
        if sycl_cached and not sycl_tensorized:
            logit_prior[9] = max(logit_prior[9], 0.3)

        # If both optimizations are already present, keep them equally likely
        # to avoid creating a synthetic preference for repeated 8 or 9 moves.
        if sycl_cached and sycl_tensorized:
            shared_prior = max(logit_prior[8], logit_prior[9])
            logit_prior[8] = shared_prior
            logit_prior[9] = shared_prior

    return logit_prior
