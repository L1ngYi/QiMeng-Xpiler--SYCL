#include <sycl/sycl.hpp>
#include <vector>

using namespace sycl;

// 这是你的 Kernel 实现
// 它接收的 A, B, C 已经是 Device 端的指针了（由外层包装器分配）
// q 也是由外层传入的
void gemm(half *A, half *B, float *C, int m, int k, int n, queue &q) {
    
    // 设置并行计算维度
    range<2> global_size(m, n);
    range<2> local_size(16, 16);

    // 提交任务
    q.submit([&](handler &h) {
        h.parallel_for(nd_range<2>(global_size, local_size), [=](nd_item<2> item) {
            int row = item.get_global_id(0);
            int col = item.get_global_id(1);

            if (row < m && col < n) {
                float sum = 0.0f;
                for (int i = 0; i < k; ++i) {
                    // 模拟 Tensor Core 行为：half 输入 -> float 累加
                    float val_a = static_cast<float>(A[row * k + i]);
                    float val_b = static_cast<float>(B[i * n + col]);
                    sum += val_a * val_b;
                }
                C[row * n + col] = sum;
            }
        });
    });
    // 注意：这里不需要 q.wait()，外层包装器会在 memcpy 回去之前 wait
}