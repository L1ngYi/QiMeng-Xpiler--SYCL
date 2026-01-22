#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

// 引入 SYCL 命名空间
using namespace sycl;

// 实现 GEMM 的具体逻辑
void gemm_impl(half *A, half *B, float *C, int m, int k, int n, queue &q) {
    
    // 1. 在设备上分配内存
    half *d_A = malloc_device<half>(m * k, q);
    half *d_B = malloc_device<half>(k * n, q);
    float *d_C = malloc_device<float>(m * n, q);

    if (!d_A || !d_B || !d_C) {
        std::cerr << "[SYCL Error] Failed to allocate device memory" << std::endl;
        return;
    }

    // 2. 数据拷贝 Host -> Device
    // SYCL 会自动推断拷贝方向，只需要 3 个参数：(目标, 源, 大小)
    q.memcpy(d_A, A, m * k * sizeof(half));
    q.memcpy(d_B, B, k * n * sizeof(half));
    q.wait(); // 等待拷贝完成

    // 3. 设置并行计算的维度
    // Global Size: 矩阵的整体大小
    // Local Size:  局部工作组大小 (16, 16)
    range<2> global_size(m, n);
    range<2> local_size(16, 16);

    // 4. 提交计算任务
    q.submit([&](handler &h) {
        h.parallel_for(nd_range<2>(global_size, local_size), [=](nd_item<2> item) {
            int row = item.get_global_id(0);
            int col = item.get_global_id(1);

            // 边界检查
            if (row < m && col < n) {
                float sum = 0.0f;
                // 标准矩阵乘法累加
                for (int i = 0; i < k; ++i) {
                    // 将 half 转为 float 计算，模拟 Tensor Core 的 float 累加器行为
                    float val_a = static_cast<float>(d_A[row * k + i]);
                    float val_b = static_cast<float>(d_B[i * n + col]);
                    sum += val_a * val_b;
                }
                // 写入结果
                d_C[row * n + col] = sum;
            }
        });
    });
    
    q.wait(); // 等待计算完成

    // 5. 数据拷回 Device -> Host
    // [修复] 移除了原先错误的第4个参数
    q.memcpy(C, d_C, m * n * sizeof(float));
    q.wait();

    // 6. 释放内存
    free(d_A, q);
    free(d_B, q);
    free(d_C, q);
}

// 供 Python ctypes 调用的外部接口
extern "C" void gemm_kernel(half *A, half *B, float *C, int m, int k, int n) {
    try {
        // 创建 SYCL 队列
        // default_selector_v 会自动选择可用的最佳设备 (CPU 或 GPU)
        queue q(default_selector_v);
        
        // 第一次运行时打印设备信息，方便调试
        static bool printed = false;
        if (!printed) {
            std::cout << "[SYCL Unit Test] Running on: " 
                      << q.get_device().get_info<info::device::name>() << std::endl;
            printed = true;
        }

        gemm_impl(A, B, C, m, k, n, q);

    } catch (sycl::exception const &e) {
        std::cerr << "[SYCL Exception] " << e.what() << std::endl;
    }
}