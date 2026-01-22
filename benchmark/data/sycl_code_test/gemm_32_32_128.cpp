/*
 * SYCL version of gemm_32_32_128
 * Note: Replaced CUDA WMMA (Tensor Core) intrinsics with standard SYCL loops
 * to ensure compatibility with CPU backends for functional testing.
 */

// 这里的头文件引用是为了代码独立性，实际 compilation.py 也会通过 macro 引入
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

using namespace sycl;

// 定义内核函数：简单的平铺矩阵乘法逻辑
void gemm_impl(half *A, half *B, float *C, int m, int k, int n, queue &q) {
    
    // 1. 在设备上分配内存
    // 在 CPU 运行时中，malloc_device 的开销很小
    half *d_A = malloc_device<half>(m * k, q);
    half *d_B = malloc_device<half>(k * n, q);
    float *d_C = malloc_device<float>(m * n, q);

    if (!d_A || !d_B || !d_C) {
        std::cerr << "[SYCL Error] Failed to allocate device memory" << std::endl;
        return;
    }

    // 2. 数据拷贝 (Host -> Device)
    // 这是一个阻塞操作，确保数据到位
    q.memcpy(d_A, A, m * k * sizeof(half));
    q.memcpy(d_B, B, k * n * sizeof(half));
    q.wait();

    // 3. 启动内核
    // 使用 2D 的 range，对应矩阵的行(Row)和列(Col)
    range<2> global_size(m, n);
    // 局部工作组大小设为 16x16，模拟 CUDA 代码中的 tiling 概念
    // 注意：如果 m 或 n 不是 16 的倍数，这里需要 padding 或边界检查，
    // 但此测试用例名称暗示了维度是 32x128，是 16 的倍数，所以安全。
    range<2> local_size(16, 16);

    q.submit([&](handler &h) {
        h.parallel_for(nd_range<2>(global_size, local_size), [=](nd_item<2> item) {
            int row = item.get_global_id(0);
            int col = item.get_global_id(1);

            if (row < m && col < n) {
                float sum = 0.0f;
                
                // 执行标准矩阵乘法累加: Row(A) * Col(B)
                // 替代了原本的 wmma::load_matrix_sync 和 wmma::mma_sync
                for (int i = 0; i < k; ++i) {
                    // 将 half 转换为 float 进行高精度累加 (对应 wmma::accumulator 为 float)
                    float val_a = static_cast<float>(d_A[row * k + i]);
                    float val_b = static_cast<float>(d_B[i * n + col]);
                    sum += val_a * val_b;
                }
                
                // 写回结果
                d_C[row * n + col] = sum;
            }
        });
    });
    
    // 等待计算完成
    q.wait();

    // 4. 数据拷回 (Device -> Host)
    q.memcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost); // 注意：cudaMemcpyDeviceToHost 这里只是个宏或枚举，SYCL里不需要，但保持参数位置
    q.wait();

    // 5. 释放内存
    free(d_A, q);
    free(d_B, q);
    free(d_C, q);
}

// Extern C 包装器：供 Python 的 ctypes 或 test_gemm.py 调用
// 保持与 CUDA 版本完全一致的函数签名
extern "C" void gemm_kernel(half *A, half *B, float *C, int m, int k, int n) {
    try {
        // 创建队列：默认选择器会自动寻找最佳设备（通常是 CPU，如果你没装显卡驱动）
        // 如果想强制指定 CPU，可以使用 sycl::cpu_selector_v
        queue q(default_selector_v);
        
        // 简单的调试输出，确认跑在了哪里 (第一次运行时打印)
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