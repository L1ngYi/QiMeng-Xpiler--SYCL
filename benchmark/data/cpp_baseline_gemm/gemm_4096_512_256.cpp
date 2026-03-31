extern "C" void gemm(float *A, float *B, float *C, int m, int k, int n) {
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (int inner = 0; inner < k; ++inner) {
        sum += A[row * k + inner] * B[inner * n + col];
      }
      C[row * n + col] = sum;
    }
  }
}
