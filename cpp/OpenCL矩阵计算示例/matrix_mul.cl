__kernel void matrix_mul(__global const float* A,
                         __global const float* B,
                         __global float* C,
                         const int N,
                         const int M,
                         const int K) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[k * M + j];
    }

    C[i * M + j] = sum;
}
