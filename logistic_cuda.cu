#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <time.h>

#define DEFAULT_NUM_SAMPLES    1000000
#define DEFAULT_NUM_FEATURES   64
#define DEFAULT_MAX_ITERATIONS 200
#define DEFAULT_LEARNING_RATE  0.01f

#define BLOCK_SIZE 256
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

typedef struct {
    cudaEvent_t start;
    cudaEvent_t stop;
} Timer;

void timer_create(Timer *t) {
    CUDA_CHECK(cudaEventCreate(&t->start));
    CUDA_CHECK(cudaEventCreate(&t->stop));
}

void timer_destroy(Timer *t) {
    CUDA_CHECK(cudaEventDestroy(t->start));
    CUDA_CHECK(cudaEventDestroy(t->stop));
}

void timer_start(Timer *t) {
    CUDA_CHECK(cudaEventRecord(t->start, 0));
}

double timer_stop(Timer *t) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventRecord(t->stop, 0));
    CUDA_CHECK(cudaEventSynchronize(t->stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t->start, t->stop));
    return ms / 1000.0;
}


float sigmoid_cpu(float z) {
    if (z >= 0.0f) {
        return 1.0f / (1.0f + expf(-z));
    } else {
        float ez = expf(z);
        return ez / (1.0f + ez);
    }
}


void generate_data(float *X, int *y, int N, int D) {
    srand(42);

    float *true_weights = (float *)malloc(D * sizeof(float));
    for (int d = 0; d < D; d++)
        true_weights[d] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

    float true_bias = 0.5f;

    for (int i = 0; i < N; i++) {
        float z = true_bias;
        for (int d = 0; d < D; d++) {
            float val = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            X[i * D + d] = val;
            z += true_weights[d] * val;
        }

        float prob = sigmoid_cpu(z);
        float noise = ((float)rand() / RAND_MAX);
        y[i] = (noise < prob) ? 1 : 0;
    }

    free(true_weights);
}

float cpu_train_step(const float *X, const int *y, float *weights,
                     float *bias, int N, int D, float lr) {
    float *gradients = (float *)calloc(D, sizeof(float));
    float grad_bias = 0.0f;
    float total_loss = 0.0f;

    for (int i = 0; i < N; i++) {
        float z = *bias;
        for (int d = 0; d < D; d++) {
            z += X[i * D + d] * weights[d];
        }

        float pred = sigmoid_cpu(z);
        float pred_clamped = fmaxf(fminf(pred, 1.0f - 1e-7f), 1e-7f);

        total_loss += -(y[i] * logf(pred_clamped) +
                        (1 - y[i]) * logf(1.0f - pred_clamped));

        float error = pred - (float)y[i];
        for (int d = 0; d < D; d++) {
            gradients[d] += error * X[i * D + d];
        }
        grad_bias += error;
    }

    for (int d = 0; d < D; d++) {
        weights[d] -= lr * (gradients[d] / N);
    }
    *bias -= lr * (grad_bias / N);

    free(gradients);
    return total_loss / N;
}

float cpu_predict_accuracy(const float *X, const int *y, const float *weights,
                           float bias, int N, int D) {
    int correct = 0;
    for (int i = 0; i < N; i++) {
        float z = bias;
        for (int d = 0; d < D; d++)
            z += X[i * D + d] * weights[d];
        int pred = (sigmoid_cpu(z) >= 0.5f) ? 1 : 0;
        if (pred == y[i]) correct++;
    }
    return 100.0f * correct / N;
}

void cpu_logistic_regression(const float *X, const int *y, float *weights,
                             float *bias, int N, int D, int max_iter, float lr) {
    for (int iter = 0; iter < max_iter; iter++) {
        float loss = cpu_train_step(X, y, weights, bias, N, D, lr);
        if ((iter + 1) % 50 == 0 || iter == 0)
            printf("    Iter %4d | Loss: %.6f\n", iter + 1, loss);
    }
}

__device__ float sigmoid_gpu(float z) {
    if (z >= 0.0f) {
        return 1.0f / (1.0f + expf(-z));
    } else {
        float ez = expf(z);
        return ez / (1.0f + ez);
    }
}


__global__ void kernel_forward_pass(const float *X, const int *y,
                                     const float *weights, float bias,
                                     float *errors, float *losses,
                                     int N, int D) {

    extern __shared__ float shared_weights[];
    for (int idx = threadIdx.x; idx < D; idx += blockDim.x) {
        shared_weights[idx] = weights[idx];
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < N; i += stride) {
        float z = bias;
        for (int d = 0; d < D; d++) {
            z += X[i * D + d] * shared_weights[d];
        }

        float pred = sigmoid_gpu(z);

        errors[i] = pred - (float)y[i];

        float p = fmaxf(fminf(pred, 1.0f - 1e-7f), 1e-7f);
        losses[i] = -(y[i] * logf(p) + (1 - y[i]) * logf(1.0f - p));
    }
}


__global__ void kernel_compute_gradients(const float *X, const float *errors,
                                          float *gradients, float *grad_bias,
                                          int N, int D) {
    extern __shared__ float sdata[];

    int feature = blockIdx.x;  
    int tid = threadIdx.x;

    float local_sum = 0.0f;
    float local_bias_sum = 0.0f;

    for (int i = tid; i < N; i += blockDim.x) {
        float err = errors[i];
        if (feature < D) {
            local_sum += err * X[i * D + feature];
        }
        if (feature == 0) {
            local_bias_sum += err;
        }
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0 && feature < D) {
        gradients[feature] = sdata[0];
    }

    if (feature == 0) {
        sdata[tid] = local_bias_sum;
        __syncthreads();
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        if (tid == 0) {
            *grad_bias = sdata[0];
        }
    }
}

__global__ void kernel_update_weights(float *weights, float *bias,
                                       const float *gradients,
                                       const float *grad_bias,
                                       float lr, int N, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < D) {
        weights[d] -= lr * (gradients[d] / N);
    }
    if (d == 0) {
        *bias -= lr * (*grad_bias / N);
    }
}


__global__ void kernel_reduce_loss(const float *losses, float *total_loss, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < N) ? losses[i] : 0.0f;

    for (int idx = i + blockDim.x * gridDim.x; idx < N; idx += blockDim.x * gridDim.x) {
        sdata[tid] += losses[idx];
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(total_loss, sdata[0]);
}


void gpu_logistic_regression(const float *h_X, const int *h_y,
                             float *h_weights, float *h_bias,
                             int N, int D, int max_iter, float lr) {

    size_t X_size = (size_t)N * D * sizeof(float);
    size_t y_size = (size_t)N * sizeof(int);
    size_t w_size = D * sizeof(float);

    float *d_X, *d_weights, *d_errors, *d_losses;
    float *d_gradients, *d_grad_bias, *d_total_loss, *d_bias;
    int *d_y;

    CUDA_CHECK(cudaMalloc(&d_X, X_size));
    CUDA_CHECK(cudaMalloc(&d_y, y_size));
    CUDA_CHECK(cudaMalloc(&d_weights, w_size));
    CUDA_CHECK(cudaMalloc(&d_bias, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_errors, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_losses, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gradients, w_size));
    CUDA_CHECK(cudaMalloc(&d_grad_bias, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_total_loss, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_X, h_X, X_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, y_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, w_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias, sizeof(float), cudaMemcpyHostToDevice));

    int forward_blocks = min((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 1024);
    int loss_blocks = min((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 1024);
    size_t shared_weights = D * sizeof(float);
    size_t shared_reduce = BLOCK_SIZE * sizeof(float);

    float h_loss;

    for (int iter = 0; iter < max_iter; iter++) {
        kernel_forward_pass<<<forward_blocks, BLOCK_SIZE, shared_weights>>>(
            d_X, d_y, d_weights, *h_bias, d_errors, d_losses, N, D);

        kernel_compute_gradients<<<D, BLOCK_SIZE, shared_reduce>>>(
            d_X, d_errors, d_gradients, d_grad_bias, N, D);

        int update_blocks = (D + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernel_update_weights<<<update_blocks, BLOCK_SIZE>>>(
            d_weights, d_bias, d_gradients, d_grad_bias, lr, N, D);

        CUDA_CHECK(cudaMemcpy(h_bias, d_bias, sizeof(float), cudaMemcpyDeviceToHost));

        if ((iter + 1) % 50 == 0 || iter == 0) {
            CUDA_CHECK(cudaMemset(d_total_loss, 0, sizeof(float)));
            kernel_reduce_loss<<<loss_blocks, BLOCK_SIZE, shared_reduce>>>(
                d_losses, d_total_loss, N);
            CUDA_CHECK(cudaMemcpy(&h_loss, d_total_loss, sizeof(float),
                                  cudaMemcpyDeviceToHost));
            printf("    Iter %4d | Loss: %.6f\n", iter + 1, h_loss / N);
        }
    }

    CUDA_CHECK(cudaMemcpy(h_weights, d_weights, w_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_errors));
    CUDA_CHECK(cudaFree(d_losses));
    CUDA_CHECK(cudaFree(d_gradients));
    CUDA_CHECK(cudaFree(d_grad_bias));
    CUDA_CHECK(cudaFree(d_total_loss));
}


void print_gpu_info() {
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("GPU Information\n");
    printf("Device:              %s\n", prop.name);
    printf("Compute Capability:  %d.%d\n", prop.major, prop.minor);
    printf("SM Count:            %d\n", prop.multiProcessorCount);
    printf("Global Memory:       %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("Shared Mem per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Max Threads/Block:   %d\n", prop.maxThreadsPerBlock);
    printf("Warp Size:           %d\n", prop.warpSize);
}

int main(int argc, char **argv) {
    int N        = (argc > 1) ? atoi(argv[1]) : DEFAULT_NUM_SAMPLES;
    int D        = (argc > 2) ? atoi(argv[2]) : DEFAULT_NUM_FEATURES;
    int max_iter = (argc > 3) ? atoi(argv[3]) : DEFAULT_MAX_ITERATIONS;
    float lr     = (argc > 4) ? atof(argv[4]) : DEFAULT_LEARNING_RATE;

    printf("Logistic Regression: CUDA Parallel Gradient Descent\n");
    printf("====================================================\n");
    printf("Samples:       %d\n", N);
    printf("Features:      %d\n", D);
    printf("Max Iters:     %d\n", max_iter);
    printf("Learning Rate: %.4f\n\n", lr);

    print_gpu_info();

    size_t X_bytes = (size_t)N * D * sizeof(float);
    float *X             = (float *)malloc(X_bytes);
    int   *y             = (int *)malloc(N * sizeof(int));
    float *cpu_weights   = (float *)calloc(D, sizeof(float));
    float *gpu_weights   = (float *)calloc(D, sizeof(float));
    float cpu_bias = 0.0f, gpu_bias = 0.0f;

    if (!X || !y || !cpu_weights || !gpu_weights) {
        fprintf(stderr, "Host memory allocation failed!\n");
        return EXIT_FAILURE;
    }

    printf("Generating synthetic dataset...\n\n");
    generate_data(X, y, N, D);

    int pos = 0;
    for (int i = 0; i < N; i++) pos += y[i];
    printf("Class distribution: %d positive (%.1f%%), %d negative (%.1f%%)\n\n",
           pos, 100.0f * pos / N, N - pos, 100.0f * (N - pos) / N);

    printf("--- CPU Training (Sequential) ---\n");
    Timer timer;
    timer_create(&timer);
    timer_start(&timer);
    cpu_logistic_regression(X, y, cpu_weights, &cpu_bias, N, D, max_iter, lr);
    double cpu_time = timer_stop(&timer);
    float cpu_acc = cpu_predict_accuracy(X, y, cpu_weights, cpu_bias, N, D);
    printf("  CPU Time:     %.4f s\n", cpu_time);
    printf("  CPU Accuracy: %.2f%%\n\n", cpu_acc);

    printf("--- GPU Training (CUDA Parallel) ---\n");
    cudaFree(0);  

    timer_start(&timer);
    gpu_logistic_regression(X, y, gpu_weights, &gpu_bias, N, D, max_iter, lr);
    double gpu_time = timer_stop(&timer);
    float gpu_acc = cpu_predict_accuracy(X, y, gpu_weights, gpu_bias, N, D);
    printf("  GPU Time:     %.4f s\n", gpu_time);
    printf("  GPU Accuracy: %.2f%%\n\n", gpu_acc);

    printf("=== Performance Comparison ===\n");
    printf("CPU Time:    %.4f s\n", cpu_time);
    printf("GPU Time:    %.4f s\n", gpu_time);
    printf("Speedup:     %.2fx\n", cpu_time / gpu_time);
    printf("CPU Accuracy: %.2f%%\n", cpu_acc);
    printf("GPU Accuracy: %.2f%%\n\n", gpu_acc);

    printf("=== Scalability Benchmark ===\n");
    printf("%-12s %-12s %-12s %-10s\n", "Samples", "CPU (s)", "GPU (s)", "Speedup");

    int test_sizes[] = {10000, 50000, 100000, 500000, 1000000};
    int num_tests = 5;
    int bench_iters = 50;  
    for (int t = 0; t < num_tests; t++) {
        int n = test_sizes[t];
        if (n > N) break;

        memset(cpu_weights, 0, D * sizeof(float));
        memset(gpu_weights, 0, D * sizeof(float));
        cpu_bias = 0.0f;
        gpu_bias = 0.0f;

        timer_start(&timer);
        cpu_logistic_regression(X, y, cpu_weights, &cpu_bias, n, D, bench_iters, lr);
        double ct = timer_stop(&timer);

        memset(gpu_weights, 0, D * sizeof(float));
        gpu_bias = 0.0f;

        timer_start(&timer);
        gpu_logistic_regression(X, y, gpu_weights, &gpu_bias, n, D, bench_iters, lr);
        double gt = timer_stop(&timer);

        printf("%-12d %-12.4f %-12.4f %-10.2f\n", n, ct, gt, ct / gt);
    }

    printf("\n=== Weight Comparison (first 5 features) ===\n");
    printf("%-10s %-15s %-15s\n", "Feature", "CPU Weight", "GPU Weight");
    for (int d = 0; d < 5 && d < D; d++) {
        printf("w[%d]       %-15.6f %-15.6f\n", d, cpu_weights[d], gpu_weights[d]);
    }
    printf("Bias:      %-15.6f %-15.6f\n", cpu_bias, gpu_bias);

    timer_destroy(&timer);
    free(X);
    free(y);
    free(cpu_weights);
    free(gpu_weights);

    printf("\nDone.\n");
    return EXIT_SUCCESS;
}
