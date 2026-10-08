#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "gpu_energy.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../../lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../lib/stb_image_write.h"

#define KERNEL_SIZE 31

void to_gray_float(int w, int h, int c, unsigned char *img, float *gray_img) {
    for(size_t i = 0; i < (size_t)w * h; i++) {
        size_t idx = i * c;
        unsigned char R = img[idx];
        unsigned char G = img[idx+1];
        unsigned char B = img[idx+2];
        
        float gray_val = (0.299f*R + 0.587f*G + 0.114f*B)/255.0f;
        gray_img[i] = gray_val;
    }
}

__global__ void convolve(const float *N, const float *F, float *P, int r, int width, int height) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int fSize = 2*r+1;
    float Pvalue = 0.0f;

    for(int fRow = 0; fRow < fSize; fRow++) {
        for(int fCol = 0; fCol < fSize; fCol++) {
            int inRow = outRow-r + fRow;
            int inCol = outCol-r + fCol;
            
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                Pvalue += F[fRow * fSize + fCol] * N[inRow * width + inCol];
            }
        }
    }
    P[outRow*width + outCol] = Pvalue;
}

int main(int argc, char *argv[]) {
    if (argc < 2) return 1;
    const char *input_path = argv[1];
    int width, height, channels;

    if(init_gpu_energy_meter() != 0) {
        fprintf(stderr, "Erro ao inicializar NVML.\n");
        return 2;
    }

    unsigned char *img = stbi_load(input_path, &width, &height, &channels, 0);
    float *gray_img = (float*)malloc((size_t)width * height * sizeof(float));
    float *blur_img = (float*)calloc((size_t)width * height, sizeof(float)); 
    to_gray_float(width, height, channels, img, gray_img);

    int radius = KERNEL_SIZE / 2;
    float *blur_kernel = (float*)malloc(KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    float weight = 1.0f / (KERNEL_SIZE * KERNEL_SIZE); 
    for(int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; i++) blur_kernel[i] = weight;

    size_t img_bytes = (size_t)width * height * sizeof(float);
    size_t kernel_bytes = KERNEL_SIZE * KERNEL_SIZE * sizeof(float);

    float *d_N, *d_F, *d_P;
    cudaMalloc((void**)&d_N, img_bytes);
    cudaMalloc((void**)&d_F, kernel_bytes);
    cudaMalloc((void**)&d_P, img_bytes);
    cudaMemcpy(d_N, gray_img, img_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, blur_kernel, kernel_bytes, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16); 
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    start_gpu_energy_measurement();
    cudaEventRecord(start);

    convolve<<<gridDim, blockDim>>>(d_N, d_F, d_P, radius, width, height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    double joules = stop_gpu_energy_measurement();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Output para o script Python
    printf("CUDA_GLOBAL,%f,%f\n", milliseconds / 1000.0f, joules);

    cudaMemcpy(blur_img, d_P, img_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_N); cudaFree(d_F); cudaFree(d_P);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    unsigned char *final_output = (unsigned char*)malloc((size_t)width * height);
    for(size_t i = 0; i < (size_t)width * height; i++) {
        float val = blur_img[i] * 255.0f;
        if(val > 255.0f) val = 255.0f;
        if(val < 0.0f) val = 0.0f;
        final_output[i] = (unsigned char)val;
    }

    // Caminho ajustado para execução a partir de src/
    stbi_write_png("../../data/output_files/output-blur-global.png", width, height, 1, final_output, width);

    free(img); free(gray_img); free(blur_img); free(blur_kernel); free(final_output);
    cleanup_gpu_energy_meter();
    return 0;
}