#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gpu_energy.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../../lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../lib/stb_image_write.h"

#define KERNEL_SIZE 31
#define FILTER_RADIUS (KERNEL_SIZE / 2)

#define OUT_TILE_DIM 16 
//input tile maior para acomodar as os pixeis usados pelo kernel em todas as direções
#define IN_TILE_DIM (OUT_TILE_DIM + 2 * FILTER_RADIUS)

__constant__ float d_F[KERNEL_SIZE * KERNEL_SIZE];

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

__global__ void convolution_tiled(const float *N, float *P, int width, int height) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int outCol = blockIdx.x * OUT_TILE_DIM + tx;
    int outRow = blockIdx.y * OUT_TILE_DIM + ty;

    //alocação da memória partilhada com o tamanho do input_tile considerando os pixeis usados pelo kernel
    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];

    //threads do bloco colaboram para carregar as imagens de input (não há uma thread por pixel)
    int numInputElements = IN_TILE_DIM * IN_TILE_DIM;
    int threadId = ty * OUT_TILE_DIM + tx; //acessar thread no vetor de threads
    int totalThreads = OUT_TILE_DIM * OUT_TILE_DIM;

    //pontos iniciais do input na imagem original
    int startRow = blockIdx.y * OUT_TILE_DIM - FILTER_RADIUS;
    int startCol = blockIdx.x * OUT_TILE_DIM - FILTER_RADIUS;

    for (int i = threadId; i < numInputElements; i += totalThreads) { //varre de [0][0]:0-15 -> [0][1]:16-31
        //conversão do valor do vetor para a coordernada da matriz
        int rowIdx = i / IN_TILE_DIM;
        int colIdx = i % IN_TILE_DIM;

        //
        int globalRow = startRow + rowIdx;
        int globalCol = startCol + colIdx;

        if (globalRow >= 0 && globalRow < height && globalCol >= 0 && globalCol < width) {
            N_s[rowIdx][colIdx] = N[globalRow * width + globalCol];
        } else {
            N_s[rowIdx][colIdx] = 0.0f; //zero-padding nas bordas da imagem
        }
    }

    //sincronização para garantir que a imagem esteja 100% carregada na shared memory e as threads estejam disponíveis para uso
    __syncthreads();

    if (outCol < width && outRow < height) {
        float Pvalue = 0.0f;
        int fSize = 2 * FILTER_RADIUS + 1;
        
        for (int fRow = 0; fRow < fSize; fRow++) {
            for (int fCol = 0; fCol < fSize; fCol++) {
                Pvalue += d_F[fRow * fSize + fCol] * N_s[ty + fRow][tx + fCol];
            }
        }
        P[outRow * width + outCol] = Pvalue;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) return 1;
    const char *input_path = argv[1];
    int width, height, channels;

    if(init_gpu_energy_meter() != 0) return 2;

    unsigned char *img = stbi_load(input_path, &width, &height, &channels, 0);
    float *gray_img = (float*)malloc((size_t)width * height * sizeof(float));
    float *blur_img = (float*)calloc((size_t)width * height, sizeof(float)); 
    to_gray_float(width, height, channels, img, gray_img);

    float *blur_kernel = (float*)malloc(KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    float weight = 1.0f / (KERNEL_SIZE * KERNEL_SIZE); 
    for(int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; i++) blur_kernel[i] = weight;

    size_t img_bytes = (size_t)width * height * sizeof(float);
    size_t kernel_bytes = KERNEL_SIZE * KERNEL_SIZE * sizeof(float);

    float *d_N, *d_P;
    cudaMalloc((void**)&d_N, img_bytes);
    cudaMalloc((void**)&d_P, img_bytes);
    cudaMemcpy(d_N, gray_img, img_bytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_F, blur_kernel, kernel_bytes);

    dim3 blockDim(OUT_TILE_DIM, OUT_TILE_DIM); 
    dim3 gridDim((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    start_gpu_energy_measurement();
    cudaEventRecord(start);

    convolution_tiled<<<gridDim, blockDim>>>(d_N, d_P, width, height);
    
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    
    double joules = stop_gpu_energy_measurement();
    cudaEventSynchronize(stop); 

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Output para o script Python
    printf("CUDA_TILED,%f,%f\n", milliseconds / 1000.0f, joules);

    cudaMemcpy(blur_img, d_P, img_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_N); cudaFree(d_P);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    unsigned char *final_output = (unsigned char*)malloc((size_t)width * height);
    for(size_t i = 0; i < (size_t)width * height; i++) {
        float val = blur_img[i] * 255.0f;
        if(val > 255.0f) val = 255.0f;
        if(val < 0.0f) val = 0.0f;
        final_output[i] = (unsigned char)val;
    }

    stbi_write_png("../../data/output_files/output-blur-tiled.png", width, height, 1, final_output, width);

    free(img); free(gray_img); free(blur_img); free(blur_kernel); free(final_output);
    cleanup_gpu_energy_meter();
    return 0;
}