#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "energy_meter.h" // biblioteca wrapper do PCM para Windows

#define STB_IMAGE_IMPLEMENTATION
#include "../../lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../lib/stb_image_write.h"

#define KERNEL_SIZE 31 // deve ser ímpar para ter um centro

void to_gray_float(int w, int h, int c, unsigned char *img, float *gray_img) {
    int i;
    #pragma omp parallel for schedule(static)
    for(i = 0; i < w * h; i++) {
        size_t idx = (size_t)i * c;
        unsigned char R = img[idx];
        unsigned char G = img[idx+1];
        unsigned char B = img[idx+2];
 
        float gray_val = (0.299f*R + 0.587f*G + 0.114f*B)/255.0f;
        gray_img[i] = gray_val;
    }
}
 
void convolve(const float *N, const float *F, float *P, int r, int width, int height) {
    int fSize = 2*r+1;
    int outRow;
    #pragma omp parallel for schedule(static)
    for(outRow = 0; outRow < height; outRow++) {
        for(int outCol = 0; outCol < width; outCol++) {
            float Pvalue = 0.0f;
 
            for(int fRow = 0; fRow < fSize; fRow++) {
                for(int fCol = 0; fCol < fSize; fCol++) {
                    int inRow = outRow - r + fRow;
                    int inCol = outCol - r + fCol;
 
                    // zero-padding nas bordas (pixels fora da imagem não contribuem)
                    if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                        Pvalue += F[fRow * fSize + fCol] * N[inRow * width + inCol];
                    }
                }
            }
            P[outRow*width + outCol] = Pvalue;
        }
    }
}
 
int main(int argc, char *argv[]){
    if(argc < 3) {
        fprintf(stderr, "Uso: %s <caminho_da_imagem> <num_threads>\n", argv[0]);
        return 1;
    }
    
    int num_threads = atoi(argv[2]);
    if(num_threads < 1) {
        fprintf(stderr, "Número de threads deve ser >= 1\n");
        return 1;
    }
    omp_set_num_threads(num_threads);
 
    if(init_energy_meter() != 0) {
        fprintf(stderr, "Falha ao inicializar o PCM. Garanta que está a correr como Administrador.\n");
        return 2;
    }
 
    int width, height, channels;
    
    unsigned char *img = stbi_load(argv[1], &width, &height, &channels, 0);
    if(img == NULL){
        fprintf(stderr, "Erro ao carregar imagem original\n");
        cleanup_energy_meter();
        return 1;
    }
 
    float *gray_img = (float*)malloc((size_t)width * height * sizeof(float));
    if(gray_img == NULL) {
        fprintf(stderr, "Erro ao alocar imagem cinza\n");
        stbi_image_free(img);
        cleanup_energy_meter();
        return 1;
    }
    to_gray_float(width, height, channels, img, gray_img);
 
    float *blur_img = (float*)calloc((size_t)width * height, sizeof(float));
    if(blur_img == NULL) {
        fprintf(stderr, "Erro ao alocar imagem de saída\n");
        stbi_image_free(img);
        free(gray_img);
        cleanup_energy_meter();
        return 1;
    }
 
    int radius = KERNEL_SIZE / 2;
    float *blur_kernel = (float*)malloc(KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    float weight = 1.0f / (KERNEL_SIZE * KERNEL_SIZE);
    for(int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; i++) blur_kernel[i] = weight;
 
    double start_time = omp_get_wtime();
    start_energy_measurement();
 
    convolve(gray_img, blur_kernel, blur_img, radius, width, height);
 
    EnergyReading energy = stop_energy_measurement();
    double end_time = omp_get_wtime();
 
    double time_spent = end_time - start_time;
    double joules = energy.cpu_joules;
 
    printf("%d,%.6f,%.6f\n", num_threads, time_spent, joules);
 
    //conversão float [0,1] para unsigned char
    unsigned char *final_output = (unsigned char*)malloc((size_t)width * height);
    for(size_t i = 0; i < (size_t)width * height; i++) {
        float val = blur_img[i] * 255.0f;
        if(val > 255.0f) val = 255.0f;
        if(val < 0.0f) val = 0.0f;
        final_output[i] = (unsigned char)val;
    }
 
    stbi_write_png("../../data/output_files/output_blur_omp.png", width, height, 1, final_output, width);
 
    free(img);
    free(gray_img);
    free(blur_img);
    free(blur_kernel);
    free(final_output);
    cleanup_energy_meter();
 
    return 0;
}