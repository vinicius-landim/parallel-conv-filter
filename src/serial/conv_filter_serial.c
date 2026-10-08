#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "energy_meter.h" 

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define KERNEL_SIZE 31 // deve ser ímpar para ter um centro

void to_gray(int w, int h, int c, unsigned char *img, unsigned char *gray_img) {
    for(size_t i=0, j=0; i<(size_t)w*h*c; i+=c, j++){
        unsigned char R = img[i];
        unsigned char G = img[i+1];
        unsigned char B = img[i+2];

        gray_img[j] = 0.299*R+0.587*G+0.114*B;
    }
}

void convolve(int w, int h, const unsigned char *img_in, unsigned char *img_out, int ksize, const double *k)
{
    int half = ksize/2;
    // loop externo de convolução para obter os pixels da imagem de saída
    for(int y = 0; y < h; y++){
        for(int x = 0; x < w; x++){
            double sum = 0.0;
            // loop interno para varredura do kernel
            for(int ky = 0; ky < ksize; ky++){
                for(int kx = 0; kx < ksize; kx++){
                    int ny = y + ky - half;
                    int nx = x + kx - half;

                    // clamp nas bordas para manter a dimensão de saída
                    if(ny < 0) ny = 0;
                    if(ny >= h) ny = h - 1;
                    if(nx < 0) nx = 0;
                    if(nx >= w) nx = w - 1;

                    sum += img_in[ny*w + nx]*k[ky*ksize + kx];
                }
            }
            if(sum < 0) sum = 0;
            if(sum > 255) sum = 255;
            img_out[y*w + x] = (unsigned char)sum; 
        }
    }
}

int main(int argc, char *argv[]){
    if(argc < 2) {
        fprintf(stderr, "Uso: %s <caminho_da_imagem>\n", argv[0]);
        return 1;
    }

    int width, height, channels;
    double start, end;

    if(init_energy_meter() != 0) {
        fprintf(stderr, "Falha ao inicializar o PCM. Garanta que esta a correr como Administrador.\n");
        exit(2);
    }

    unsigned char *img = stbi_load(argv[1], &width, &height, &channels, 0);
    if(img == NULL){
        fprintf(stderr, "Erro ao carregar imagem original");
        exit(1);
    }

    unsigned char *gray_img = malloc((size_t)width*height);
    if(gray_img == NULL){
        fprintf(stderr, "Erro ao alocar imagem cinza");
        exit(1);
    }
    to_gray(width, height, channels, img, gray_img);

    unsigned char *blur_img = malloc((size_t)width*height);
    if(blur_img == NULL) {
        fprintf(stderr, "Erro ao alocar imagem de saída\n");
        exit(1);
    }

    double blur_kernel[KERNEL_SIZE][KERNEL_SIZE];
    double weigth = 1.0/(KERNEL_SIZE*KERNEL_SIZE);

    for(int i = 0; i < KERNEL_SIZE; i++) {
        for(int j = 0; j < KERNEL_SIZE; j++) {
            blur_kernel[i][j] = weigth;
        }
    }

    start = omp_get_wtime();
    start_energy_measurement();

    convolve(width, height, gray_img, blur_img, KERNEL_SIZE, &blur_kernel[0][0]);

    EnergyReading energy = stop_energy_measurement();
    end = omp_get_wtime();

    double time_spent = end - start;
    double joules = energy.cpu_joules;

    printf("SERIAL,%.6f,%.6f\n", time_spent, joules);

    // Guarda o output relativo à pasta src/
    stbi_write_png("../data/output_files/output_blur_serial.png", width, height, 1, blur_img, width);

    free(img);
    free(gray_img);
    free(blur_img);
    cleanup_energy_meter();

    return 0;
}