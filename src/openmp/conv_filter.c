#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../../lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../lib/stb_image_write.h"

#define KERNEL_SIZE 31 //deve ser ímpar para ter um centro

void to_gray(int w, int h, int c, unsigned char *img, unsigned char *gray_img) {
    size_t i;
#pragma omp parallel for
    for(i=0; i<(size_t)w*h; i++){
        size_t index = i*c;
        unsigned char R = img[index];
        unsigned char G = img[index+1];
        unsigned char B = img[index+2];

        gray_img[i] = 0.299*R+0.587*G+0.114*B; //https://www.mathworks.com/help/matlab/ref/rgb2gray.html
    }
}

//https://c-lang-learning.com/en/image-processing.html
void convolve(int w, int h, const unsigned char *img_in, unsigned char *img_out, int ksize, const double *k)
{
    int half = ksize/2;
    //loop externo: convolução para obter os pixels da imagem de saída
#pragma omp parallel for schedule(dynamic)
    for(int y = 0; y < h; y++){
        for(int x = 0; x < w; x++){
            double sum = 0.0;
            //loop interno: varredura do kernel
            for(int ky = 0; ky < ksize; ky++){
                for(int kx = 0; kx < ksize; kx++){
                    //(nx, ny): coordenada do pixel da imagem original que será multiplicada com o (kx,ky) do kernel
                    int ny = y + ky - half;
                    int nx = x + kx - half;

                    //clamp nas bordas para manter dimensão de saída
                    if(ny < 0) ny = 0;
                    if(ny >= h) ny = h - 1;
                    if(nx < 0) nx = 0;
                    if(nx >= w) nx = w - 1;

                    sum += img_in[ny*w + nx]*k[ky*ksize + kx]; //pixel_imagem x pixel_kernel 
                }
            }
            if(sum < 0) sum = 0;
            if(sum > 255) sum = 255;
            img_out[y*w + x] = (unsigned char)sum; //y*w + x -> imagem armazenada em vetor construído pela concatenação das linhas da imagem
        }
    }
}

int main(){
    int width, height, channels;
    double start, end;

    unsigned char *img = stbi_load("../../data/bird-4k.png", &width, &height, &channels, 0);
    if(img == NULL){
        fprintf(stderr, "Erro ao alocar imagem-fonte"); 
        exit(1);
    }
    printf("Imagem de dimensão (%d, %d, %d) carregada\n", width, height, channels);

    unsigned char *gray_img = malloc((size_t)width*height);
    if(gray_img == NULL){
        fprintf(stderr, "Erro ao alocar imagem cinza");
        exit(1);
    }

    start = omp_get_wtime();

    to_gray(width, height, channels, img, gray_img);
    //stbi_write_png("../../data/output_files/bird-4k-gray.png", width, height, 1, gray_img, width);

    end = omp_get_wtime();
    printf("toGray: %f seconds.\n", end - start);

    unsigned char *blur_img = malloc((size_t)width*height);
    if(blur_img == NULL) {
        fprintf(stderr, "Erro ao alocar simagem Sobel\n");
        exit(1);
    }

    double blur_kernel[KERNEL_SIZE][KERNEL_SIZE];
    double weigth = 1.0/(KERNEL_SIZE*KERNEL_SIZE); 

    for(int i = 0; i < KERNEL_SIZE; i++) {
        for(int j = 0; j < KERNEL_SIZE; j++) {
            blur_kernel[i][j] = weigth; //https://docs.opencv.org/4.13.0/d4/d13/tutorial_py_filtering.html
        }
    }

    start = omp_get_wtime();

    convolve(width, height, gray_img, blur_img, KERNEL_SIZE, &blur_kernel[0][0]);

    end = omp_get_wtime();
    printf("Convolve: %f seconds.\n", end - start);

    stbi_write_png("../../data/output_files/bird-4k-blur.png", width, height, 1, blur_img, width);

    return 0;
}