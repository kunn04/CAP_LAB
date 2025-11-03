#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


/* Funcion de kernel*/

__global__ void greyScale(uint8_t *d_grey_image, uint8_t *d_rgb_image, int width, int height){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if(i< width && j> height){
        int idx = (j*width + i) * 4;
        int r = d_rgb_image[idx];
        int g = d_rgb_image[idx+1];
        int b = d_rgb_image[idx+2];
        d_grey_image[j*width+i] = (uint8_t)(0.2989 * r + 0.5870 * g + 0.1140 * b);
    }

}

int main(int nargs, char **argv)
{
    int width, height, nchannels;
    struct timeval fin, ini;
    uint8_t *d_grey_image, *d_rgb_image;

    if (nargs < 2)
    {
        printf("Usage: %s <image1> [<image2> ...]\n", argv[0]);
    }
    // For each image
    // Bucle 0
    for (int file_i = 1; file_i < nargs; file_i++)
    {
        printf("[info] Processing %s\n", argv[file_i]);
        /****** Reading file ******/
        uint8_t *rgb_image = stbi_load(argv[file_i], &width, &height, &nchannels, 4);
        if (!rgb_image)
        {
            perror("Image could not be opened");
        }

        /****** Allocating memory ******/
        // - RGB2Grey
        uint8_t *grey_image = (uint8_t *)malloc(width * height);
        if (!grey_image)
        {
            perror("Could not allocate memory");
        }

        cudaMalloc((void **)&d_grey_image, width * height * sizeof(uint8_t));
        cudaMalloc((void **)&d_rgb_image, width * height * sizeof(uint8_t) * 4);
        cudaMemcpy(d_rgb_image, rgb_image, width * height * sizeof(uint8_t) * 4, cudaMemcpyHostToDevice);

        // - Filenames
        for (int i = strlen(argv[file_i]) - 1; i >= 0; i--)
        {
            if (argv[file_i][i] == '.')
            {
                argv[file_i][i] = 0;
                break;
            }
        }

        char *grey_image_filename = 0;
        asprintf(&grey_image_filename, "%s_grey.jpg", argv[file_i]);
        if (!grey_image_filename)
        {
            perror("Could not allocate memory");
            exit(-1);
        }

        /****** Computations ******/
        printf("[info] %s: width=%d, height=%d, nchannels=%d\n", argv[file_i], width, height, nchannels);

        if (nchannels != 3 && nchannels != 4)
        {
            printf("[error] Num of channels=%d not supported. Only three (RGB), four (RGBA) are supported.\n", nchannels);
            continue;
        }



        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

        gettimeofday(&ini, NULL);
        // RGB to grey scale
        int r, g, b;
        greyScale <<<gridSize, blockSize>>> (d_grey_image, d_rgb_image, width, height);
        cudaDeviceSynchronize();
        cudaMemcpy(grey_image, d_grey_image, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost);

        gettimeofday(&fin, NULL);
        stbi_write_jpg(grey_image_filename, width, height, 1, grey_image, 10);
        free(rgb_image);

        FILE *pf = fopen("time.txt", "a");
        fprintf(pf, "%s %f\n", argv[file_i], ((fin.tv_sec * 1000000 + fin.tv_usec) - (ini.tv_sec * 1000000 + ini.tv_usec)) * 1.0 / 1000000.0);

        cudaFree(d_grey_image);
        cudaFree(d_rgb_image);
        free(grey_image_filename);
        free(grey_image);
        fclose(pf);
    }
}
