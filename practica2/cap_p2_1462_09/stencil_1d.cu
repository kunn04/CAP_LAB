#include <iostream>
#include <stdlib.h>
#include <sys/time.h> // Para gettimeofday()
#include <stdio.h>
#include <string.h> // Para memset

// --- Constantes ---
// Radio del stencil, como en las diapositivas
#define RADIUS 3 
// Hilos por bloque (BLOCK_SIZE)
#define BLOCK_SIZE 256 

/**
 * @brief Calcula el tiempo transcurrido en milisegundos.
 */
double get_time_ms(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
}

/**
 * @brief Implementación de Stencil 1D en CPU.
 * Versión secuencial simple para comparar rendimiento.
 */
void stencil_1d_cpu(int *in, int *out, int N) {
    for (int i = 0; i < N; i++) {
        int result = 0;
        for (int offset = -RADIUS; offset <= RADIUS; offset++) {
            int index = i + offset;
            // Comprobación de límites
            if (index >= 0 && index < N) {
                result += in[index];
            }
        }
        out[i] = result;
    }
}

/**
 * @brief Kernel CUDA para Stencil 1D optimizado.
 *
 * Utiliza memoria compartida para reducir accesos a memoria global.
 * Este kernel corrige los bugs críticos de las diapositivas.
 */
__global__ void stencil_1d_gpu(int *in, int *out, int N) {
    
    // Declaración de memoria compartida
    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];

    // Índices global y local
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + RADIUS;

    // 1. Cargar datos de memoria global a compartida
    
    // Cargar elemento central (con chequeo de límites)
    if (gindex < N) {
        temp[lindex] = in[gindex];
    } else {
        temp[lindex] = 0; // Padding si gindex > N
    }

    // Cargar halos (solo los primeros/últimos RADIUS hilos)
    if (threadIdx.x < RADIUS) {
        // Cargar halo izquierdo
        int left_gindex = gindex - RADIUS;
        // CORRECCIÓN BUG: Chequear 'gindex < RADIUS'
        if (left_gindex < 0) {
            temp[lindex - RADIUS] = 0;
        } else {
            temp[lindex - RADIUS] = in[left_gindex];
        }

        // Cargar halo derecho
        int right_gindex = gindex + blockDim.x; // (blockDim.x es el BLOCK_SIZE)
        // CORRECCIÓN BUG: Chequear 'gindex + BLOCK_SIZE > N-1'
        if (right_gindex >= N) {
            temp[lindex + blockDim.x] = 0;
        } else {
            temp[lindex + blockDim.x] = in[right_gindex];
        }
    }

    // Sincronizar hilos del bloque
    // Asegura que toda la memoria compartida esté cargada antes de calcular.
    __syncthreads();

    // 2. Aplicar el stencil desde la memoria compartida
    if (gindex < N) {
        int result = 0;
        for (int offset = -RADIUS; offset <= RADIUS; offset++) {
            result += temp[lindex + offset];
        }
        out[gindex] = result;
    }
}


/**
 * @brief Función principal: Orquesta la comparación CPU vs GPU.
 */
int main() {
    struct timeval start, end;
    
    printf("Comparativa Stencil 1D (RADIUS=%d, BLOCK_SIZE=%d)\n", RADIUS, BLOCK_SIZE);
    // Imprimimos un formato fácil de parsear por Python
    printf("%-10s | %-15s | %-15s | %-10s\n", "N", "CPU (ms)", "GPU (ms)", "Speedup");
    printf("------------------------------------------------------------------\n");

    // Bucle para N de 100.000 a 1.000.000
    for (int N = 100000; N <= 1000000; N += 100000) {
        size_t size = N * sizeof(int);

        // 1. Asignar memoria (Host y Device)
        int *h_in, *h_out_cpu, *h_out_gpu;
        int *d_in, *d_out;
        
        h_in = (int*)malloc(size);
        h_out_cpu = (int*)malloc(size);
        h_out_gpu = (int*)malloc(size);
        
        // Inicializar datos
        for (int i = 0; i < N; i++) {
            h_in[i] = i % 100; // Datos predecibles
        }
        memset(h_out_cpu, 0, size);
        memset(h_out_gpu, 0, size);


        cudaMalloc((void**)&d_in, size); 
        cudaMalloc((void**)&d_out, size); 

        // --- 2. Ejecución CPU ---
        gettimeofday(&start, NULL);
        stencil_1d_cpu(h_in, h_out_cpu, N);
        gettimeofday(&end, NULL);
        double cpu_time_ms = get_time_ms(start, end);

        
        // --- 3. Ejecución GPU ---
        // La medición debe incluir la copia de datos
        gettimeofday(&start, NULL);

        // Copiar datos H -> D
        cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

        // Configurar grid y bloques
        int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        // Lanzar kernel
        stencil_1d_gpu<<<grid_size, BLOCK_SIZE>>>(d_in, d_out, N);

        // Copiar resultados D -> H
        cudaMemcpy(h_out_gpu, d_out, size, cudaMemcpyDeviceToHost);
        
        // Sincronizar para asegurar que todo ha terminado
        cudaDeviceSynchronize(); 
        
        gettimeofday(&end, NULL);
        double gpu_time_ms = get_time_ms(start, end);

        
        // --- 4. Verificación (Opcional pero recomendada) ---
        bool error = false;
        for (int i = 0; i < N; i++) {
            if (h_out_cpu[i] != h_out_gpu[i]) {
                printf("Error en N=%d, índice %d: CPU=%d, GPU=%d\n", N, i, h_out_cpu[i], h_out_gpu[i]);
                error = true;
                break;
            }
        }
        if (error) continue; // Si hay error, no imprimir esta línea

        // --- 5. Resultados ---
        double speedup = cpu_time_ms / gpu_time_ms;
        printf("%-10d | %-15.4f | %-15.4f | %-10.2fx\n", N, cpu_time_ms, gpu_time_ms, speedup);

        // Liberar memoria
        free(h_in); free(h_out_cpu); free(h_out_gpu);
        cudaFree(d_in); cudaFree(d_out); 
    }

    return 0;
}