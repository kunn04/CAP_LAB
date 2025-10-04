#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 1000000

double compute(double *data, int len) {
    double sum = 0.0;
    for (int i = 0; i < len; i++) {
        sum += data[i] * data[i];
    }
    return sum;
}

int main(int argc, char *argv[]) {
    int size, rank;
    double *array = NULL;
    double *sub_array;
    int chunk;
    double local_sum, total_sum;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    chunk = N / size;

    sub_array = (double *)malloc(chunk * sizeof(double));

    if (rank == 0) {
        array = (double*)malloc(N * sizeof(double));
        srand(42); // Semilla para reproducibilidad
        for (int i = 0; i < N; i++) {
            array[i] = (double)rand() / RAND_MAX;
        }
        printf("Datos generados. Distribuyendo entre %d procesos...\n", size);
    }

    MPI_Scatter(array, chunk, MPI_DOUBLE, sub_array, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    local_sum = compute(sub_array, chunk);

    MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Suma total de cuadrados: %lf\n", total_sum);
    }

    free(sub_array);
    if (rank == 0) {
        free(array);
    }

    MPI_Finalize();
    return 0;
}