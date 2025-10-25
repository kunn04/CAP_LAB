#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

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
        srand(time(NULL)); 
        for (int i = 0; i < N; i++) {
            array[i] = (double)rand() / RAND_MAX;
        }
        if (N % size != 0) {
            fprintf(stderr, "[Aviso] N=%d no es divisible por size=%d. Se ignorarán %d elementos.\n", N, size, N % size);
        }
        printf("Datos generados. Distribuyendo entre %d procesos...\n", size);
    }

    

    MPI_Scatter(array, chunk, MPI_DOUBLE, sub_array, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    local_sum = compute(sub_array, chunk);

    MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double *partial_sums = NULL;
    if (rank == 0) partial_sums = (double*)malloc(size * sizeof(double));
    MPI_Gather(&local_sum, 1, MPI_DOUBLE, partial_sums, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double total_sum_gather = 0.0;
        for (int i = 0; i < size; i++) total_sum_gather += partial_sums[i];
        printf("Suma total de cuadrados (Reduce): %lf\n", total_sum);
        printf("Suma total de cuadrados (Gather+sum): %lf\n", total_sum_gather);
    }

    free(sub_array);
    if (rank == 0) {
        free(array);
        free(partial_sums);
    }

    MPI_Finalize();
    return 0;
}