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
    int chunk;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    chunk = N / size;

    double *sub_array = (double *)malloc(chunk * sizeof(double));

    if (rank == 0){
        array = (double*)malloc(N * sizeof(double));
        srand(time(NULL));
        for (int i = 0; i < N; i++) {
            array[i] = (double)rand() / RAND_MAX;
        }

        for (int i = 1; i < size; i++)
            MPI_Send(array + i * chunk, chunk, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        
        for (int i = 0; i < chunk; i++)
            sub_array[i] = array[i];
        
    } else {
        MPI_Recv(sub_array, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }


    double local_sum = compute(sub_array, chunk);

    if (rank != 0) {
        MPI_Send(&local_sum, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);

    } else {
        double total_sum = local_sum;
        double temp;
        for (int i = 1; i < size; i++) {
            MPI_Recv(&temp, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_sum += temp;
        }
        printf("Suma total de cuadrados: %lf\n", total_sum);
    }

    free(sub_array);
    if (rank == 0) free(array);

    MPI_Finalize(); 
    return 0;

}