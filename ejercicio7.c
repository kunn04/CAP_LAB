#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

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
    long N; 

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 2) {
        if (rank == 0)
            printf("Uso: mpirun -np <nprocs> ./ejercicio7 <N>\n");
        MPI_Finalize();
        return 0;
    }

    N = atol(argv[1]);
    chunk = N / size;
    double *sub_array = (double *)malloc(chunk * sizeof(double));

    // ===============================
    // Sincronizar y empezar cronómetro
    // ===============================
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    if (rank == 0){
        array = (double*)malloc(N * sizeof(double));
        srand(42);
        for (long i = 0; i < N; i++) {
            array[i] = (double)rand() / RAND_MAX;
        }

        for (int i = 1; i < size; i++)
            MPI_Send(array + i * chunk, chunk, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);

        for (int i = 0; i < chunk; i++)
            sub_array[i] = array[i];
    } 
    else {
        MPI_Recv(sub_array, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Cálculo
    double local_sum = compute(sub_array, chunk);

    // Recolección
    if (rank != 0) {
        MPI_Send(&local_sum, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    } else {
        double total_sum = local_sum;
        double temp;
        for (int i = 1; i < size; i++) {
            MPI_Recv(&temp, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_sum += temp;
        }
    }

    // ===============================
    // Fin de la medición
    // ===============================
    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    double total_time = end - start;
    double max_time;

    MPI_Reduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("N=%ld\tProcesos=%d\tTiempoTotal=%f s\n", N, size, max_time);
    }

    free(sub_array);
    if (rank == 0) free(array);

    MPI_Finalize(); 
    return 0;
}
