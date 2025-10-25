// Ejemplo de medida de ancho de banda de CAP
// Autor: Daniel Perdices
// © 2025

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <strings.h>

#define TAG_MEDIR_BW 33
#define MB 1048576

void print_usage(int argc, char** argv) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
        printf("Uso %s <block_size>\n", argv[0]);
}

int main(int argc, char *argv[])
{
    int num_procs, my_rank, other;
    char mach_name[MPI_MAX_PROCESSOR_NAME];
    int mach_len;
    char buffer_env[2*MB];
    char buffer_rec[2*MB];
    ssize_t block_size = 0;

    MPI_Status status;
    fclose(stderr);
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Get_processor_name(mach_name,&mach_len);
    
    if (argc != 2) {
        print_usage(argc, argv);
        return -1;
    }

    if (sscanf(argv[1], "%d", &block_size) != 1) {
        print_usage(argc, argv);
        return -1;
    }

    // Imprimimos por stdout metadatos de quién es quién.
    // Esto es conveniente porque de cara a datos podemos filtrar estas líneas con un grep
    // pero de cara a identificar experimentos contienen información valiosa.
    printf("##%d\t%d\t%s\n", my_rank, num_procs, mach_name);

    // Si soy el maestro
    if (my_rank == 0) {
        // Cabecera datos + resolucion reloj
        printf("#Rango\tTime\tBW\n", MPI_Wtick());
        
        // Para cada programa
        for (int i = 1; i < num_procs; i++) {
            double t0 = MPI_Wtime();
            ssize_t contador = 0;
            // Mandamos hasta que se haya enviado 1 MB
            while(contador < 1*MB) {
                MPI_Send(buffer_env, block_size, MPI_CHAR, i, TAG_MEDIR_BW, MPI_COMM_WORLD);
                contador += block_size;
            }
            // Esperamos confirmación
            MPI_Recv(buffer_rec, 1, MPI_INT, i, TAG_MEDIR_BW, MPI_COMM_WORLD, &status);

            double tf = MPI_Wtime();

            printf("%d\t%lf\t%lf\n", i, (tf-t0)*1000, 1*MB/(tf-t0));
        }
    }
    else {
        ssize_t contador = 0;
        // Recibimos 1MB
        while(contador < 1*MB) {
            MPI_Recv(buffer_rec, block_size, MPI_CHAR, 0, TAG_MEDIR_BW, MPI_COMM_WORLD, &status);
            contador += block_size;
        }
        MPI_Send(&my_rank, 1, MPI_INT, 0, TAG_MEDIR_BW, MPI_COMM_WORLD);
    }
    MPI_Finalize();
}