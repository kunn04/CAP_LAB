// Ejemplo de medida de latencia de CAP
// Autor: Daniel Perdices
// © 2025

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <strings.h>

#define TAG_MEDIR_LATENCIA 33

int main(int argc, char *argv[])
{
    int num_procs, my_rank, other;
    char mach_name[MPI_MAX_PROCESSOR_NAME];
    int mach_len;
    MPI_Status status;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Get_processor_name(mach_name,&mach_len);
    
    // Imprimimos por stdout metadatos de quién es quién.
    // Esto es conveniente porque de cara a datos podemos filtrar estas líneas con un grep
    // pero de cara a identificar experimentos contienen información valiosa.
    printf("##%d\t%d\t%s\n", my_rank, num_procs, mach_name);
    // Si soy el maestro
    if (my_rank == 0) {
        // Cabecera datos + resolucion reloj
        printf("#Rango\tRTT(res=%.3le)\n", MPI_Wtick());
        // Para cada programa
        for (int i = 1; i < num_procs; i++) {
            double t0 = MPI_Wtime();
            // Le mandamos un byte con el TAG concreto
            MPI_Send(&my_rank, 1, MPI_INT, i, TAG_MEDIR_LATENCIA, MPI_COMM_WORLD);
            // Esperamos que lo reciba
            MPI_Recv(&other, 1, MPI_INT, i, TAG_MEDIR_LATENCIA, MPI_COMM_WORLD, &status);
            
            double tf = MPI_Wtime();

            printf("%d\t%lf\n", i, (tf-t0)*1000);
        }
    }
    else {
        // Recibimos del maestro
        MPI_Recv(&other, 1, MPI_INT, 0, TAG_MEDIR_LATENCIA, MPI_COMM_WORLD, &status);
        // Respondemos inmediatamente
        MPI_Send(&my_rank, 1, MPI_INT, 0, TAG_MEDIR_LATENCIA, MPI_COMM_WORLD);
    }
    MPI_Finalize();
}