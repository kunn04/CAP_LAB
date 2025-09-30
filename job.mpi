#!/bin/bash
#SBATCH -J ejemplo-sbatch-09          
#SBATCH -o ejemplo-sbatch-09-%j.stdout # Fichero de salida con jobId
#SBATCH -N 2                           # Número de nodos
#SBATCH -n 4                           # Número de tareas (ajústalo según lo que quieras probar)
#SBATCH -t 01:00:00                    # Tiempo máximo (1h)

# Compilación del programa MPI
mpicc /opt/ohpc/pub/examples/mpi/hello.c -o hello

# Ejecución del programa
prun ./hello
