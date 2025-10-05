#!/bin/bash
#SBATCH -J experimento_mpi
#SBATCH -o resultados/%x_%j.txt
#SBATCH --time=00:10:00

# ===========================================
# Script genérico para el ejercicio 6
# ===========================================

# Comprobamos argumentos
if [ $# -lt 1 ]; then
    echo "Uso: sbatch -n <num_procs> -w <lista_nodos> experimento_mpi.sbatch <latencia|ancho_banda> [tam_bloque]"
    exit 1
fi

PROGRAMA=$1
BLOCK_SIZE=${2:-1024}  # Por defecto 1024 bytes para ancho de banda

# Crear carpeta de resultados si no existe
mkdir -p resultados

# Compilar el programa correspondiente
if [ "$PROGRAMA" == "latencia" ]; then
    mpicc ejemplo_latencia.c -o latencia
    EXEC="./latencia"
elif [ "$PROGRAMA" == "ancho_banda" ]; then
    mpicc ejemplo_ancho_banda.c -o ancho_banda
    EXEC="./ancho_banda $BLOCK_SIZE"
else
    echo "Error: argumento inválido ($PROGRAMA)"
    exit 1
fi

# Nombre del archivo de salida
OUTFILE="resultados/${PROGRAMA}_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}.txt"

echo "Ejecutando experimento: $PROGRAMA"
echo "Guardando resultados en: $OUTFILE"

# Ejecutar con MPI (SLURM ya define nodos y número de procesos con -n y -w)
mpirun $EXEC > "$OUTFILE"
