#!/bin/bash
#SBATCH -J experimento_mpi
#SBATCH -o resultados/%x_%j.txt
#SBATCH --time=00:10:00

# ===========================================
# Script genérico para el ejercicio 6
# ===========================================

if [ $# -lt 1 ]; then
    echo "Uso: sbatch -n <num_procs> -w <lista_nodos> experimento_mpi.sbatch <latencia|ancho_banda> [tam_bloque]"
    exit 1
fi

PROGRAMA=$1
BLOCK_SIZE=${2:-1024} 

mkdir -p resultados

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

OUTFILE="resultados/${PROGRAMA}_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}.txt"

echo "Ejecutando experimento: $PROGRAMA"
echo "Guardando resultados en: $OUTFILE"

mpirun $EXEC > "$OUTFILE"
