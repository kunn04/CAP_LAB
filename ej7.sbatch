#!/bin/bash
#SBATCH -J experimento_mpi
#SBATCH -o resultados/%x_%j.txt
#SBATCH --time=00:10:00

# ===========================================
# Script ejercicio 7
# ===========================================

# Comprobamos argumentos
if [ $# -lt 1 ]; then
    echo "Uso: sbatch -n <num_procs> [-w <nodos>] $0 <programa> [N]"
    echo "  <programa>: ejercicio7  "
    echo "  [N]: tamaño del problema (por defecto 1000000)"
    exit 1
fi

PROGRAMA=$1
N=${2:-1000000}

mkdir -p resultados

# Compilar el programa 
if [ "$PROGRAMA" == "ejercicio7" ]; then
    mpicc ejercicio7.c -o ejercicio7
    EXEC="./ejercicio7"
else
    echo "Error: programa no reconocido ($PROGRAMA)"
    exit 1
fi

# Nombre del archivo de salida
OUTFILE="resultados/${PROGRAMA}_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}.txt"

echo "Ejecutando experimento: $PROGRAMA con N=$N y P=${SLURM_NTASKS:-1} procesos"
echo "Guardando resultados en: $OUTFILE"

# Ejecutar el programa MPI con el número de tareas solicitadas
if command -v mpirun >/dev/null 2>&1; then
    mpirun -np "${SLURM_NTASKS:-1}" $EXEC "$N" > "$OUTFILE"
else
    srun -n "${SLURM_NTASKS:-1}" $EXEC "$N" > "$OUTFILE"
fi
