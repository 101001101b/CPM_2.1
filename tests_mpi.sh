#!/bin/bash
export LC_NUMERIC=C

# Compilació del teu codi C
mpicc -O3 kmean_p.c -o kmean_p

SEQ_JUTGES=26.2
ROW_FMT="%-20s | %-17s | %-15s | %-20s |\n"

echo "========================================================================="
echo " PRUEBAS MPI EN MATRICES JUTGES (Tiempo seq base: 26.2s)"
echo " OBJETIVO: Media armónica > 4.0 | Max Speedup > 15 | Min Speedup > 1.75"
echo "========================================================================="
sum_inv_S=0
count=0

printf "$ROW_FMT" "Nodes / PPN (Procs)" "Time (s)" "Speedup" "Iterations"

CONFIGS=(
    "2,1" "4,1" "8,1"
    "16,1" "16,2" "16,4"
    "16,8" "16,16" "16,32"
)

for config in "${CONFIGS[@]}"; do
    IFS=',' read -r nodes ppn <<< "$config"
    procs=$((nodes * ppn))
    
    if [ "$ppn" -gt "$nodes" ]; then
        OVERCOMMIT_FLAG="--overcommit"
    else
        OVERCOMMIT_FLAG=""
    fi
    
    # Executem la comanda neta de SLURM (sense el time de bash)
    OUTPUT=$(salloc -p jutjat -N $nodes --exclusive srun -N $nodes -n $procs --ntasks-per-node=$ppn --distribution=block:block $OVERCOMMIT_FLAG ./kmean_p 2>&1)
    
    # Extraiem la variable de temps que imprimeix el teu codi C
    TIME_PAR=$(echo "$OUTPUT" | grep "TIEMPO:" | awk '{print $2}')
    ITER=$(echo "$OUTPUT" | grep -oE 'iter [0-9]+' | head -n 1)
    
    if [ -z "$TIME_PAR" ]; then
        TIME_PAR=999.0
        SPEEDUP="0.0"
        ITER="ERROR"
    else
        # Càlcul de l'Speedup individual
        SPEEDUP=$(awk -v seq=$SEQ_JUTGES -v par=$TIME_PAR 'BEGIN { printf "%.5f", seq/par }')
        
        # Acumulem l'invers (1 / Speedup) per a la Mitjana Harmònica
        sum_inv_S=$(awk -v sum=$sum_inv_S -v seq=$SEQ_JUTGES -v par=$TIME_PAR 'BEGIN { print sum + (par/seq) }')
    fi
    
    printf "$ROW_FMT" "$nodes / $ppn ($procs)" "${TIME_PAR} s" "${SPEEDUP}" "$ITER"
    ((count++))
done

# Càlcul final de la Mitjana Harmònica d'Speedup per a la P2.1
HMEAN=$(awk -v n=$count -v sum_inv=$sum_inv_S 'BEGIN { printf "%.5f", n/sum_inv }')

echo "-------------------------------------------------------------------------"
echo "-> MEDIA ARMÓNICA DE SPEEDUP OBTENIDA EN JUTGES: $HMEAN"
echo "========================================================================="

rm kmean_p