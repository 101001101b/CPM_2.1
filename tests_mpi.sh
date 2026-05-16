#!/bin/bash
export LC_NUMERIC=C

# Compilació de la versió MPI amb màxima optimització
mpicc -O3 kmean_p.c -o kmean_p

# Temps seqüencial base a les màquines JUTGES extret de l'enunciat
SEQ_JUTGES=26.2

# Format de les columnes per al printf
ROW_FMT="%-20s | %-17s | %-15s | %-20s |\n"

echo "========================================================================="
echo " PRUEBAS MPI EN MATRICES JUTGES (Tiempo seq base: 26.2s)"
echo " OBJETIVO: Media armónica > 4.0 | Max Speedup > 15 | Min Speedup > 1.75"
echo "========================================================================="
sum_inv_S=0
count=0

printf "$ROW_FMT" "Nodes / PPN (Procs)" "Time (s)" "Speedup" "Iterations"

# Les 9 configuracions en blau NODES, NPN (NODES FISICS, PROCESSOS PER NODE)
CONFIGS=(
    "2,1"
    "4,1"
    "8,1"
    "16,1"
    "16,2"
    "16,4"
    "16,8"
    "16,16"
    "16,32" # --overcommit, 16 nodes físics però 32 processos (2 per node)
)

for config in "${CONFIGS[@]}"; do
    # Separem els nodes i els processadors per node (npn)
    IFS=',' read -r nodes npn <<< "$config"
    procs=$((nodes * npn))
    
    # Si el segon paràmetre (ppn) és més gran que el primer (nodes), activem l'overcommit.
    if [ "$npn" -gt "$nodes" ]; then
        OVERCOMMIT_FLAG="--overcommit"
    else
        OVERCOMMIT_FLAG=""
    fi
    
    # Executem el comandament de SLURM injectant la flag només quan toca
    OUTPUT=$(salloc -p jutjat -N $nodes --exclusive srun -N $nodes -n $procs --ntasks-per-node=$npn --distribution=block:block $OVERCOMMIT_FLAG time ./kmean_p 2>&1)
    
    # Extraiem el valor del temps transcorregut (elapsed)
    TIME_STR=$(echo "$OUTPUT" | grep -oE '[0-9]+:[0-9]+\.[0-9]+elapsed|[0-9]+\.[0-9]+elapsed' | sed 's/elapsed//')
    
    # Si per alguna raó el format de time canvia o falla, intentem capturar-ho
    if [ -z "$TIME_STR" ]; then
        TIME_STR="0.0"
        TIME_PAR=1
        SPEEDUP="0.0"
        ITER="ERROR"
    else
        # Passem de minuts:segons a segons exactes
        TIME_PAR=$(echo "$TIME_STR" | awk -F: '{ if (NF==2) print ($1*60)+$2; else print $1 }')
        # Calculem l'Speedup exacte
        SPEEDUP=$(awk -v seq=$SEQ_JUTGES -v par=$TIME_PAR 'BEGIN { printf "%.5f", seq/par }')
        # Extraiem les iteracions per validar convergència
        ITER=$(echo "$OUTPUT" | grep -oE 'iter [0-9]+')
        if [ -z "$ITER" ]; then ITER="No conv"; fi
    fi
    
    # Imprimim la fila de la taula
    printf "$ROW_FMT" "$nodes / $npn ($procs)" "$TIME_STR s" "${SPEEDUP}" "$ITER"
    
    # Acumulem per a la mitjana harmònica: sum(1/Speedup) = sum(Par/Seq)
    sum_inv_S=$(awk -v sum=$sum_inv_S -v seq=$SEQ_JUTGES -v par=$TIME_PAR 'BEGIN { print sum + (par/seq) }')
    ((count++))
done

# Càlcul final de la Mitjana Harmònica d'Speedup
HMEAN=$(awk -v n=$count -v sum_inv=$sum_inv_S 'BEGIN { printf "%.5f", n/sum_inv }')
echo "-------------------------------------------------------------------------"
echo "-> MEDIA ARMÓNICA DE SPEEDUP OBTENIDA EN JUTGES: $HMEAN"
echo "========================================================================="

# Netegem l'executable a l'acabar
rm kmean_p