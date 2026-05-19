#!/bin/bash
export LC_NUMERIC=C

# Compilació
mpicc -O3 kmean_p.c -o kmean_p

if [ $? -ne 0 ]; then
    echo "❌ ERROR: No se ha podido compilar kmean_p.c. Abortando pruebas."
    exit 1
fi

# ==============================================================================
# TRUCO HPC: Autoreserva de SLURM
# Si no estamos dentro de una asignación, pedimos 16 nodos y relanzamos el script.
# ==============================================================================
if [ -z "$SLURM_JOB_ID" ]; then
    echo "=========================================================================================================="
    echo " ⏳ Solicitando reserva exclusiva de 16 nodos de golpe. Esperando a SLURM..."
    echo "=========================================================================================================="
    # Pedimos los 16 nodos y, una vez concedidos, ejecutamos este mismo script dentro
    salloc -p jutjat -N 16 --exclusive bash "$0" "$@"
    
    # Al terminar (o si el usuario cancela), salimos limpiamente liberando los nodos
    exit $?
fi

# ==============================================================================
# A PARTIR DE AQUÍ: Ya somos dueños de 16 nodos exclusivos. Todo será instantáneo.
# ==============================================================================

SEQ_JUTGES=26.2
ROW_FMT="%-20s | %-10s | %-13s | %-14s | %-14s | %-8s | %-10s |\n"

echo "=========================================================================================================="
echo " PRUEBAS MPI EN MATRICES JUTGES (Tiempo seq base: 26.2s)"
echo " OBJETIVO: Media armónica > 4.0 | Max Speedup > 15 | Min Speedup > 1.75"
echo "=========================================================================================================="

echo "⏳ Generando Referencia Maestra (1 proceso) para validación estricta..."
# Ya no hace falta salloc, simplemente disparamos el srun que cogerá 1 nodo de los 16
srun -N 1 -n 1 --ntasks-per-node=1 --distribution=block:block ./kmean_p | grep '^R\[' > ref.out

if [ ! -s ref.out ]; then
    echo "❌ ERROR: No se ha podido generar la referencia. Revisa el código C."
    exit 1
fi
echo "✅ Referencia generada correctamente."
echo "----------------------------------------------------------------------------------------------------------"

sum_inv_S_time=0
sum_inv_S_mpi=0
count=0

printf "$ROW_FMT" "Nodes / PPN (Procs)" "Time (s)" "MPI_Wtime (s)" "Speedup (Time)" "Speedup (MPI)" "Valid" "Iters"
echo "----------------------------------------------------------------------------------------------------------"

CONFIGS=(
    "2 1" "4 1" "8 1"
    "16 1" "16 2" "16 4"
    "16 8" "16 16" "16 32"
)

for config in "${CONFIGS[@]}"; do
    read -r nodes ppn <<< "$config"
    procs=$((nodes * ppn))
    
    if [ "$ppn" -gt "$nodes" ]; then
        OVERCOMMIT_FLAG="--overcommit"
    else
        OVERCOMMIT_FLAG=""
    fi
    
    # Executem guardant la sortida (ja sense salloc, només capturem el 'time' del srun)
    OUTPUT=$( { TIMEFORMAT='TIEMPO_REAL:%R'; time srun -N $nodes -n $procs --ntasks-per-node=$ppn --distribution=block:block $OVERCOMMIT_FLAG ./kmean_p; } 2>&1 )
    
    # Extraiem els temps i la validació
    TIME_REAL=$(echo "$OUTPUT" | grep "TIEMPO_REAL:" | cut -d':' -f2 | awk '{print $1}')
    TIME_ALGO=$(echo "$OUTPUT" | grep "TIEMPO_ALGO:" | cut -d':' -f2 | awk '{print $1}')
    ITER=$(echo "$OUTPUT" | grep -oE 'iter [0-9]+' | head -n 1)
    
    # VERIFICACIÓN ESTRICTA (Byte a Byte)
    echo "$OUTPUT" | grep "^R\[" > test.out
    if diff -q ref.out test.out > /dev/null; then
        VALID="✅ OK"
    else
        VALID="❌ MAL"
    fi
    
    if [ -z "$TIME_REAL" ] || [ "$TIME_REAL" == "" ]; then
        TIME_REAL_FMT="ERR"
        TIME_ALGO_FMT="ERR"
        SPEEDUP_TIME="0.00000"
        SPEEDUP_MPI="0.00000"
        ITER="ERROR"
        VALID="ERR"
    else
        SPEEDUP_TIME=$(awk -v seq=$SEQ_JUTGES -v par=$TIME_REAL 'BEGIN { printf "%.5f", seq/par }')
        sum_inv_S_time=$(awk -v sum=$sum_inv_S_time -v seq=$SEQ_JUTGES -v par=$TIME_REAL 'BEGIN { print sum + (par/seq) }')
        TIME_REAL_FMT=$(awk -v real=$TIME_REAL 'BEGIN { printf "%.3f", real }')
        
        if [ -n "$TIME_ALGO" ] && [ "$TIME_ALGO" != "" ]; then
            SPEEDUP_MPI=$(awk -v seq=$SEQ_JUTGES -v par=$TIME_ALGO 'BEGIN { printf "%.5f", seq/par }')
            sum_inv_S_mpi=$(awk -v sum=$sum_inv_S_mpi -v seq=$SEQ_JUTGES -v par=$TIME_ALGO 'BEGIN { print sum + (par/seq) }')
            TIME_ALGO_FMT=$(awk -v algo=$TIME_ALGO 'BEGIN { printf "%.3f", algo }')
        else
            TIME_ALGO_FMT="ERR"
            SPEEDUP_MPI="ERR"
        fi
    fi
    
    printf "$ROW_FMT" "$nodes / $ppn ($procs)" "${TIME_REAL_FMT}" "${TIME_ALGO_FMT}" "${SPEEDUP_TIME}" "${SPEEDUP_MPI}" "$VALID" "$ITER"
    ((count++))
done

HMEAN_TIME=$(awk -v n=$count -v sum_inv=$sum_inv_S_time 'BEGIN { printf "%.5f", n/sum_inv }')
HMEAN_MPI=$(awk -v n=$count -v sum_inv=$sum_inv_S_mpi 'BEGIN { printf "%.5f", n/sum_inv }')

echo "----------------------------------------------------------------------------------------------------------"
echo "-> MEDIA ARMÓNICA DE SPEEDUP (Time):      $HMEAN_TIME"
echo "-> MEDIA ARMÓNICA DE SPEEDUP (MPI_Wtime): $HMEAN_MPI"
echo "=========================================================================================================="

# Neteja de fitxers temporals
rm kmean_p ref.out test.out