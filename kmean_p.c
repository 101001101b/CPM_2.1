#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h> // IMPRESCINDIBLE per a memcpy i memset

#define N 2000000
#define G 400

// Funció Quicksort seqüencial intacta (Només per al Procés 0)
void qs(int ii, int fi, long fV[], int fA[]) {
    int i, f;
    long pi, pa, vtmp, vta, vfi, vfa;

    pi = fV[ii];
    pa = fA[ii];
    i = ii + 1;
    f = fi;
    vtmp = fV[i];
    vta = fA[i];

    while (i <= f) {
        if (vtmp < pi) {
            fV[i - 1] = vtmp;
            fA[i - 1] = vta;
            i++;
            vtmp = fV[i];
            vta = fA[i];
        } else {
            vfi = fV[f];
            vfa = fA[f];
            fV[f] = vtmp;
            fA[f] = vta;
            f--;
            vtmp = vfi;
            vta = vfa;
        }
    }
    fV[i - 1] = pi;
    fA[i - 1] = pa;

    if (ii < f) qs(ii, f, fV, fA);
    if (i < fi) qs(i, fi, fV, fA);
}

void kmean_mpi(int fN_local, int fK, long fV_local[], long fR[], int fA_global[], int rank) {
    int i, j, min, iter = 0;
    long dif, t;
    
    // Bucle original seqüencial intacte
    int *fD_local = (int *)malloc(fN_local * sizeof(int));

    // Arrays d'acumulació
    long f_local[G * 2];
    long f_global[G * 2];
    
    // Buffer combinat per fer un sol Bcast net
    long bcast_buf[G + 1];

    do {
        // 1. Càlcul de distàncies (Amb promoció a registre)
        for (i = 0; i < fN_local; i++) {
            min = 0;
            long val = fV_local[i]; 
            long current_dif = labs(val - fR[0]);
            
            for (j = 1; j < fK; j++) {
                long d = labs(val - fR[j]);
                if (d < current_dif) {
                    min = j;
                    current_dif = d;
                }
            }
            fD_local[i] = min;
        }

        // 2. Inicialització a zero ultraràpida per maquinari (memset)
        memset(f_local, 0, fK * 2 * sizeof(long));

        // 3. Acumulació
        for (i = 0; i < fN_local; i++) {
            int idx = fD_local[i] * 2;         
            f_local[idx] += fV_local[i];       
            f_local[idx + 1]++;                
        }

        // 4. Reducció jeràrquica interna natural d'MPI
        MPI_Reduce(f_local, f_global, fK * 2, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        dif = 0;
        if (rank == 0) {
            for (i = 0; i < fK; i++) {
                t = fR[i];
                long suma = f_global[i * 2];
                int quants = (int)f_global[i * 2 + 1]; 
                
                if (quants) fR[i] = suma / quants;
                dif += labs(t - fR[i]);
                
                fA_global[i] = quants; 
            }
            
            // Empaquetem dif i els centres junts de cop usant bloqueig de memòria (memcpy)
            bcast_buf[0] = dif;
            memcpy(&bcast_buf[1], fR, fK * sizeof(long));
        }

        // 5. UN SOL Bcast de repartiment
        MPI_Bcast(bcast_buf, fK + 1, MPI_LONG, 0, MPI_COMM_WORLD);

        // Desempaquetem ràpid
        dif = bcast_buf[0];
        if (rank != 0) {
            memcpy(fR, &bcast_buf[1], fK * sizeof(long));
        }

        iter++;
    } while (dif);

    if (rank == 0) printf("iter %d\n", iter);
    free(fD_local);
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ===================================================================
    // CRONÒMETRE GLOBAL (Engloba TOT el procés, sense excepcions)
    // ===================================================================
    MPI_Barrier(MPI_COMM_WORLD); 
    double t_start = MPI_Wtime(); 

    long *V_all = NULL;
    long R[G];
    int A_global[G];

    int base_count = N / size;
    int remainder = N % size;

    int N_local = base_count + (rank < remainder ? 1 : 0);
    long *V_local = (long *)malloc(N_local * sizeof(long));

    int *sendcounts = NULL;
    int *displs = NULL;

    if (rank == 0) {
        V_all = (long *)malloc(N * sizeof(long));
        sendcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
        
        int offset = 0;
        for (int i = 0; i < size; i++) {
            sendcounts[i] = base_count + (i < remainder ? 1 : 0);
            displs[i] = offset;
            offset += sendcounts[i];
        }

        for (int i = 0; i < N; i++) {
            V_all[i] = (rand() % rand()) / N;
        }
        for (int i = 0; i < G; i++) {
            R[i] = V_all[i];
        }
    }

    // Repartiment
    MPI_Scatterv(V_all, sendcounts, displs, MPI_LONG,
                 V_local, N_local, MPI_LONG,
                 0, MPI_COMM_WORLD);

    MPI_Bcast(R, G, MPI_LONG, 0, MPI_COMM_WORLD);

    // Algorisme paral·lel
    kmean_mpi(N_local, G, V_local, R, A_global, rank);

    // Ordenació i impressió seqüencial
    if (rank == 0) {
        qs(0, G - 1, R, A_global);
        
        /*for (int i = 0; i < G; i++) {
            printf("R[%d] : %ld te %d agrupats\n", i, R[i], A_global[i]);
        }*/
        free(V_all); 
        free(sendcounts);
        free(displs);
    }

    free(V_local);

    // Assegurem que tothom ha acabat abans d'aturar el rellotge
    MPI_Barrier(MPI_COMM_WORLD); 
    
    if (rank == 0) {
        double t_end = MPI_Wtime(); 
        printf("TIEMPO: %f s\n", t_end - t_start);
    }

    MPI_Finalize();
    return 0;
} 