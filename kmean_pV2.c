#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define N 2000000
#define G 400

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
    
    int *fD_local = (int *)malloc(fN_local * sizeof(int));
    
    long f_local[G * 2] __attribute__((aligned(64))); 
    long f_global[G * 2] __attribute__((aligned(64)));

    do {
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

        memset(f_local, 0, fK * 2 * sizeof(long)); // Reiniciem f_local a zero abans de l'acumulació

        for (i = 0; i < fN_local; i++) {
            int idx = fD_local[i] * 2;         
            f_local[idx] += fV_local[i];       
            f_local[idx + 1]++;                
        }

        MPI_Allreduce(f_local, f_global, fK * 2, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

        dif = 0;
        for (i = 0; i < fK; i++) {
            t = fR[i];
            long suma = f_global[i * 2]; // Suma de valors per al cluster i (parells)
            int quants = (int)f_global[i * 2 + 1];  // Nombre de valors per al cluster i (senars)
            
            if (quants) fR[i] = suma / quants;
            dif += labs(t - fR[i]);
            
            fA_global[i] = quants; 
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

    MPI_Barrier(MPI_COMM_WORLD);
    double t_algo_start = MPI_Wtime();
    
    long R[G]; 
    int A_global[G];

    int base_count = N / size;
    int remainder = N % size;
    int N_local = base_count + (rank < remainder ? 1 : 0);
    
    // 1. Cada procés calcula el seu offset global de forma independent
    int offset = 0;
    for (int i = 0; i < rank; i++) {
        offset += base_count + (i < remainder ? 1 : 0);
    }

    long *V_local = (long *)malloc(N_local * sizeof(long));

    srand(1); // Forcem mateixa llavor a tots els processos
    for (int i = 0; i < N; i++) {
        long val = (rand() % rand()) / N;
        
        // Tots guarden els primers G com a centroides inicials
        if (i < G) {
            R[i] = val;
        }
        
        // Només emmagatzemem a memòria el troç que ens pertoca a nosaltres
        if (i >= offset && i < offset + N_local) {
            V_local[i - offset] = val;
        }
    }

    kmean_mpi(N_local, G, V_local, R, A_global, rank);

    MPI_Barrier(MPI_COMM_WORLD); 
    double t_algo_end = MPI_Wtime(); 

    if (rank == 0) {
        qs(0, G - 1, R, A_global);
        
        char *out_buf = (char *)malloc(65536);
        int p_offset = 0;
        for (int i = 0; i < G; i++) {
            p_offset += sprintf(out_buf + p_offset, "R[%d] : %ld te %d agrupats\n", i, R[i], A_global[i]);
        }
        printf("%s", out_buf);
        free(out_buf);
        
        printf("TIEMPO_ALGO:%f\n", t_algo_end - t_algo_start);
    }

    free(V_local);
    MPI_Finalize();
    return 0;
}
