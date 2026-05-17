/*
----------------------------------------------------------------------------------------------------------
Nodes / PPN (Procs)  | Time (s)   | MPI_Wtime (s) | Speedup (Time) | Speedup (MPI)  | Valid    | Iters      |
----------------------------------------------------------------------------------------------------------
2 / 1 (2)            | 6.585      | 6.094         | 3.97874        | 4.29909        | ✅ OK   | iter 68    |
4 / 1 (4)            | 3.621      | 3.127         | 7.23557        | 8.37753        | ✅ OK   | iter 68    |
8 / 1 (8)            | 2.234      | 1.709         | 11.72784       | 15.32789       | ✅ OK   | iter 68    |
16 / 1 (16)          | 1.464      | 0.928         | 17.89617       | 28.23547       | ✅ OK   | iter 68    |
16 / 2 (32)          | 1.379      | 0.713         | 18.99927       | 36.74362       | ✅ OK   | iter 68    |
16 / 4 (64)          | 1.247      | 0.535         | 21.01043       | 48.95119       | ✅ OK   | iter 68    |
16 / 8 (128)         | 1.289      | 0.553         | 20.32583       | 47.39937       | ✅ OK   | iter 68    |
16 / 16 (256)        | 1.301      | 0.530         | 20.13836       | 49.48027       | ✅ OK   | iter 68    |
16 / 32 (512)        | 7.200      | 6.320         | 3.63889        | 4.14559        | ✅ OK   | iter 68    |
----------------------------------------------------------------------------------------------------------
-> MEDIA ARMÓNICA DE SPEEDUP (Time):      8.95897
-> MEDIA ARMÓNICA DE SPEEDUP (MPI_Wtime): 11.49715
==========================================================================================================

*/

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <immintrin.h> // Llibreria d'intrínseques per a AVX-512

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
    int i, j, iter = 0;
    long dif, t;
    
    int *fD_local = (int *)malloc(fN_local * sizeof(int));
    
    long f_local[G * 2] __attribute__((aligned(64)));
    long f_global[G * 2] __attribute__((aligned(64)));

    do {
        // ===================================================================
        // 1. CÀLCUL DE DISTÀNCIES: AVX-512 EXPLICIT VECTORIZATION
        // ===================================================================
        for (i = 0; i < fN_local; i++) {
            long val = fV_local[i]; 
            
            // Creem un vector ZMM amb el nostre punt repetit 8 vegades
            __m512i v_val = _mm512_set1_epi64(val);
            
            // Inicialitzem els vectors de distàncies mínimes al màxim possible
            __m512i v_min_dists = _mm512_set1_epi64(9223372036854775807LL);
            __m512i v_min_idxs  = _mm512_setzero_si512();
            
            // L'índex actual per als 8 centroides que estem processant [7,6,5,4,3,2,1,0]
            __m512i v_curr_idxs = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
            __m512i v_idx_step  = _mm512_set1_epi64(8);

            // Saltem de 8 en 8 (400 / 8 = 50 iteracions perfectes)
            for (j = 0; j < fK; j += 8) {
                // Càrrega no alineada de 8 centroides de cop
                __m512i v_centroids = _mm512_loadu_epi64(&fR[j]);
                
                // Resta i Valor Absolut nadiu a la CPU
                __m512i v_diff = _mm512_sub_epi64(v_val, v_centroids);
                __m512i v_abs_diff = _mm512_abs_epi64(v_diff);
                
                // MÀSCARA (Substitueix a l'IF): Posem un 1 on la nova distància sigui menor
                __mmask8 mask = _mm512_cmp_epi64_mask(v_abs_diff, v_min_dists, _MM_CMPINT_LT);
                
                // BLEND: Actualitzem els mínims només on la màscara ens hagi donat permís
                v_min_dists = _mm512_mask_blend_epi64(mask, v_min_dists, v_abs_diff);
                v_min_idxs  = _mm512_mask_blend_epi64(mask, v_min_idxs, v_curr_idxs);
                
                // Sumem 8 a tots els índex per al següent bloc de centroides
                v_curr_idxs = _mm512_add_epi64(v_curr_idxs, v_idx_step);
            }

            // REDUCCIÓ HORITZONTAL: Bolquem els 8 guanyadors finals a la memòria L1
            long tmp_dists[8] __attribute__((aligned(64)));
            long tmp_idxs[8]  __attribute__((aligned(64)));
            
            _mm512_store_epi64(tmp_dists, v_min_dists);
            _mm512_store_epi64(tmp_idxs, v_min_idxs);

            long final_min_dist = tmp_dists[0];
            int final_min_idx = (int)tmp_idxs[0];

            // Bucle escalar final per desempatar entre els 8 finalistes
            for (int k = 1; k < 8; k++) {
                if (tmp_dists[k] < final_min_dist || (tmp_dists[k] == final_min_dist && tmp_idxs[k] < final_min_idx)) {
                    final_min_dist = tmp_dists[k];
                    final_min_idx = (int)tmp_idxs[k];
                }
            }
            
            fD_local[i] = final_min_idx;
        }

        // 2 i 3. Neteja i Acumulació local
        memset(f_local, 0, fK * 2 * sizeof(long));

        for (i = 0; i < fN_local; i++) {
            int idx = fD_local[i] * 2;         
            f_local[idx] += fV_local[i];       
            f_local[idx + 1]++;                
        }

        // 4. Sincronització Simètrica
        MPI_Allreduce(f_local, f_global, fK * 2, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

        // 5. Actualització de Centroides Local
        dif = 0;
        for (i = 0; i < fK; i++) {
            t = fR[i];
            long suma = f_global[i * 2];
            int quants = (int)f_global[i * 2 + 1]; 
            
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

    double t_algo_start = MPI_Wtime();
    
    // Assegurem que l'array de centroides estigui alineat per l'AVX-512
    long R[G] __attribute__((aligned(64))); 
    int A_global[G];

    int base_count = N / size;
    int remainder = N % size;
    int N_local = base_count + (rank < remainder ? 1 : 0);
    
    int offset = 0;
    for (int i = 0; i < rank; i++) {
        offset += base_count + (i < remainder ? 1 : 0);
    }

    long *V_local = (long *)malloc(N_local * sizeof(long));

    // GENERACIÓ A L'OMBRA
    srand(1);
    for (int i = 0; i < N; i++) {
        long val = (rand() % rand()) / N;
        
        if (i < G) {
            R[i] = val;
        }
        
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