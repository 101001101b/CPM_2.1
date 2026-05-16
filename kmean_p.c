#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

#define N 2000000
#define G 400

// Funció Quicksort seqüencial intacta (S'executarà només al Procés 0)
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

// Nucli de l'algorisme K-Means adaptat per a MPI
void kmean_mpi(int fN_local, int fK, long fV_local[], long fR[], int fA_global[], int rank) {
    int i, j, min, iter = 0;
    long dif, t;
    long fS_local[G], fS_global[G];
    int fA_local[G];
    int *fD_local = (int *)malloc(fN_local * sizeof(int));

    do {
        // 1. Assignació de centroides al tros local de dades
        for (i = 0; i < fN_local; i++) {
            min = 0;
            dif = labs(fV_local[i] - fR[0]); // Ús de labs per evitar desbordaments amb long
            for (j = 1; j < fK; j++) {
                long d = labs(fV_local[i] - fR[j]);
                if (d < dif) {
                    min = j;
                    dif = d;
                }
            }
            fD_local[i] = min;
        }

        // 2. Preparació d'acumuladors locals
        for (i = 0; i < fK; i++) {
            fS_local[i] = 0;
            fA_local[i] = 0;
        }

        // 3. Acumulació de coordenades locals
        for (i = 0; i < fN_local; i++) {
            fS_local[fD_local[i]] += fV_local[i];
            fA_local[fD_local[i]]++;
        }

        // 4. Sincronització intel·ligent per xarxa (Només sumem els 400 centroides!)
        MPI_Allreduce(fS_local, fS_global, fK, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(fA_local, fA_global, fK, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // 5. Tots els nodes recalculen el nou centre per tenir la versió actualitzada
        dif = 0;
        for (i = 0; i < fK; i++) {
            t = fR[i];
            if (fA_global[i]) fR[i] = fS_global[i] / fA_global[i];
            dif += labs(t - fR[i]);
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

    long *V_all = NULL;
    long R[G];
    int A_global[G];

    // Càlcul del repartiment (Scatterv)
    int base_count = N / size;
    int remainder = N % size;

    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    
    int offset = 0;
    for (int i = 0; i < size; i++) {
        // Opció B: Repartir el residu donant una dada més als primers processos
        sendcounts[i] = base_count + (i < remainder ? 1 : 0);
        displs[i] = offset;
        offset += sendcounts[i];
    }

    int N_local = sendcounts[rank];
    long *V_local = (long *)malloc(N_local * sizeof(long));

    // NOMÉS el Procés 0 genera tota la memòria massiva i la carrega de dades inicials
    if (rank == 0) {
        V_all = (long *)malloc(N * sizeof(long));
        for (int i = 0; i < N; i++) {
            V_all[i] = (rand() % rand()) / N;
        }
        for (int i = 0; i < G; i++) {
            R[i] = V_all[i];
        }
    }

    // Repartim el pastís asimètricament per la xarxa
    MPI_Scatterv(V_all, sendcounts, displs, MPI_LONG,
                 V_local, N_local, MPI_LONG,
                 0, MPI_COMM_WORLD);

    // Repartim la versió inicial dels centroides a tothom
    MPI_Bcast(R, G, MPI_LONG, 0, MPI_COMM_WORLD);

    // Comença la computació
    kmean_mpi(N_local, G, V_local, R, A_global, rank);

    // Finalitzem: El master ordena i imprimeix l'estat definitiu
    if (rank == 0) {
        qs(0, G - 1, R, A_global);
        for (int i = 0; i < G; i++) {
            printf("R[%d] : %ld te %d agrupats\n", i, R[i], A_global[i]);
        }
        free(V_all); // Alliberem la matriu mestra
    }

    // Neteja de la memòria local que tenen tots els nodes
    free(V_local);
    free(sendcounts);
    free(displs);

    MPI_Finalize();
    return 0;
}