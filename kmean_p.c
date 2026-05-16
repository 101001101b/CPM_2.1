#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

#define N 2000000  
#define G 400      

long *V_local;      // Jutges que processa cada procés (N/mida)
long R[G];          // centroides. Tots els threads tenen una copia
int A_global[G];    // Quants jutges han caigut a cada grup (total de tots els processos)

void kmean_mpi(int fN_local, int fK, long fV_local[], long fR[], int fA_global[], int rank)
{
    int i, j, min, iter = 0;
    long dif, t;
    long fS_local[G];   // Suma local de les coordenades dels jutges d'aquest procés
    long fS_global[G];  // Suma global de coordenades (resultat de Allreduce)
    int fA_local[G];    // Comptador local de jutges per grup d'aquest procés
    int *fD_local = (int *)malloc(fN_local * sizeof(int)); // Índex del cluster per a cada jutge local

    do {
        for (i = 0; i < fN_local; i++) {
            min = 0; 
            dif = abs(fV_local[i] - fR[0]);
            for (j = 1; j < fK; j++) {
                long d = abs(fV_local[i] - fR[j]);
                if (d < dif) {
                    min = j; // centroide més proper
                    dif = d;
                }
            }
            fD_local[i] = min; // el jutge 'i' va al grup 'min'
        }

        // --- Preparar sumes ---
        for (i = 0; i < fK; i++) {
            fS_local[i] = 0;
            fA_local[i] = 0;
        }

        for (i = 0; i < fN_local; i++) {
            fS_local[fD_local[i]] += fV_local[i]; // Acumulem el valor del jutge al seu grup
            fA_local[fD_local[i]]++;              // Incrementem el comptador del grup
        }

        // --- Sincronització memoria distribuïda ---
        // Sumem les 'pissarres' de tots els processos per tenir la foto global
        MPI_Allreduce(fS_local, fS_global, fK, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(fA_local, fA_global, fK, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // --- Càlcul de nous centres ---
        dif = 0; // Acumula el desplaçament total dels centres
        for (i = 0; i < fK; i++) {
            t = fR[i]; // Guardem el centre vell per comparar
            if (fA_global[i]) fR[i] = fS_global[i] / fA_global[i]; // Nou centre = Mitjana
            dif += abs(t - fR[i]); // Si el centre s'ha mogut, dif serà > 0
        }
        iter++;
    } while (dif); // Si dif == 0, els centres ja no es mouen i ha convergit

    if (rank == 0) printf("iter %d\n", iter);
    free(fD_local);
}

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

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    int N_local = N / size; // Quants jutges li toquen a cada procés
    V_local = (long *)malloc(N_local * sizeof(long));
    long *V_all = NULL; // Vector amb tots els jutges (només l'omple el rank 0)

    if (rank == 0) {
        V_all = (long *)malloc(N * sizeof(long));
        srand(123); 
        for (int i = 0; i < N; i++) V_all[i] = (rand() % rand()) / N; // Generem dades
        for (int i = 0; i < G; i++) R[i] = V_all[i]; // Candidats inicials a centres
    }

    // El master envia a cadascú el seu tros de jutges
    MPI_Scatter(V_all, N_local, MPI_LONG, V_local, N_local, MPI_LONG, 0, MPI_COMM_WORLD);
    
    // El master envia els centres inicials a tots
    MPI_Bcast(R, G, MPI_LONG, 0, MPI_COMM_WORLD);

    kmean_mpi(N_local, G, V_local, R, A_global, rank);
    qs(0, G-1, R, A_global);
    
    // El master imprimeix el resultat
    if (rank == 0) {
        for (int i = 0; i < G; i++) 
            printf("R[%d] : %ld te %d agrupats\n", i, R[i], A_global[i]);
        free(V_all);
    }

    free(V_local);
    MPI_Finalize(); 
    return 0;
}