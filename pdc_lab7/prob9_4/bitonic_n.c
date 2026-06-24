/* Problem 9.4 - Standard bitonic sort with n processes (1 element each), MPI.
 * Used as the baseline to compare against the n/2-process block version.
 *
 * Build:  mpicc -O2 -o bitonic_n bitonic_n.c
 * Run:    mpirun -np 16 ./bitonic_n     (n = 16)
 *         mpirun -np 32 ./bitonic_n     (n = 32)
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/* compare-exchange one element with partner; keep min if keep_low else max */
static long compare_exchange(int *x, int partner, int keep_low) {
    int other;
    MPI_Sendrecv(x, 1, MPI_INT, partner, 0, &other, 1, MPI_INT, partner, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (keep_low) { if (other < *x) *x = other; }
    else          { if (other > *x) *x = other; }
    return 1;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, n;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n);

    if (n < 2 || (n & (n - 1))) {
        if (rank == 0) fprintf(stderr, "Number of processes must be a power of 2.\n");
        MPI_Finalize(); return 1;
    }

    int x;
    srand(1000 + rank);
    x = rand() % 1000;

    long messages = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    int d = 0; while ((1 << d) < n) d++;
    for (int i = 0; i < d; i++) {
        for (int j = i; j >= 0; j--) {
            int partner = rank ^ (1 << j);
            int ascending = ((rank >> (i + 1)) & 1) == 0;
            int keep_low;
            if (rank < partner) keep_low =  ascending;
            else                keep_low = !ascending;
            messages += compare_exchange(&x, partner, keep_low);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    int *all = NULL;
    if (rank == 0) all = malloc((size_t)n * sizeof(int));
    MPI_Gather(&x, 1, MPI_INT, all, 1, MPI_INT, 0, MPI_COMM_WORLD);

    long total_messages = 0;
    MPI_Reduce(&messages, &total_messages, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int sorted = 1;
        for (int t = 1; t < n; t++) if (all[t] < all[t - 1]) { sorted = 0; break; }
        printf("BITONIC n-proc :  n=%d  p=%d  k=1  time=%.6f s  messages=%ld  %s\n",
               n, n, t1 - t0, total_messages, sorted ? "SORTED" : "NOT SORTED");
        free(all);
    }
    MPI_Finalize();
    return 0;
}
