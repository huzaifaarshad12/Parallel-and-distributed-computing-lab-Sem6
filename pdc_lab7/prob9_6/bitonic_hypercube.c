/* Problem 9.6 - Full bitonic sort on a hypercube (Algorithm 9.1), MPI.
 *
 * Each of n = 2^d processes holds one element. The algorithm runs the standard
 * bitonic sort: for i = 0..d-1 (phase building bitonic seqs of size 2^(i+1)),
 * for j = i..0 (stage), each process compare-exchanges with neighbour
 * rank XOR (1<<j). The sort direction at a node is determined by the bitwise
 * condition:  ascending iff bit (i+1) of the rank is 0.
 *
 * Direction of keep:
 *   - if the (j)-th bit of rank is 0  -> this process is the "low" side
 *   - combine with the ascending/descending window to decide keep-min/keep-max
 *
 * Verification: gather at rank 0 and check ascending order.
 * Debug: pass "1" as argv[1] to print the global distribution after each phase.
 *
 * Build:  mpicc -O2 -o bitonic_hypercube bitonic_hypercube.c
 * Run:    mpirun -np 8  ./bitonic_hypercube       (n=8)
 *         mpirun -np 16 ./bitonic_hypercube 1     (n=16, with debug)
 *         mpirun -np 32 ./bitonic_hypercube       (n=32)
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static void dump_distribution(int x, int n, int rank, int phase) {
    int *all = NULL;
    if (rank == 0) all = malloc(n * sizeof(int));
    MPI_Gather(&x, 1, MPI_INT, all, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("  after phase %d: ", phase);
        for (int i = 0; i < n; i++) printf("%d ", all[i]);
        printf("\n");
        free(all);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, n;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n);

    int debug = (argc > 1) ? atoi(argv[1]) : 0;

    if (n < 2 || (n & (n - 1))) {
        if (rank == 0) fprintf(stderr, "n must be a power of 2 (8, 16, 32).\n");
        MPI_Finalize(); return 1;
    }
    int d = 0; while ((1 << d) < n) d++;

    int x;
    srand(777 + rank * 13);
    x = rand() % 1000;

    if (debug && rank == 0) printf("n=%d, d=log2(n)=%d\n", n, d);
    if (debug) dump_distribution(x, n, rank, -1);   /* initial */

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int i = 0; i < d; i++) {
        for (int j = i; j >= 0; j--) {
            int partner = rank ^ (1 << j);
            int other;
            MPI_Sendrecv(&x, 1, MPI_INT, partner, 0, &other, 1, MPI_INT, partner, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            /* ascending window iff bit (i+1) of rank is 0 */
            int ascending = ((rank >> (i + 1)) & 1) == 0;
            /* low side iff bit j of rank is 0 */
            int low_side  = ((rank >> j) & 1) == 0;
            /* low side in ascending window keeps the min; flip otherwise */
            int keep_min = ascending ? low_side : !low_side;
            if (keep_min) { if (other < x) x = other; }
            else          { if (other > x) x = other; }
        }
        if (debug) dump_distribution(x, n, rank, i);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    int *all = NULL;
    if (rank == 0) all = malloc(n * sizeof(int));
    MPI_Gather(&x, 1, MPI_INT, all, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        int sorted = 1;
        for (int i = 1; i < n; i++) if (all[i] < all[i - 1]) sorted = 0;
        printf("HYPERCUBE BITONIC SORT  n=%d  time=%.6f s  %s\n",
               n, t1 - t0, sorted ? "SORTED -> PASS" : "NOT SORTED -> FAIL");
        if (n <= 32) {
            printf("final: ");
            for (int i = 0; i < n; i++) printf("%d ", all[i]);
            printf("\n");
        }
        free(all);
    }

    MPI_Finalize();
    return 0;
}
