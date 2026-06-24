/* Problem 9.8 - Block-based bitonic sort (MPI).
 *
 * p processes (power of 2), each holds n/p elements. The bitonic sort network
 * runs over the p blocks; each compare-EXCHANGE becomes a compare-SPLIT:
 * exchange the whole block, merge the two sorted blocks, keep low or high half.
 * Each process keeps its block locally sorted throughout.
 *
 * Test: p in {2,4,8}, n in {16,32,64} (n must be a multiple of p).
 * Reports: total compare-split operations and total communication volume,
 * for comparison with the element-wise version (which would need n processes).
 *
 * Build:  mpicc -O2 -o block_bitonic block_bitonic.c
 * Run:    mpirun -np 4 ./block_bitonic 32      (p from -np, n from argv)
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static int cmp_int(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

static void compare_split(int *mine, int k, int partner, int keep_low,
                          long *msgs, long *bytes) {
    int *other  = malloc((size_t)k * sizeof(int));
    int *merged = malloc((size_t)2 * k * sizeof(int));
    MPI_Sendrecv(mine, k, MPI_INT, partner, 0, other, k, MPI_INT, partner, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    *msgs += 1; *bytes += (long)k * sizeof(int);
    int i = 0, j = 0, m = 0;
    while (i < k && j < k) merged[m++] = (mine[i] <= other[j]) ? mine[i++] : other[j++];
    while (i < k) merged[m++] = mine[i++];
    while (j < k) merged[m++] = other[j++];
    if (keep_low) for (int t = 0; t < k; t++) mine[t] = merged[t];
    else          for (int t = 0; t < k; t++) mine[t] = merged[k + t];
    free(other); free(merged);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int n = (argc > 1) ? atoi(argv[1]) : 32;
    if (p < 2 || (p & (p - 1))) {
        if (rank == 0) fprintf(stderr, "p must be a power of 2.\n");
        MPI_Finalize(); return 1;
    }
    if (n % p != 0) {
        if (rank == 0) fprintf(stderr, "n (%d) must be a multiple of p (%d).\n", n, p);
        MPI_Finalize(); return 1;
    }
    int k = n / p;

    int *mine = malloc((size_t)k * sizeof(int));
    srand(555 + rank * 31);
    for (int i = 0; i < k; i++) mine[i] = rand() % 10000;
    qsort(mine, k, sizeof(int), cmp_int);     /* local sort first */

    long msgs = 0, bytes = 0, splits = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    int d = 0; while ((1 << d) < p) d++;
    for (int i = 0; i < d; i++) {
        for (int j = i; j >= 0; j--) {
            int partner = rank ^ (1 << j);
            int ascending = ((rank >> (i + 1)) & 1) == 0;
            int low_side  = ((rank >> j) & 1) == 0;
            int keep_low  = ascending ? low_side : !low_side;
            compare_split(mine, k, partner, keep_low, &msgs, &bytes);
            splits++;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    int *all = NULL;
    if (rank == 0) all = malloc((size_t)n * sizeof(int));
    MPI_Gather(mine, k, MPI_INT, all, k, MPI_INT, 0, MPI_COMM_WORLD);

    long tot_msgs, tot_bytes, tot_splits;
    MPI_Reduce(&msgs,   &tot_msgs,   1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&bytes,  &tot_bytes,  1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&splits, &tot_splits, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int sorted = 1;
        for (int t = 1; t < n; t++) if (all[t] < all[t - 1]) sorted = 0;
        /* element-wise version would use n processes; its compare-exchange
         * count = n * (d_n*(d_n+1)/2) where d_n = log2(n). */
        int dn = 0; while ((1 << dn) < n) dn++;
        long elemwise_ops = (long)n * (dn * (dn + 1) / 2);
        printf("BLOCK-BITONIC  n=%d p=%d k=%d  time=%.6f s  compare-splits=%ld  "
               "comm_bytes=%ld  %s\n",
               n, p, k, t1 - t0, tot_splits, tot_bytes,
               sorted ? "SORTED" : "NOT SORTED");
        printf("   (element-wise n-proc version would need ~%ld compare-exchanges "
               "on %d processes)\n", elemwise_ops, n);
        free(all);
    }

    free(mine);
    MPI_Finalize();
    return 0;
}
