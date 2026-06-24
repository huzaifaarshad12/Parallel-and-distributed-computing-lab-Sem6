/* Problem 9.9 - Block version on a ring for weak-scaling / isoefficiency.
 *
 * Fix p (e.g. 4) and vary block size k = n/p. Each process holds k elements;
 * run block-based bitonic sort over the ring and measure execution time so the
 * report can compute efficiency and fit an isoefficiency function.
 *
 * Build:  mpicc -O2 -o block_ring block_ring.c
 * Run:    mpirun -np 4 ./block_ring <k>     (k = n/p, e.g. 4, 8, 16)
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static int cmp_int(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

static void compare_split(int *mine, int k, int partner, int keep_low) {
    int *other  = malloc((size_t)k * sizeof(int));
    int *merged = malloc((size_t)2 * k * sizeof(int));
    MPI_Sendrecv(mine, k, MPI_INT, partner, 0, other, k, MPI_INT, partner, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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

    int k = (argc > 1) ? atoi(argv[1]) : 8;
    if (p < 2 || (p & (p - 1))) {
        if (rank == 0) fprintf(stderr, "p must be a power of 2.\n");
        MPI_Finalize(); return 1;
    }
    int n = p * k;

    int dims[1] = {p}, periods[1] = {1};
    MPI_Comm ring;
    MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, 0, &ring);

    int *mine = malloc((size_t)k * sizeof(int));
    srand(321 + rank * 17);
    for (int i = 0; i < k; i++) mine[i] = rand() % 100000;
    qsort(mine, k, sizeof(int), cmp_int);

    MPI_Barrier(ring);
    double t0 = MPI_Wtime();

    int d = 0; while ((1 << d) < p) d++;
    for (int i = 0; i < d; i++) {
        for (int j = i; j >= 0; j--) {
            int partner = rank ^ (1 << j);
            int ascending = ((rank >> (i + 1)) & 1) == 0;
            int low_side  = ((rank >> j) & 1) == 0;
            int keep_low  = ascending ? low_side : !low_side;
            compare_split(mine, k, partner, keep_low);
        }
    }

    MPI_Barrier(ring);
    double t1 = MPI_Wtime();

    int *all = NULL;
    if (rank == 0) all = malloc((size_t)n * sizeof(int));
    MPI_Gather(mine, k, MPI_INT, all, k, MPI_INT, 0, ring);
    if (rank == 0) {
        int sorted = 1;
        for (int t = 1; t < n; t++) if (all[t] < all[t - 1]) sorted = 0;
        printf("BLOCK-RING  p=%d  k=%d  n=%d  time=%.6f s  %s\n",
               p, k, n, t1 - t0, sorted ? "SORTED" : "NOT SORTED");
        free(all);
    }
    free(mine);
    MPI_Comm_free(&ring);
    MPI_Finalize();
    return 0;
}
