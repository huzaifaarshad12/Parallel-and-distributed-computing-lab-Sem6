/* Problem 9.10 - Block-based odd-even transposition sort (MPI, linear array).
 *
 * p processes in a linear array, each holds n/p elements (locally sorted).
 * Run p phases. In even phases, processes (0,1),(2,3),... compare-split; in
 * odd phases, (1,2),(3,4),... compare-split. The lower-rank partner keeps the
 * low half, the higher-rank keeps the high half. After p phases the global
 * sequence is sorted.
 *
 * Test: p in {2,4,8}, n in {16,32,64}. Reports execution time and total
 * communication volume for comparison with block bitonic (Problem 9.8).
 *
 * Build:  mpicc -O2 -o oddeven oddeven.c
 * Run:    mpirun -np 4 ./oddeven 32
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
    if (n % p != 0) {
        if (rank == 0) fprintf(stderr, "n (%d) must be a multiple of p (%d).\n", n, p);
        MPI_Finalize(); return 1;
    }
    int k = n / p;

    int *mine = malloc((size_t)k * sizeof(int));
    srand(909 + rank * 23);
    for (int i = 0; i < k; i++) mine[i] = rand() % 10000;
    qsort(mine, k, sizeof(int), cmp_int);     /* local sort */

    long msgs = 0, bytes = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int phase = 0; phase < p; phase++) {
        int partner;
        if (phase % 2 == 0)                       /* even phase: (0,1)(2,3)... */
            partner = (rank % 2 == 0) ? rank + 1 : rank - 1;
        else                                      /* odd phase: (1,2)(3,4)...  */
            partner = (rank % 2 == 0) ? rank - 1 : rank + 1;
        if (partner < 0 || partner >= p) continue;  /* boundary idles */
        int keep_low = (rank < partner);          /* lower rank keeps low half */
        compare_split(mine, k, partner, keep_low, &msgs, &bytes);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    int *all = NULL;
    if (rank == 0) all = malloc((size_t)n * sizeof(int));
    MPI_Gather(mine, k, MPI_INT, all, k, MPI_INT, 0, MPI_COMM_WORLD);

    long tot_msgs, tot_bytes;
    MPI_Reduce(&msgs,  &tot_msgs,  1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&bytes, &tot_bytes, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int sorted = 1;
        for (int t = 1; t < n; t++) if (all[t] < all[t - 1]) sorted = 0;
        printf("ODD-EVEN  n=%d p=%d k=%d  time=%.6f s  messages=%ld  comm_bytes=%ld  %s\n",
               n, p, k, t1 - t0, tot_msgs, tot_bytes, sorted ? "SORTED" : "NOT SORTED");
        free(all);
    }
    free(mine);
    MPI_Finalize();
    return 0;
}
