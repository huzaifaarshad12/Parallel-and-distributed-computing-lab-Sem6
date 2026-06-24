/* Problem 9.4 - Bitonic sort with n/2 processes (MPI)
 *
 * Standard bitonic sort assumes n processes for n items (1 element each).
 * Here we use p = n/2 processes, each holding a block of 2 elements.
 * The phase/stage structure of bitonic sort is unchanged; every
 * compare-EXCHANGE between two processes becomes a compare-SPLIT between
 * their 2-element blocks (merge the two sorted blocks, keep the appropriate
 * half).  Each process keeps its block internally sorted.
 *
 * We run the classic bitonic sort network over p "super-elements" (blocks).
 * The direction of each compare-split (ascending => lower rank keeps small
 * half; descending => lower rank keeps large half) follows the standard
 * bitonic sort rule based on stage bits.
 *
 * Test sizes: n=16 (p=8) and n=32 (p=16). Block size local_k = n/p = 2.
 *
 * Build:  mpicc -O2 -o bitonic_np2 bitonic_np2.c
 * Run:    mpirun -np 8  ./bitonic_np2     (n = 16)
 *         mpirun -np 16 ./bitonic_np2     (n = 32)
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static int cmp_int(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

/* compare-split: this process has 'mine' (sorted, length k); exchange with
 * partner, merge, keep low half if keep_low else high half. Returns #messages. */
static long compare_split(int *mine, int k, int partner, int keep_low) {
    int *other  = malloc((size_t)k * sizeof(int));
    int *merged = malloc((size_t)2 * k * sizeof(int));
    MPI_Sendrecv(mine, k, MPI_INT, partner, 0,
                 other, k, MPI_INT, partner, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    int i = 0, j = 0, m = 0;
    while (i < k && j < k) merged[m++] = (mine[i] <= other[j]) ? mine[i++] : other[j++];
    while (i < k) merged[m++] = mine[i++];
    while (j < k) merged[m++] = other[j++];
    if (keep_low) for (int t = 0; t < k; t++) mine[t] = merged[t];
    else          for (int t = 0; t < k; t++) mine[t] = merged[k + t];
    free(other); free(merged);
    return 1; /* one logical message pair from this side */
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    /* require p to be a power of 2 */
    if (p < 2 || (p & (p - 1))) {
        if (rank == 0) fprintf(stderr, "Number of processes must be a power of 2.\n");
        MPI_Finalize(); return 1;
    }

    int k = 2;                 /* each process holds 2 elements */
    int n = p * k;             /* total items */

    int *mine = malloc((size_t)k * sizeof(int));
    srand(1000 + rank);
    for (int i = 0; i < k; i++) mine[i] = rand() % 1000;
    qsort(mine, k, sizeof(int), cmp_int);

    long messages = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    int d = 0; while ((1 << d) < p) d++;   /* d = log2(p) */

    /* Standard bitonic sort over p blocks.
     * Outer loop: size of bitonic sequence being built = 2^(i+1).
     * Inner loop: compare distance = 2^j. */
    for (int i = 0; i < d; i++) {
        for (int j = i; j >= 0; j--) {
            int partner = rank ^ (1 << j);
            /* ascending if bit (i+1) of rank is 0, else descending */
            int ascending = ((rank >> (i + 1)) & 1) == 0;
            int keep_low;
            if (rank < partner) keep_low =  ascending; /* lower rank keeps small half when ascending */
            else                keep_low = !ascending;
            messages += compare_split(mine, k, partner, keep_low);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    /* gather all blocks at rank 0 and verify global sorted order */
    int *all = NULL;
    if (rank == 0) all = malloc((size_t)n * sizeof(int));
    MPI_Gather(mine, k, MPI_INT, all, k, MPI_INT, 0, MPI_COMM_WORLD);

    long total_messages = 0;
    MPI_Reduce(&messages, &total_messages, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int sorted = 1;
        for (int t = 1; t < n; t++) if (all[t] < all[t - 1]) { sorted = 0; break; }
        printf("BITONIC n/2-proc:  n=%d  p=%d  k=%d  time=%.6f s  messages=%ld  %s\n",
               n, p, k, t1 - t0, total_messages, sorted ? "SORTED" : "NOT SORTED");
        free(all);
    }

    free(mine);
    MPI_Finalize();
    return 0;
}
