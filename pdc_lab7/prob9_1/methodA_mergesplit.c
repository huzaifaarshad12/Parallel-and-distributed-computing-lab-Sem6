/* Problem 9.1 - Method A: Standard merge-split compare-split (MPI, 2 processes)
 *
 * Each of the two processes holds a sorted block of k random integers.
 * P0 keeps the smaller half, P1 keeps the larger half (a compare-split).
 *
 * Method A works by exchanging the WHOLE block once, merging the two sorted
 * blocks locally, and then each process keeps its half of the merged array.
 *
 * Messages: 2 (one each direction).  Bytes: 2 * k * sizeof(int).
 *
 * Build:  mpicc -O2 -o methodA methodA_mergesplit.c
 * Run:    mpirun -np 2 ./methodA <k>
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int cmp_int(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) fprintf(stderr, "This program requires exactly 2 processes.\n");
        MPI_Finalize();
        return 1;
    }

    int k = (argc > 1) ? atoi(argv[1]) : 1000;
    if (k <= 0) k = 1000;

    int *mine  = malloc((size_t)k * sizeof(int));
    int *other = malloc((size_t)k * sizeof(int));
    int *merged = malloc((size_t)2 * k * sizeof(int));

    /* Each process seeds differently so the data differs. */
    srand(12345 + rank * 777);
    for (int i = 0; i < k; i++) mine[i] = rand() % 100000;
    qsort(mine, k, sizeof(int), cmp_int);   /* blocks start sorted */

    int partner = 1 - rank;          /* the other process */
    long messages = 0, bytes = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    /* Exchange the entire block (one send + one recv). */
    MPI_Sendrecv(mine, k, MPI_INT, partner, 0,
                 other, k, MPI_INT, partner, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    messages += 1;                   /* this process sent one message */
    bytes += (long)k * sizeof(int);

    /* Merge two sorted arrays of length k into 'merged' (length 2k). */
    int i = 0, j = 0, m = 0;
    while (i < k && j < k) {
        if (mine[i] <= other[j]) merged[m++] = mine[i++];
        else                     merged[m++] = other[j++];
    }
    while (i < k) merged[m++] = mine[i++];
    while (j < k) merged[m++] = other[j++];

    /* Lower-rank process keeps the smaller half, higher-rank the larger half. */
    if (rank < partner) memcpy(mine, merged,         (size_t)k * sizeof(int));
    else                memcpy(mine, merged + k,     (size_t)k * sizeof(int));

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    /* --- verification: P0's max must be <= P1's min --- */
    int my_min = mine[0], my_max = mine[k - 1];
    int peer_min, peer_max;
    MPI_Sendrecv(&my_min, 1, MPI_INT, partner, 1, &peer_min, 1, MPI_INT, partner, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&my_max, 1, MPI_INT, partner, 2, &peer_max, 1, MPI_INT, partner, 2,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    long total_messages = 0, total_bytes = 0;
    MPI_Reduce(&messages, &total_messages, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&bytes,    &total_bytes,    1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int ok = (my_max <= peer_min); /* P0 max <= P1 min */
        printf("METHOD_A  k=%d  time=%.6f s  messages=%ld  bytes=%ld  %s\n",
               k, t1 - t0, total_messages, total_bytes, ok ? "PASS" : "FAIL");
    }

    free(mine); free(other); free(merged);
    MPI_Finalize();
    return 0;
}
