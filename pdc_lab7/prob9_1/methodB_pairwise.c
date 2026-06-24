/* Problem 9.1 - Method B: Pairwise-exchange compare-split (MPI, 2 processes)
 *
 * P_i (lower rank) holds x_1 <= x_2 <= ... <= x_k  (ascending).
 * P_j (higher rank) holds y_1 >= y_2 >= ... >= y_k  (descending).
 *
 * For l = 1..k:  P_i sends x_l to P_j; P_j compares with y_l and sends the
 * LARGER back to P_i (keeping the smaller).  Because x is increasing and y is
 * decreasing, (x_l - y_l) is non-decreasing in l, so once x_l >= y_l no further
 * swaps are ever needed -> we may STOP EARLY at that l.
 *
 * Result: P_i accumulates the larger elements, P_j the smaller ones.  After
 * re-sorting, every element of P_j is <= every element of P_i.
 *
 * As the hint says, we send ONE element at a time (educational, not optimised).
 *
 * Build:  mpicc -O2 -o methodB methodB_pairwise.c
 * Run:    mpirun -np 2 ./methodB <k>
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static int cmp_asc(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}
static int cmp_desc(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (y > x) - (y < x);
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

    int *block = malloc((size_t)k * sizeof(int));
    srand(12345 + rank * 777);
    for (int i = 0; i < k; i++) block[i] = rand() % 100000;

    int partner = 1 - rank;
    int lower = (rank < partner);
    if (lower) qsort(block, k, sizeof(int), cmp_asc);
    else       qsort(block, k, sizeof(int), cmp_desc);

    long messages = 0, bytes = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    int stop = 0;
    for (int l = 0; l < k && !stop; l++) {
        if (lower) {
            int xl = block[l];
            MPI_Send(&xl, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
            messages++; bytes += sizeof(int);
            int larger;
            MPI_Recv(&larger, 1, MPI_INT, partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            block[l] = larger;
            MPI_Recv(&stop, 1, MPI_INT, partner, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            int xl;
            MPI_Recv(&xl, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int yl = block[l];
            int larger  = (xl > yl) ? xl : yl;
            int smaller = (xl > yl) ? yl : xl;
            block[l] = smaller;
            MPI_Send(&larger, 1, MPI_INT, partner, 1, MPI_COMM_WORLD);
            messages++; bytes += sizeof(int);
            stop = (xl >= yl) ? 1 : 0;
            MPI_Send(&stop, 1, MPI_INT, partner, 2, MPI_COMM_WORLD);
            messages++; bytes += sizeof(int);
        }
    }

    qsort(block, k, sizeof(int), cmp_asc);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    int my_min = block[0], my_max = block[k - 1];
    int peer_min, peer_max;
    MPI_Sendrecv(&my_min, 1, MPI_INT, partner, 5, &peer_min, 1, MPI_INT, partner, 5,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&my_max, 1, MPI_INT, partner, 6, &peer_max, 1, MPI_INT, partner, 6,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    long total_messages = 0, total_bytes = 0;
    MPI_Reduce(&messages, &total_messages, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&bytes,    &total_bytes,    1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (lower) {
        int ok = (my_min >= peer_max);
        printf("METHOD_B  k=%d  time=%.6f s  messages=%ld  bytes=%ld  %s\n",
               k, t1 - t0, total_messages, total_bytes, ok ? "PASS" : "FAIL");
    }

    free(block);
    MPI_Finalize();
    return 0;
}
