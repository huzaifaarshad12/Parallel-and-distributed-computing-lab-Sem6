/* Problem 9.9 - Bitonic sort on a ring (MPI, 1-D Cartesian, periodic).
 *
 * Part 1 (mapping cost): For n=16 processes on a ring, run the bitonic sort
 * communication pattern and total the hop distances (ring distance =
 * min(|a-b|, n-|a-b|)) for three mappings of bitonic labels onto ring
 * positions:
 *    - optimal   : Gray-code mapping (adjacent labels differ by 1 bit and are
 *                  placed at adjacent ring positions where possible)
 *    - consecutive: identity (label = ring position)
 *    - random    : a fixed random permutation
 *
 * Part 2 (block weak-scaling) is driven by block_ring.c separately; this file
 * focuses on the mapping comparison required by the problem.
 *
 * Build:  mpicc -O2 -o ring_bitonic ring_bitonic.c -lm
 * Run:    mpirun -np 16 ./ring_bitonic
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static int ring_dist(int a, int b, int n) {
    int d = abs(a - b);
    return (d < n - d) ? d : n - d;
}

/* binary-reflected Gray code of x */
static int gray(int x) { return x ^ (x >> 1); }

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, n;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n);

    if (n < 2 || (n & (n - 1))) {
        if (rank == 0) fprintf(stderr, "n must be a power of 2 (use 16).\n");
        MPI_Finalize(); return 1;
    }

    /* create a periodic 1-D Cartesian (ring) topology */
    int dims[1] = {n}, periods[1] = {1};
    MPI_Comm ring;
    MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, 0, &ring);

    int d = 0; while ((1 << d) < n) d++;

    /* Build the three label->position maps on rank 0, broadcast. */
    int *pos_opt = malloc(n * sizeof(int));  /* position of label i */
    int *pos_con = malloc(n * sizeof(int));
    int *pos_rnd = malloc(n * sizeof(int));
    if (rank == 0) {
        for (int i = 0; i < n; i++) {
            pos_con[i] = i;                 /* consecutive */
            pos_opt[gray(i)] = i;           /* Gray: label gray(i) at position i */
        }
        for (int i = 0; i < n; i++) pos_rnd[i] = i;
        srand(99);
        for (int i = n - 1; i > 0; i--) { int j = rand() % (i + 1); int t = pos_rnd[i]; pos_rnd[i] = pos_rnd[j]; pos_rnd[j] = t; }
    }
    MPI_Bcast(pos_opt, n, MPI_INT, 0, ring);
    MPI_Bcast(pos_con, n, MPI_INT, 0, ring);
    MPI_Bcast(pos_rnd, n, MPI_INT, 0, ring);

    const char *names[3] = {"optimal(Gray)", "consecutive", "random"};
    int *maps[3] = {pos_opt, pos_con, pos_rnd};

    if (rank == 0) {
        for (int m = 0; m < 3; m++) {
            long hops = 0, msgs = 0;
            for (int i = 0; i < d; i++) {
                for (int j = i; j >= 0; j--) {
                    for (int a = 0; a < n; a++) {
                        int b = a ^ (1 << j);
                        if (a < b) {
                            hops += ring_dist(maps[m][a], maps[m][b], n);
                            msgs++;
                        }
                    }
                }
            }
            printf("mapping=%-14s  messages=%ld  total_ring_hops=%ld\n",
                   names[m], msgs, hops);
        }
    }

    free(pos_opt); free(pos_con); free(pos_rnd);
    MPI_Comm_free(&ring);
    MPI_Finalize();
    return 0;
}
