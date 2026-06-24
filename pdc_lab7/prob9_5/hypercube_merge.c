/* Problem 9.5 - Hypercube bitonic merge (MPI)
 *
 * A bitonic merge of a sequence of size 2^k is performed on a k-dimensional
 * hypercube: starting from a bitonic sequence, we do compare-exchange steps
 * along dimensions (k-1), (k-2), ..., 0.  At dimension j, process 'rank'
 * communicates with neighbour rank XOR (1<<j).  The process whose bit j is 0
 * keeps the smaller element (for an ascending merge); the other keeps larger.
 *
 * Each process holds one integer; the global input is a bitonic sequence
 * (we build one: first half ascending, second half descending). After the
 * merge the elements are globally sorted ascending.
 *
 * Build:  mpicc -O2 -o hypercube_merge hypercube_merge.c
 * Run:    mpirun -np 8 ./hypercube_merge      (k=3)
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    if (P < 2 || (P & (P - 1))) {
        if (rank == 0) fprintf(stderr, "P must be a power of 2 (e.g. 8 for k=3).\n");
        MPI_Finalize(); return 1;
    }
    int k = 0; while ((1 << k) < P) k++;     /* P = 2^k */

    /* Build a bitonic sequence across processes: ascending for first half of
     * ranks, descending for second half. We assign deterministic values. */
    srand(42);
    int *seed = malloc(P * sizeof(int));
    if (rank == 0) {
        int half = P / 2, v = 0;
        for (int i = 0; i < half; i++) { v += 1 + rand() % 9; seed[i] = v; }
        v += 50;
        for (int i = half; i < P; i++) { seed[i] = v; v -= 1 + rand() % 9; }
    }
    MPI_Bcast(seed, P, MPI_INT, 0, MPI_COMM_WORLD);
    int x = seed[rank];
    free(seed);

    /* record communication partners for printing */
    char partners[128] = {0};

    /* ascending bitonic merge: dimensions k-1 down to 0 */
    for (int j = k - 1; j >= 0; j--) {
        int partner = rank ^ (1 << j);
        int other;
        MPI_Sendrecv(&x, 1, MPI_INT, partner, 0, &other, 1, MPI_INT, partner, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        /* the process with bit j == 0 keeps the smaller (ascending) */
        int bit = (rank >> j) & 1;
        if (bit == 0) { if (other < x) x = other; }   /* keep min */
        else          { if (other > x) x = other; }   /* keep max */
        char buf[16];
        snprintf(buf, sizeof buf, "%s%d", partners[0] ? "->" : "", partner);
        strncat(partners, buf, sizeof(partners) - strlen(partners) - 1);
    }

    /* ordered printing of partner sequences and final element */
    for (int r = 0; r < P; r++) {
        if (r == rank)
            printf("rank %d  partners: %-20s  final=%d\n", rank, partners, x);
        MPI_Barrier(MPI_COMM_WORLD);
        fflush(stdout);
    }

    /* gather & verify ascending */
    int *all = NULL;
    if (rank == 0) all = malloc(P * sizeof(int));
    MPI_Gather(&x, 1, MPI_INT, all, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        int sorted = 1;
        for (int i = 1; i < P; i++) if (all[i] < all[i - 1]) sorted = 0;
        printf("merged result: ");
        for (int i = 0; i < P; i++) printf("%d ", all[i]);
        printf("\n%s\n", sorted ? "ASCENDING -> PASS" : "NOT SORTED -> FAIL");
        free(all);
    }

    MPI_Finalize();
    return 0;
}
