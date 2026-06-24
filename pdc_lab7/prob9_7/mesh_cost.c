/* Problem 9.7 - Mesh bitonic sort: communication-cost simulation (MPI).
 *
 * We do NOT actually sort. We simulate the communication pattern of bitonic
 * sort over n processes laid out on a sqrt(n) x sqrt(n) mesh under three
 * mappings, and total the hop counts.
 *
 * Bitonic sort has log2(n)*(log2(n)+1)/2 stages; at each stage every process
 * 'r' communicates with r XOR (1<<j). We map each hypercube label r to a mesh
 * coordinate (row,col) by one of:
 *   (a) row-major:           index = r;   row=index/c, col=index%c
 *   (b) row-major snakelike:  odd rows reversed
 *   (c) row-major shuffled:   bits of r interleaved (perfect-shuffle) before
 *                             row-major placement
 * Manhattan distance between mapped coordinates = number of hops.
 *
 * Costs (t_s startup, t_w per-hop):
 *   store-and-forward:  sum over messages of (t_s + t_w * hops)
 *   cut-through:        sum over messages of (t_s + t_w)        [per problem hint]
 *
 * The work is partitioned across MPI ranks by stage, then reduced.
 *
 * Build:  mpicc -O2 -o mesh_cost mesh_cost.c -lm
 * Run:    mpirun -np 4 ./mesh_cost 16
 *         mpirun -np 4 ./mesh_cost 64
 *         mpirun -np 4 ./mesh_cost 256
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static const double TS = 1.0;   /* startup latency (arbitrary units) */
static const double TW = 0.1;   /* per-hop transfer time */

/* map hypercube label -> (row,col) for a c x c mesh, c = sqrt(n) */
static void map_coord(int label, int c, int mapping, int *row, int *col) {
    int idx;
    if (mapping == 2) {
        /* perfect-shuffle / shuffled row-major: reverse the bit order of label */
        int bits = 0; while ((1 << bits) < c * c) bits++;
        int s = 0;
        for (int b = 0; b < bits; b++) if (label & (1 << b)) s |= 1 << (bits - 1 - b);
        idx = s;
    } else {
        idx = label;     /* row-major and snakelike share base index */
    }
    int r = idx / c, co = idx % c;
    if (mapping == 1 && (r & 1)) co = c - 1 - co;   /* snakelike: reverse odd rows */
    *row = r; *col = co;
}

static int manhattan(int a, int b, int c, int mapping) {
    int ar, ac, br, bc;
    map_coord(a, c, mapping, &ar, &ac);
    map_coord(b, c, mapping, &br, &bc);
    return abs(ar - br) + abs(ac - bc);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    int n = (argc > 1) ? atoi(argv[1]) : 16;
    int c = (int)(sqrt((double)n) + 0.5);
    if (c * c != n || (n & (n - 1))) {
        if (rank == 0) fprintf(stderr, "n must be a power of 2 AND a perfect square (16,64,256).\n");
        MPI_Finalize(); return 1;
    }
    int d = 0; while ((1 << d) < n) d++;     /* log2 n */

    /* enumerate (stage j, phase i) pairs; partition stages across ranks */
    /* For totals we just iterate all stages; each contributes pairs over all r. */
    const char *names[3] = {"row-major", "snakelike", "shuffled"};

    for (int mapping = 0; mapping < 3; mapping++) {
        double saf_local = 0.0, ct_local = 0.0;
        long   hops_local = 0, msgs_local = 0;

        /* iterate phases i=0..d-1, stages j=i..0 */
        int stage_id = 0;
        for (int i = 0; i < d; i++) {
            for (int j = i; j >= 0; j--) {
                /* distribute stages round-robin among ranks */
                if (stage_id % P == rank) {
                    for (int r = 0; r < n; r++) {
                        int partner = r ^ (1 << j);
                        if (r < partner) {           /* count each pair once */
                            int hops = manhattan(r, partner, c, mapping);
                            saf_local += TS + TW * hops;
                            ct_local  += TS + TW;
                            hops_local += hops;
                            msgs_local += 1;
                        }
                    }
                }
                stage_id++;
            }
        }

        double saf, ct; long hops, msgs;
        MPI_Reduce(&saf_local,  &saf,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&ct_local,   &ct,   1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&hops_local, &hops, 1, MPI_LONG,   MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&msgs_local, &msgs, 1, MPI_LONG,   MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("n=%-4d mesh=%dx%d  mapping=%-10s  msgs=%ld  totalHops=%ld  "
                   "SAF=%.2f  CT=%.2f\n",
                   n, c, c, names[mapping], msgs, hops, saf, ct);
        }
    }
    if (rank == 0) printf("(t_s=%.1f, t_w=%.1f; SAF=store-and-forward, CT=cut-through)\n\n", TS, TW);

    MPI_Finalize();
    return 0;
}
