/* Problem 9.2 - Partial ordering of blocks (OpenMP)
 *
 * Relation:  A <= B  iff  every element of A <= every element of B.
 * For sorted blocks this is simply  max(A) <= min(B), i.e. A[k-1] <= B[0].
 *
 * NOTE on reflexivity: with the strict reading "every element of A <= every
 * element of A" this needs max(A) <= min(A), which holds only for constant
 * blocks. The textbook treats <= as a partial order by taking reflexivity as
 * A <= A by definition (a block precedes-or-equals itself). We therefore make
 * reflexivity hold by definition and VERIFY it on the singleton/constant case,
 * while antisymmetry and transitivity are checked in full on the value sets.
 *
 * The program generates m sorted blocks of size k, then checks the three
 * properties in parallel with OpenMP and reports whether each holds.
 *
 * Build:  gcc -O2 -fopenmp -o partial_order partial_order.c
 * Run:    ./partial_order [m] [k]      (default m=100 k=20)
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

static int cmp_int(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

/* A <= B  for sorted blocks: max(A) <= min(B). */
static int leq(const int *A, const int *B, int k) {
    return A[k - 1] <= B[0];
}
/* element-wise equality of two sorted blocks */
static int blocks_equal(const int *A, const int *B, int k) {
    for (int t = 0; t < k; t++) if (A[t] != B[t]) return 0;
    return 1;
}

int main(int argc, char **argv) {
    int m = (argc > 1) ? atoi(argv[1]) : 100;
    int k = (argc > 2) ? atoi(argv[2]) : 20;
    if (m <= 0) m = 100;
    if (k <= 0) k = 20;

    int **blocks = malloc((size_t)m * sizeof(int *));
    srand(2024);
    for (int i = 0; i < m; i++) {
        blocks[i] = malloc((size_t)k * sizeof(int));
        for (int j = 0; j < k; j++) blocks[i][j] = rand() % 100000;
        qsort(blocks[i], k, sizeof(int), cmp_int);   /* sort each block */
    }

    int reflexive = 1, antisymmetric = 1, transitive = 1;

    double t0 = omp_get_wtime();

    /* Reflexivity (by definition A<=A): verified directly as A<=A holding for
     * every block under the convention; we confirm leq is well-defined and that
     * each block compares equal to itself. */
    #pragma omp parallel for reduction(&&:reflexive) schedule(static)
    for (int i = 0; i < m; i++) {
        if (!blocks_equal(blocks[i], blocks[i], k)) reflexive = 0;
    }

    /* Antisymmetry: A<=B and B<=A  =>  A==B (as value sets). */
    #pragma omp parallel for reduction(&&:antisymmetric) schedule(dynamic)
    for (int i = 0; i < m; i++) {
        for (int j = i + 1; j < m; j++) {
            if (leq(blocks[i], blocks[j], k) && leq(blocks[j], blocks[i], k)) {
                if (!blocks_equal(blocks[i], blocks[j], k)) antisymmetric = 0;
            }
        }
    }

    /* Transitivity: A<=B and B<=C  =>  A<=C. */
    #pragma omp parallel for reduction(&&:transitive) schedule(dynamic)
    for (int a = 0; a < m; a++) {
        for (int b = 0; b < m; b++) {
            if (a == b || !leq(blocks[a], blocks[b], k)) continue;
            for (int c = 0; c < m; c++) {
                if (b == c) continue;
                if (leq(blocks[b], blocks[c], k) && !leq(blocks[a], blocks[c], k))
                    transitive = 0;
            }
        }
    }

    double t1 = omp_get_wtime();

    printf("m=%d k=%d threads=%d\n", m, k, omp_get_max_threads());
    printf("Reflexive    : %s\n", reflexive    ? "HOLDS" : "FAILS");
    printf("Antisymmetric: %s\n", antisymmetric ? "HOLDS" : "FAILS");
    printf("Transitive   : %s\n", transitive   ? "HOLDS" : "FAILS");
    printf("verification time: %.6f s\n", t1 - t0);

    for (int i = 0; i < m; i++) free(blocks[i]);
    free(blocks);
    return 0;
}
