/* Problem 9.3 - Bitonic split properties (C / OpenMP)
 *
 * A bitonic sequence s[0..n-1] (n a power of 2) is split into s1, s2 by
 * comparing s[i] with s[i + n/2] for 0 <= i < n/2:
 *     s1[i] = min(s[i], s[i+n/2])   (the "low" half)
 *     s2[i] = max(s[i], s[i+n/2])   (the "high" half)
 * Properties to verify: (1) s1 and s2 are each bitonic, and
 *                       (2) every element of s1 <= every element of s2.
 *
 * Three bitonic input types are generated:
 *   (a) increasing then decreasing (classic)
 *   (b) increasing up to index i, then decreasing (random pivot)
 *   (c) a cyclic shift of an increasing-decreasing bitonic sequence
 *
 * Build:  gcc -O2 -fopenmp -o bitonic_split bitonic_split.c
 * Run:    ./bitonic_split [n]          (default n=16; n must be power of 2)
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

static void print_seq(const char *label, const int *a, int n) {
    printf("%s", label);
    for (int i = 0; i < n; i++) printf("%5d", a[i]);
    printf("\n");
}

/* Is a sequence bitonic? It is bitonic iff it has at most one "local max then
 * local min" structure allowing one wraparound: i.e. the number of times the
 * direction changes (up<->down), counting the cyclic boundary, is <= 2. */
static int is_bitonic(const int *a, int n) {
    if (n <= 2) return 1;
    int changes = 0;
    int prev = 0; /* 0 unknown, 1 up, -1 down */
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        int dir = (a[j] > a[i]) ? 1 : (a[j] < a[i]) ? -1 : 0;
        if (dir != 0) {
            if (prev != 0 && dir != prev) changes++;
            prev = dir;
        }
    }
    /* a cyclic bitonic sequence has at most 2 direction changes around the ring */
    return changes <= 2;
}

/* generate three kinds of bitonic sequences */
static void gen_type_a(int *a, int n) {           /* up then down */
    int half = n / 2, v = 0;
    for (int i = 0; i < half; i++)  { v += 1 + rand() % 5; a[i] = v; }
    for (int i = half; i < n; i++)  { v -= 1 + rand() % 5; a[i] = v; }
}
static void gen_type_b(int *a, int n) {           /* up to random i, then down */
    int piv = 1 + rand() % (n - 1), v = 0;
    for (int i = 0; i <= piv; i++)  { v += 1 + rand() % 5; a[i] = v; }
    for (int i = piv + 1; i < n; i++){ v -= 1 + rand() % 5; a[i] = v; }
}
static void gen_type_c(int *a, int n) {           /* cyclic shift of type a */
    int *tmp = malloc(n * sizeof(int));
    gen_type_a(tmp, n);
    int shift = 1 + rand() % (n - 1);
    for (int i = 0; i < n; i++) a[i] = tmp[(i + shift) % n];
    free(tmp);
}

static int run_case(const char *name, void (*gen)(int *, int), int n) {
    int *s  = malloc(n * sizeof(int));
    int *s1 = malloc((n / 2) * sizeof(int));
    int *s2 = malloc((n / 2) * sizeof(int));
    gen(s, n);

    /* bitonic split, n/2 comparisons in parallel */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n / 2; i++) {
        int lo = s[i], hi = s[i + n / 2];
        if (lo > hi) { int t = lo; lo = hi; hi = t; }
        s1[i] = lo;       /* low half  */
        s2[i] = hi;       /* high half */
    }

    /* verify property 2: max(s1) <= min(s2) (in parallel via reductions) */
    int max1 = s1[0], min2 = s2[0];
    #pragma omp parallel for reduction(max:max1) reduction(min:min2)
    for (int i = 0; i < n / 2; i++) {
        if (s1[i] > max1) max1 = s1[i];
        if (s2[i] < min2) min2 = s2[i];
    }
    int prop2 = (max1 <= min2);
    int prop1 = is_bitonic(s1, n / 2) && is_bitonic(s2, n / 2);

    printf("---- case %s (n=%d) ----\n", name, n);
    print_seq("  s  : ", s,  n);
    print_seq("  s1 : ", s1, n / 2);
    print_seq("  s2 : ", s2, n / 2);
    int pass = prop1 && prop2;
    printf("  s1,s2 bitonic: %s   max(s1)<=min(s2): %s   ==> %s\n\n",
           prop1 ? "yes" : "no", prop2 ? "yes" : "no", pass ? "PASS" : "FAIL");

    free(s); free(s1); free(s2);
    return pass;
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 16;
    if (n < 2 || (n & (n - 1))) { fprintf(stderr, "n must be a power of 2 >= 2\n"); return 1; }
    srand(7);
    int ok = 1;
    ok &= run_case("a: increasing-decreasing", gen_type_a, n);
    ok &= run_case("b: increasing-to-pivot-then-decreasing", gen_type_b, n);
    ok &= run_case("c: cyclic shift of a bitonic sequence", gen_type_c, n);
    printf("OVERALL: %s\n", ok ? "ALL PASS" : "SOME FAIL");
    return ok ? 0 : 1;
}
