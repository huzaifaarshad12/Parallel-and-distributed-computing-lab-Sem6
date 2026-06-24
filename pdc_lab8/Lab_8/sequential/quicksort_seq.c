/*
 * Sequential Quick Sort
 * Lab 8 - PDC Lab
 * Student: 2023-CS-77
 *
 * Implements standard recursive quicksort with median-of-three pivot selection.
 * Measures wall-clock time using clock_gettime (CLOCK_MONOTONIC).
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Utility                                                             */
/* ------------------------------------------------------------------ */

static double wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void swap(int *a, int *b) {
    int tmp = *a; *a = *b; *b = tmp;
}

/* ------------------------------------------------------------------ */
/*  Median-of-three pivot                                              */
/* ------------------------------------------------------------------ */

static int median3(int *arr, int lo, int hi) {
    int mid = lo + (hi - lo) / 2;
    if (arr[lo] > arr[mid]) swap(&arr[lo], &arr[mid]);
    if (arr[lo] > arr[hi])  swap(&arr[lo], &arr[hi]);
    if (arr[mid] > arr[hi]) swap(&arr[mid], &arr[hi]);
    /* median is now at mid; move it to hi-1 as the pivot */
    swap(&arr[mid], &arr[hi]);
    return arr[hi];
}

/* ------------------------------------------------------------------ */
/*  Partition                                                           */
/* ------------------------------------------------------------------ */

static int partition(int *arr, int lo, int hi) {
    int pivot = median3(arr, lo, hi);
    int i = lo - 1, j = hi;
    for (;;) {
        while (arr[++i] < pivot);
        while (j > lo && arr[--j] > pivot);
        if (i >= j) break;
        swap(&arr[i], &arr[j]);
    }
    swap(&arr[i], &arr[hi]);   /* restore pivot */
    return i;
}

/* ------------------------------------------------------------------ */
/*  Recursive quicksort                                                 */
/* ------------------------------------------------------------------ */

static void quicksort(int *arr, int lo, int hi) {
    if (lo >= hi) return;
    int p = partition(arr, lo, hi);
    quicksort(arr, lo, p - 1);
    quicksort(arr, p + 1, hi);
}

/* ------------------------------------------------------------------ */
/*  Verification                                                        */
/* ------------------------------------------------------------------ */

static int is_sorted(const int *arr, int n) {
    for (int i = 1; i < n; i++)
        if (arr[i] < arr[i-1]) return 0;
    return 1;
}

/* ------------------------------------------------------------------ */
/*  Main                                                                */
/* ------------------------------------------------------------------ */

int main(int argc, char *argv[]) {
    int n = (argc > 1) ? atoi(argv[1]) : 10000000;   /* default 10 M */

    printf("=== Sequential Quick Sort ===\n");
    printf("Array size : %d\n", n);

    /* Allocate and fill with random data */
    int *arr = (int *)malloc((size_t)n * sizeof(int));
    if (!arr) { perror("malloc"); return 1; }

    srand(42);
    for (int i = 0; i < n; i++) arr[i] = rand();

    double t0 = wall_time();
    quicksort(arr, 0, n - 1);
    double t1 = wall_time();

    double elapsed = t1 - t0;
    printf("Sorted     : %s\n", is_sorted(arr, n) ? "YES" : "NO");
    printf("Time       : %.6f seconds\n", elapsed);

    free(arr);
    return 0;
}
