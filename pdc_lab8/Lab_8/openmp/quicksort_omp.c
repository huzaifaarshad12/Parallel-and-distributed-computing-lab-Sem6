/*
 * OpenMP Parallel Quick Sort
 * Lab 8 - PDC Lab
 * Student: 2023-CS-77
 *
 * Strategy:
 *   - Divide-and-conquer: after each partition, spawn two OpenMP tasks
 *     for the left and right sub-arrays (task-parallel quicksort).
 *   - A cutoff threshold (CUTOFF) switches to sequential sort for
 *     small sub-arrays to avoid task-creation overhead.
 *   - Thread count is read from OMP_NUM_THREADS or the command line.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>

#define CUTOFF 100000   /* sub-arrays smaller than this go sequential */

/* ------------------------------------------------------------------ */
/*  Utility                                                             */
/* ------------------------------------------------------------------ */

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
    swap(&arr[i], &arr[hi]);
    return i;
}

/* ------------------------------------------------------------------ */
/*  Sequential quicksort (used below the cutoff)                       */
/* ------------------------------------------------------------------ */

static void quicksort_seq(int *arr, int lo, int hi) {
    if (lo >= hi) return;
    int p = partition(arr, lo, hi);
    quicksort_seq(arr, lo, p - 1);
    quicksort_seq(arr, p + 1, hi);
}

/* ------------------------------------------------------------------ */
/*  Task-parallel quicksort                                             */
/* ------------------------------------------------------------------ */

static void quicksort_omp(int *arr, int lo, int hi) {
    if (lo >= hi) return;

    /* Switch to sequential for small ranges */
    if ((hi - lo + 1) <= CUTOFF) {
        quicksort_seq(arr, lo, hi);
        return;
    }

    int p = partition(arr, lo, hi);

    #pragma omp task shared(arr) firstprivate(lo, p)
    quicksort_omp(arr, lo, p - 1);

    #pragma omp task shared(arr) firstprivate(p, hi)
    quicksort_omp(arr, p + 1, hi);

    #pragma omp taskwait
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
    int n        = (argc > 1) ? atoi(argv[1]) : 10000000;
    int nthreads = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();

    omp_set_num_threads(nthreads);

    printf("=== OpenMP Parallel Quick Sort ===\n");
    printf("Array size : %d\n", n);
    printf("Threads    : %d\n", nthreads);
    printf("Cutoff     : %d\n", CUTOFF);

    int *arr = (int *)malloc((size_t)n * sizeof(int));
    if (!arr) { perror("malloc"); return 1; }

    srand(42);
    for (int i = 0; i < n; i++) arr[i] = rand();

    double t0 = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single nowait
        quicksort_omp(arr, 0, n - 1);
    }

    double t1 = omp_get_wtime();
    double elapsed = t1 - t0;

    printf("Sorted     : %s\n", is_sorted(arr, n) ? "YES" : "NO");
    printf("Time       : %.6f seconds\n", elapsed);

    free(arr);
    return 0;
}
