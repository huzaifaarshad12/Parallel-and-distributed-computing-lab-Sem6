/*
 * MPI Parallel Quick Sort  (Hyperquicksort / Pivot-broadcast approach)
 * Lab 8 - PDC Lab
 * Student: 2023-CS-77
 *
 * Algorithm:
 *   1. Process 0 generates the full array, then distributes it evenly.
 *   2. Each process locally sorts its chunk with qsort().
 *   3. Iterative log2(p) rounds:
 *        - Process 0 (of the active group) broadcasts a pivot
 *          (median of its local chunk).
 *        - Each process splits its local data around the pivot.
 *        - Partners exchange data: lower-half processes keep values < pivot;
 *          upper-half processes keep values >= pivot.
 *   4. Each process locally sorts its final partition.
 *   5. All chunks are gathered to process 0 for output/verification.
 *
 * This works correctly for p = power-of-2 process counts.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

/* ------------------------------------------------------------------ */
/*  Comparison function for qsort                                       */
/* ------------------------------------------------------------------ */

static int cmp_int(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

/* ------------------------------------------------------------------ */
/*  Split local array around pivot                                      */
/*  Returns: number of elements < pivot (written to low[])             */
/* ------------------------------------------------------------------ */

static int split_array(const int *arr, int n, int pivot,
                       int **low_out, int *low_cap,
                       int **high_out, int *high_cap) {
    /* Count */
    int nlow = 0;
    for (int i = 0; i < n; i++)
        if (arr[i] < pivot) nlow++;
    int nhigh = n - nlow;

    /* (Re)allocate if needed */
    if (nlow > *low_cap) {
        free(*low_out);
        *low_out  = (int *)malloc((size_t)nlow  * sizeof(int));
        *low_cap  = nlow;
    }
    if (nhigh > *high_cap) {
        free(*high_out);
        *high_out = (int *)malloc((size_t)nhigh * sizeof(int));
        *high_cap = nhigh;
    }

    int li = 0, hi = 0;
    for (int i = 0; i < n; i++) {
        if (arr[i] < pivot) (*low_out)[li++] = arr[i];
        else                (*high_out)[hi++] = arr[i];
    }
    return nlow;
}

/* ------------------------------------------------------------------ */
/*  Main                                                                */
/* ------------------------------------------------------------------ */

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int n = (argc > 1) ? atoi(argv[1]) : 10000000;

    /* ---- Process 0: generate data and scatter ---- */
    int *global_arr = NULL;
    int local_n     = n / nprocs;   /* elements per process (approx) */

    /* Adjust for n not divisible by nprocs */
    int *send_counts = (int *)malloc(nprocs * sizeof(int));
    int *displs      = (int *)malloc(nprocs * sizeof(int));
    int rem = n % nprocs;
    for (int i = 0; i < nprocs; i++) {
        send_counts[i] = n / nprocs + (i < rem ? 1 : 0);
        displs[i]      = (i == 0) ? 0 : displs[i-1] + send_counts[i-1];
    }
    local_n = send_counts[rank];

    if (rank == 0) {
        global_arr = (int *)malloc((size_t)n * sizeof(int));
        if (!global_arr) { perror("malloc global"); MPI_Abort(MPI_COMM_WORLD, 1); }
        srand(42);
        for (int i = 0; i < n; i++) global_arr[i] = rand();
        printf("=== MPI Parallel Quick Sort ===\n");
        printf("Array size : %d\n", n);
        printf("Processes  : %d\n", nprocs);
        fflush(stdout);
    }

    int *local_arr = (int *)malloc((size_t)local_n * sizeof(int));
    if (!local_arr) { perror("malloc local"); MPI_Abort(MPI_COMM_WORLD, 1); }

    MPI_Scatterv(global_arr, send_counts, displs, MPI_INT,
                 local_arr,  local_n,              MPI_INT,
                 0, MPI_COMM_WORLD);

    double t0 = MPI_Wtime();

    /* ---- Local sort ---- */
    qsort(local_arr, local_n, sizeof(int), cmp_int);

    /* ---- Iterative hyperquicksort ---- */
    int *low_buf  = NULL, low_cap  = 0;
    int *high_buf = NULL, high_cap = 0;
    int *recv_buf = NULL; int recv_cap = 0;

    MPI_Comm comm = MPI_COMM_WORLD;

    for (int step = nprocs; step > 1; step >>= 1) {
        int comm_rank, comm_size;
        MPI_Comm_rank(comm, &comm_rank);
        MPI_Comm_size(comm, &comm_size);

        /* Root of this communicator broadcasts its median as pivot */
        int pivot = 0;
        if (comm_rank == 0) {
            pivot = local_arr[local_n / 2];   /* median of sorted local chunk */
        }
        MPI_Bcast(&pivot, 1, MPI_INT, 0, comm);

        /* Split local data */
        int nlow = split_array(local_arr, local_n, pivot,
                               &low_buf, &low_cap,
                               &high_buf, &high_cap);
        int nhigh = local_n - nlow;

        /* Partner exchange */
        int half     = comm_size / 2;
        int is_lower = (comm_rank < half);
        int partner  = is_lower ? (comm_rank + half) : (comm_rank - half);

        /* Exchange sizes first */
        int send_n = is_lower ? nhigh : nlow;
        int recv_n = 0;
        MPI_Sendrecv(&send_n, 1, MPI_INT, partner, 0,
                     &recv_n,  1, MPI_INT, partner, 0,
                     comm, MPI_STATUS_IGNORE);

        if (recv_n > recv_cap) {
            free(recv_buf);
            recv_buf = (int *)malloc((size_t)recv_n * sizeof(int));
            recv_cap = recv_n;
        }

        int *send_ptr = is_lower ? high_buf : low_buf;
        MPI_Sendrecv(send_ptr, send_n, MPI_INT, partner, 1,
                     recv_buf, recv_n, MPI_INT, partner, 1,
                     comm, MPI_STATUS_IGNORE);

        /* Merge kept half with received data */
        int kept_n   = is_lower ? nlow  : nhigh;
        int *kept_ptr = is_lower ? low_buf : high_buf;

        /* Merge two sorted arrays */
        int new_n = kept_n + recv_n;
        int *new_arr = (int *)malloc((size_t)new_n * sizeof(int));
        if (!new_arr) { perror("malloc merge"); MPI_Abort(comm, 1); }

        /* Simple merge of two sorted arrays */
        int ai = 0, bi = 0, ci = 0;
        while (ai < kept_n && bi < recv_n) {
            if (kept_ptr[ai] <= recv_buf[bi]) new_arr[ci++] = kept_ptr[ai++];
            else                               new_arr[ci++] = recv_buf[bi++];
        }
        while (ai < kept_n) new_arr[ci++] = kept_ptr[ai++];
        while (bi < recv_n) new_arr[ci++] = recv_buf[bi++];

        free(local_arr);
        local_arr = new_arr;
        local_n   = new_n;

        /* Split communicator */
        MPI_Comm new_comm;
        MPI_Comm_split(comm, is_lower ? 0 : 1, comm_rank, &new_comm);
        if (comm != MPI_COMM_WORLD) MPI_Comm_free(&comm);
        comm = new_comm;
    }
    if (comm != MPI_COMM_WORLD) MPI_Comm_free(&comm);

    /* Final local sort (optional, data should already be sorted) */
    qsort(local_arr, local_n, sizeof(int), cmp_int);

    double t1 = MPI_Wtime();

    /* ---- Gather results to rank 0 ---- */
    int *recv_counts2 = NULL, *displs2 = NULL;
    int *sorted_arr   = NULL;

    if (rank == 0) {
        recv_counts2 = (int *)malloc(nprocs * sizeof(int));
        displs2      = (int *)malloc(nprocs * sizeof(int));
    }

    MPI_Gather(&local_n, 1, MPI_INT,
               recv_counts2, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        displs2[0] = 0;
        for (int i = 1; i < nprocs; i++)
            displs2[i] = displs2[i-1] + recv_counts2[i-1];
        sorted_arr = (int *)malloc((size_t)n * sizeof(int));
    }

    MPI_Gatherv(local_arr, local_n, MPI_INT,
                sorted_arr, recv_counts2, displs2, MPI_INT,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        /* Verify global sorted order */
        int ok = 1;
        for (int i = 1; i < n; i++)
            if (sorted_arr[i] < sorted_arr[i-1]) { ok = 0; break; }

        printf("Sorted     : %s\n", ok ? "YES" : "NO");
        printf("Time       : %.6f seconds\n", t1 - t0);

        free(sorted_arr);
        free(recv_counts2);
        free(displs2);
        free(global_arr);
    }

    free(local_arr);
    free(low_buf);
    free(high_buf);
    free(recv_buf);
    free(send_counts);
    free(displs);

    MPI_Finalize();
    return 0;
}
