CC     = gcc
MPICC  = mpicc
CFLAGS = -O2 -Wall
OMPFLAGS = -fopenmp

TARGETS = sequential/quicksort_seq \
          openmp/quicksort_omp \
          mpi/quicksort_mpi

all: $(TARGETS)

sequential/quicksort_seq: sequential/quicksort_seq.c
	$(CC) $(CFLAGS) -o $@ $<

openmp/quicksort_omp: openmp/quicksort_omp.c
	$(CC) $(CFLAGS) $(OMPFLAGS) -o $@ $<

mpi/quicksort_mpi: mpi/quicksort_mpi.c
	$(MPICC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean
