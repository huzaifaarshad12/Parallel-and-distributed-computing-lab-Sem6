# Problem 9.3 - bitonic split properties (OpenMP)
CC     = gcc
CFLAGS = -O2 -Wall -fopenmp

all: bitonic_split
bitonic_split: bitonic_split.c
	$(CC) $(CFLAGS) -o bitonic_split bitonic_split.c
run: all
	OMP_NUM_THREADS=4 ./bitonic_split 16
clean:
	rm -f bitonic_split
.PHONY: all run clean
