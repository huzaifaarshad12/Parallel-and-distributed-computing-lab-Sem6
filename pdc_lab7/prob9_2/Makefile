# Problem 9.2 - partial ordering verification (OpenMP)
CC     = gcc
CFLAGS = -O2 -Wall -fopenmp

all: partial_order

partial_order: partial_order.c
	$(CC) $(CFLAGS) -o partial_order partial_order.c

run: all
	OMP_NUM_THREADS=4 ./partial_order 100 20

clean:
	rm -f partial_order

.PHONY: all run clean
