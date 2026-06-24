# Problem 9.9 - bitonic sort on a ring
CC     = mpicc
CFLAGS = -O2 -Wall
LDLIBS = -lm
all: ring_bitonic block_ring
ring_bitonic: ring_bitonic.c
	$(CC) $(CFLAGS) -o ring_bitonic ring_bitonic.c $(LDLIBS)
block_ring: block_ring.c
	$(CC) $(CFLAGS) -o block_ring block_ring.c
run: all
	mpirun --oversubscribe -np 16 ./ring_bitonic
	for k in 4 8 16 ; do mpirun --oversubscribe -np 4 ./block_ring $$k ; done
clean:
	rm -f ring_bitonic block_ring
.PHONY: all run clean
