# Problem 9.8 - block-based bitonic sort
CC     = mpicc
CFLAGS = -O2 -Wall
all: block_bitonic
block_bitonic: block_bitonic.c
	$(CC) $(CFLAGS) -o block_bitonic block_bitonic.c
run: all
	for p in 2 4 8 ; do for n in 16 32 64 ; do \
	  mpirun --oversubscribe -np $$p ./block_bitonic $$n ; done ; done
clean:
	rm -f block_bitonic
.PHONY: all run clean
