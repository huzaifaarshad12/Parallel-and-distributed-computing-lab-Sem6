# Problem 9.6 - full hypercube bitonic sort (Algorithm 9.1)
CC     = mpicc
CFLAGS = -O2 -Wall
all: bitonic_hypercube
bitonic_hypercube: bitonic_hypercube.c
	$(CC) $(CFLAGS) -o bitonic_hypercube bitonic_hypercube.c
run: all
	mpirun --oversubscribe -np 8  ./bitonic_hypercube
	mpirun --oversubscribe -np 16 ./bitonic_hypercube
	mpirun --oversubscribe -np 32 ./bitonic_hypercube
clean:
	rm -f bitonic_hypercube
.PHONY: all run clean
