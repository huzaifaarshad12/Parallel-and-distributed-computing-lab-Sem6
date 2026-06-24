# Problem 9.5 - hypercube bitonic merge
CC     = mpicc
CFLAGS = -O2 -Wall
all: hypercube_merge
hypercube_merge: hypercube_merge.c
	$(CC) $(CFLAGS) -o hypercube_merge hypercube_merge.c
run: all
	mpirun --oversubscribe -np 8 ./hypercube_merge
clean:
	rm -f hypercube_merge
.PHONY: all run clean
