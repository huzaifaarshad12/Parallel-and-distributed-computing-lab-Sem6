# Problem 9.4 - bitonic sort with n/2 processes vs n processes
CC     = mpicc
CFLAGS = -O2 -Wall
all: bitonic_np2 bitonic_n
bitonic_np2: bitonic_np2.c
	$(CC) $(CFLAGS) -o bitonic_np2 bitonic_np2.c
bitonic_n: bitonic_n.c
	$(CC) $(CFLAGS) -o bitonic_n bitonic_n.c
run: all
	mpirun --oversubscribe -np 8  ./bitonic_np2
	mpirun --oversubscribe -np 16 ./bitonic_n
	mpirun --oversubscribe -np 16 ./bitonic_np2
	mpirun --oversubscribe -np 32 ./bitonic_n
clean:
	rm -f bitonic_np2 bitonic_n
.PHONY: all run clean
