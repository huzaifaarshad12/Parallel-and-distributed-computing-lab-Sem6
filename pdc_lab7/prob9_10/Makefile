# Problem 9.10 - block-based odd-even transposition sort
CC     = mpicc
CFLAGS = -O2 -Wall
all: oddeven
oddeven: oddeven.c
	$(CC) $(CFLAGS) -o oddeven oddeven.c
run: all
	for p in 2 4 8 ; do for n in 16 32 64 ; do \
	  mpirun --oversubscribe -np $$p ./oddeven $$n ; done ; done
clean:
	rm -f oddeven
.PHONY: all run clean
