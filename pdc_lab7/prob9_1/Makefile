# Problem 9.1 - compare-split: merge-split (A) vs pairwise-exchange (B)
CC      = mpicc
CFLAGS  = -O2 -Wall
MPIRUN  = mpirun
# On a single multi-core box you may need: NPFLAGS = --oversubscribe
NPFLAGS =

all: methodA methodB

methodA: methodA_mergesplit.c
	$(CC) $(CFLAGS) -o methodA methodA_mergesplit.c

methodB: methodB_pairwise.c
	$(CC) $(CFLAGS) -o methodB methodB_pairwise.c

# Run both methods across the required block sizes and print one table.
run: all
	@echo "k        methodA                              methodB"
	@for k in 10 100 1000 10000 ; do \
		$(MPIRUN) $(NPFLAGS) -np 2 ./methodA $$k ; \
		$(MPIRUN) $(NPFLAGS) -np 2 ./methodB $$k ; \
	done

clean:
	rm -f methodA methodB

.PHONY: all run clean
