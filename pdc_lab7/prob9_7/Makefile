# Problem 9.7 - mesh bitonic communication-cost simulation
CC     = mpicc
CFLAGS = -O2 -Wall
LDLIBS = -lm
all: mesh_cost
mesh_cost: mesh_cost.c
	$(CC) $(CFLAGS) -o mesh_cost mesh_cost.c $(LDLIBS)
run: all
	mpirun --oversubscribe -np 4 ./mesh_cost 16
	mpirun --oversubscribe -np 4 ./mesh_cost 64
	mpirun --oversubscribe -np 4 ./mesh_cost 256
clean:
	rm -f mesh_cost
.PHONY: all run clean
