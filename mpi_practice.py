from mpi4py import MPI
import networkx as nx
import time


comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()  # 5
N = 200
# reading in graph data here

start_time = time.time()
if rank == 0:   # broadcaster
    G = nx.path_graph(N)
else:
    G = None
G = comm.bcast(G, root=0)

x = range(int((rank*N)/size), int(((rank+1)*N)/size))
print("I am processor", rank)
lengths = []
sums = []
ccs = []

# doing dijkstra's
for i in x:
    # print("Source:", i)
    for j in range(N):
        if nx.has_path(G, i, j):
            length, path = (nx.bidirectional_dijkstra(G, i, j))
            lengths.append(length)
    sums.append(sum(lengths))
    for j in sums:
        j = (N-1)/j
    ccs.append(j)
    lengths.clear()

cc_vals = comm.gather(ccs, root=0)
end_time = time.time()

if rank == 0:
    print(cc_vals, '\n')
    print("Time taken:", (end_time - start_time))
