from mpi4py import MPI
import networkx as nx
import time


comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()  # 5
# reading in graph data here

start_time = time.time()
if rank == 0:   # broadcaster
    #G = nx.path_graph(200)
    G = nx.read_edgelist("twitter_combined.txt", create_using=nx.DiGraph)
else:
    G = None
G = comm.bcast(G, root=0)
# partitioning
all_nodes = list(G.nodes)
N = len(all_nodes)
nodes_subset = all_nodes[int((rank*N)/size):int(((rank+1)*N)/size)]
print("I am processor", rank)
lengths = []
sums = []
ccs = []

# doing dijkstra's
for i in nodes_subset:
    # print("Source:", i)
    for j in all_nodes:
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
