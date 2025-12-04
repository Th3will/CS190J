import numpy as np
import torch
import torch_geometric.datasets as geom_datasets
from torch_geometric.utils import to_undirected

from topomodelx.nn.hypergraph.unigcnii import UniGCNII

import xgi
import sys
import random
import scipy

torch.manual_seed(0)

# 

def coo2Sparse(coo) -> torch.Tensor:
    if not isinstance(coo, scipy.sparse.coo_array):
        coo = coo.tocoo()
    indices = np.vstack((coo.row, coo.col))
    values = coo.data
    shape = coo.shape

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    return torch.sparse_coo_tensor(i, v, torch.Size(shape))


def hypergraph_random_walk(incidence_matrix):
    C = incidence_matrix.T @ incidence_matrix # hyper-edge "adjacency" matrix
    # C_hat is defined by just the diagonal of C
    C_hat = C * scipy.sparse.eye_array(C.shape[0])

    adj_mat = incidence_matrix @ incidence_matrix.T
    transition_mat =  incidence_matrix @ C_hat @ incidence_matrix.T - adj_mat
    # zero diagonals
    transition_mat.setdiag(0)
    zero_to_one = lambda i : 1 if i == 0 else i
    return transition_mat/np.array([zero_to_one(i) for i in transition_mat.sum(axis=1)])

def anchor_positional_encoding(incidence_1, anchor_nodes, iterations):
    transition_mat = coo2Sparse(hypergraph_random_walk(incidence_1))
    
    ary = []
    if len(anchor_nodes) == 0:
        print("No anchor nodes provided, returning zero positional encoding")
        return torch.zeros(incidence_1.shape[0], 0)
    
    # initialize anchor node matrix which is a one-hot encoding of the anchor nodes
    
    # for node in anchor_nodes:
    #     temp = np.zeros(transition_mat.shape[0])
    #     temp[node] = 1
    #     ary.append(temp)
    # anchor_node = np.array(ary).T  # shape: num_nodes x num_anchors

    indicies = torch.stack([torch.as_tensor(anchor_nodes), torch.arange(len(anchor_nodes))])
    values = torch.ones(len(anchor_nodes))

    anchor_node = torch.sparse_coo_tensor(
        indicies, 
        values, 
        size=(transition_mat.shape[0], len(anchor_nodes))
    )
    
    # perform random walk for specified iterations
    for _ in range(iterations):
        anchor_node = transition_mat @ anchor_node

    return torch.tensor(anchor_node)

def arnoldi_encoding(incidence_1, k : int, smallestOnly = True):
    H = xgi.from_incidence_matrix(incidence_1.to_dense())
    if k == 0: 
        return np.zeros((incidence_1.shape[0],0))
    
    laplacian = xgi.linalg.laplacian_matrix.normalized_hypergraph_laplacian(
        H, 
        sparse=True, 
        index=False
    )

    n = laplacian.shape[0]

    kLargest = torch.from_numpy(np.zeros((H.num_nodes,0)))
    if not smallestOnly:
        k = k//2
        DETERMINISM_EIGS_L = np.random.rand(n, k)
        kLargest = torch.from_numpy(np.real(scipy.sparse.linalg.lobpcg(laplacian, DETERMINISM_EIGS_L, largest=True, tol=1e-4)[1]))

    DETERMINISM_EIGS_S = np.random.rand(n, k)
    
    kSmallest = torch.from_numpy(np.real(scipy.sparse.linalg.lobpcg(laplacian, DETERMINISM_EIGS_S, largest=False, tol=1e-4)[1]))

    return torch.cat((kLargest, kSmallest), axis=1).to_sparse()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

cora = geom_datasets.Planetoid(root="data/", name="cora")
data = cora.data

x_0s = data.x
y = data.y
edge_index = data.edge_index

train_mask = data.train_mask
val_mask = data.val_mask
test_mask = data.test_mask

# Ensure the graph is undirected (optional but often useful for one-hop neighborhoods).
edge_index = to_undirected(edge_index)

# Create a list of one-hop neighborhoods for each node.
one_hop_neighborhoods = []
for node in range(data.num_nodes):
    # Get the one-hop neighbors of the current node.
    neighbors = data.edge_index[1, data.edge_index[0] == node]

    # Append the neighbors to the list of one-hop neighborhoods.
    one_hop_neighborhoods.append(neighbors.numpy())

# Detect and eliminate duplicate hyperedges.
unique_hyperedges = set()
hyperedges = []
for neighborhood in one_hop_neighborhoods:
    # Sort the neighborhood to ensure consistent comparison.
    neighborhood = tuple(sorted(neighborhood))
    if neighborhood not in unique_hyperedges:
        hyperedges.append(list(neighborhood))
        unique_hyperedges.add(neighborhood)

# Calculate hyperedge statistics.
hyperedge_sizes = [len(he) for he in hyperedges]
min_size = min(hyperedge_sizes)
max_size = max(hyperedge_sizes)
mean_size = np.mean(hyperedge_sizes)
median_size = np.median(hyperedge_sizes)
std_size = np.std(hyperedge_sizes)
num_single_node_hyperedges = sum(np.array(hyperedge_sizes) == 1)

# Print the hyperedge statistics.
print("Hyperedge statistics: ")
print("Number of hyperedges without duplicated hyperedges", len(hyperedges))
print(f"min = {min_size}, ")
print(f"max = {max_size}, ")
print(f"mean = {mean_size}, ")
print(f"median = {median_size}, ")
print(f"std = {std_size}, ")
print(f"Number of hyperedges with size equal to one = {num_single_node_hyperedges}")

max_edges = len(hyperedges)
incidence_1 = np.zeros((x_0s.shape[0], max_edges))
for col, neighibourhood in enumerate(hyperedges):
    for row in neighibourhood:
        incidence_1[row, col] = 1

assert all(incidence_1.sum(0) > 0) is True, "Some hyperedges are empty"
assert all(incidence_1.sum(1) > 0) is True, "Some nodes are not in any hyperedges"
incidence_1 = torch.Tensor(incidence_1).to_sparse_coo()

# adding encodings

use_arnoldis = False
eigenvectors = 4

use_random_walk_pe = False
anchorNodes = 0
numWalks = 0

num_epochs = 100

if len(sys.argv) > 1:
    i = 1
    while i < len(sys.argv):
        if sys.argv[i].startswith("-a"):
            use_arnoldis = True
            i += 1
            eigenvectors = int(sys.argv[i])
            i += 1
        elif sys.argv[i].startswith("-w"):
            use_random_walk_pe = True
            i += 1
            anchorNodes = int(sys.argv[i])
            i += 1
            numWalks = int(sys.argv[i])
        elif sys.argv[i].startswith("-e"):
            i += 1
            num_epochs = int(sys.argv[i])
        i += 1

## add encodings
print("Adding encodings...")

if use_arnoldis:
    arnoldis = arnoldi_encoding(incidence_1, k=eigenvectors)
    x_0s = torch.cat([x_0s, arnoldis.to_dense()], dim=1)

if use_random_walk_pe:
    rw_pe = anchor_positional_encoding(
        incidence_1=incidence_1,
        anchor_nodes=random.sample(range(incidence_1.shape[0]), k=anchorNodes),
        iterations=numWalks
    )
    x_0s = torch.cat([x_0s, rw_pe.to_dense()], dim=1)



class Network(torch.nn.Module):
    """Network class that initializes the base model and readout layer.

    Base model parameters:
    ----------
    Reqired:
    in_channels : int
        Dimension of the input features.
    hidden_channels : int
        Dimension of the hidden features.

    Optitional:
    **kwargs : dict
        Additional arguments for the base model.

    Readout layer parameters:
    ----------
    out_channels : int
        Dimension of the output features.
    task_level : str
        Level of the task. Either "graph" or "node".
    """

    def __init__(
        self, in_channels, hidden_channels, out_channels, task_level="graph", **kwargs
    ):
        super().__init__()

        # Define the model
        self.base_model = UniGCNII(
            in_channels=in_channels, hidden_channels=hidden_channels, **kwargs
        )

        # Readout
        self.linear = torch.nn.Linear(hidden_channels, out_channels)
        self.out_pool = task_level == "graph"

    def forward(self, x_0, incidence_1):
        # Base model
        x_0, x_1 = self.base_model(x_0, incidence_1)

        # Pool over all nodes in the hypergraph
        x = torch.max(x_0, dim=0)[0] if self.out_pool is True else x_0

        return self.linear(x)

# Base model hyperparameters
in_channels = x_0s.shape[1]
hidden_channels = 128
n_layers = 2
mlp_num_layers = 1
input_drop = 0.5

# Readout hyperparameters
out_channels = torch.unique(y).shape[0]
task_level = "graph" if out_channels == 1 else "node"


model = Network(
    in_channels=in_channels,
    hidden_channels=hidden_channels,
    out_channels=out_channels,
    n_layers=n_layers,
    input_drop=input_drop,
    task_level=task_level,
).to(device)

test_interval = 5

# Optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Categorial cross-entropy loss
loss_fn = torch.nn.CrossEntropyLoss()


# Accuracy
def acc_fn(y, y_hat):
    return (y == y_hat).float().mean()

x_0s = torch.tensor(x_0s)
x_0s, incidence_1, y = (
    x_0s.float().to(device),
    incidence_1.float().to(device),
    torch.tensor(y, dtype=torch.long).to(device),
)

for epoch in range(num_epochs):
    # set model to training mode
    model.train()

    y_hat = model(x_0s, incidence_1)
    loss = loss_fn(y_hat[train_mask], y[train_mask])

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % test_interval == 0:
        model.eval()
        y_hat = model(x_0s, incidence_1)

        train_loss = loss_fn(y_hat[train_mask], y[train_mask])
        loss = loss_fn(y_hat[test_mask], y[test_mask])
        print(
            f"Epoch: {epoch} \
              Train_loss: {train_loss:.4f}, Train_acc: {acc_fn(y_hat[train_mask].argmax(1), y[train_mask]):.4f} \
              Test_loss: {loss:.4f}, Test_acc: {acc_fn(y_hat[test_mask].argmax(1), y[test_mask]):.4f}",
            flush=True,
        )