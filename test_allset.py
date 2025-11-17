"""
This module contains the AllSetTransformer class for hypergraph-based neural networks.

The AllSet class implements a specific hypergraph-based neural network architecture
used for solving certain types of problems.

Author: Your Name

"""

import numpy as np
import torch
import torch_geometric.datasets as geom_datasets
from torch_geometric.utils import to_undirected

from topomodelx.nn.hypergraph.allset_transformer import AllSetTransformer

torch.manual_seed(0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

cora = geom_datasets.Planetoid(root="tmp/", name="cora")
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


# # Calculate hyperedge statistics.
hyperedge_sizes = [len(he) for he in hyperedges]
min_size = min(hyperedge_sizes)
max_size = max(hyperedge_sizes)
mean_size = np.mean(hyperedge_sizes)
median_size = np.median(hyperedge_sizes)
std_size = np.std(hyperedge_sizes)
num_single_node_hyperedges = sum(np.array(hyperedge_sizes) == 1)

# Print the hyperedge statistics.
print(f"Hyperedge statistics: ")
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

class Network(torch.nn.Module):
    """Network class that initializes the AllSet model and readout layer.

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
        self.base_model = AllSetTransformer(
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

heads = 4
n_layers = 1
mlp_num_layers = 2

# Readout hyperparameters
out_channels = torch.unique(y).shape[0]
task_level = "graph" if out_channels == 1 else "node"


model = Network(
    in_channels=in_channels,
    hidden_channels=hidden_channels,
    out_channels=out_channels,
    n_layers=n_layers,
    mlp_num_layers=mlp_num_layers,
    task_level=task_level,
).to(device)

# Optimizer and loss
opt = torch.optim.Adam(model.parameters(), lr=0.01)

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


test_interval = 1
num_epochs = 5

epoch_loss = []
for epoch_i in range(1, num_epochs + 1):
    model.train()

    opt.zero_grad()

    # Extract edge_index from sparse incidence matrix
    y_hat = model(x_0s, incidence_1)
    loss = loss_fn(y_hat[train_mask], y[train_mask])

    loss.backward()
    opt.step()
    epoch_loss.append(loss.item())

    if epoch_i % test_interval == 0:
        model.eval()
        y_hat = model(x_0s, incidence_1)

        loss = loss_fn(y_hat[train_mask], y[train_mask])
        print(f"Epoch: {epoch_i} ")
        print(
            f"Train_loss: {np.mean(epoch_loss):.4f}, acc: {acc_fn(y_hat[train_mask].argmax(1), y[train_mask]):.4f}",
            flush=True,
        )

        loss = loss_fn(y_hat[val_mask], y[val_mask])

        print(
            f"Val_loss: {loss:.4f}, Val_acc: {acc_fn(y_hat[val_mask].argmax(1), y[val_mask]):.4f}",
            flush=True,
        )

        loss = loss_fn(y_hat[test_mask], y[test_mask])
        print(
            f"Test_loss: {loss:.4f}, Test_acc: {acc_fn(y_hat[test_mask].argmax(1), y[test_mask]):.4f}",
            flush=True,
        )