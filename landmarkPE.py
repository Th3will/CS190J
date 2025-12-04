# %%
# imports
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T

import numpy as np
import torch
import torch_geometric.datasets as geom_datasets
from torch_geometric.utils import to_undirected

from topomodelx.nn.hypergraph.allset_transformer import AllSetTransformer

import xgi # for working with hypergraphs

import pandas as pd
import random
import numpy as np
from scipy.sparse import coo_array
from scipy.sparse import csr_array
import scipy
import sklearn.metrics

from sklearn.utils.extmath import randomized_svd

import sys

import time

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# %%
# Helpers

def coo2Sparse(coo) -> torch.Tensor:
    if coo is not scipy.sparse.coo_array:
        coo = coo.tocoo()
    indices = np.vstack((coo.row, coo.col))
    values = coo.data
    shape = coo.shape

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    return torch.sparse_coo_tensor(i, v, torch.Size(shape))

dataset = "walmart"

# Parsing the walmart set:
if dataset == "walmart":
    df_he = pd.read_csv('data/walmart/walmart-hyperedges.tsv', sep='\t')
    df_hyper_edges = df_he["nodes"].to_list()
    df_hyper_edges = list(map(lambda x: list(map(int, x.split(','))), df_hyper_edges))

    df_day_of_week_features = df_he["dayofweek"].to_list()
    df_day_of_week_features = list(map(lambda x: [int(x)], df_day_of_week_features))

    df_triptype_labels = df_he["triptype"].to_list()
    df_triptype_labels = list(map(int, df_triptype_labels))

    # y = torch.Tensor(df_triptype_labels)
    y = [i - 1 for i in df_triptype_labels]

    numClasses = np.max(y) + 1
    oneHotY = np.zeros((len(y), numClasses))
    oneHotY[np.arange(len(y)), y] = 1

    y = oneHotY
    y = torch.Tensor(y)

    df_n = pd.read_csv('data/walmart/walmart-nodes.tsv', sep='\t')
    df_categories = np.array(df_n["category"], dtype=np.int64)
    df_nodes = df_n["node_id"].to_list()

    df = df_he

    # Nodes are not ordered by default in xgi so need to make sure 
    H = xgi.Hypergraph()
    
    H.add_nodes_from(df_nodes)
    for he in df_hyper_edges:
        H.add_edge(he)
    
    isolates = H.nodes.isolates()
    H.remove_nodes_from(isolates)

    # Map original node IDs to their categories
    node_cat_map = dict(zip(df_nodes, df_categories))

    # Re-build df_categories to match the current H.nodes order exactly
    # This prevents dimension mismatch in the torch tensor creation below
    df_categories = np.array([node_cat_map[n] for n in H.nodes])

    incidence_1 = coo2Sparse(xgi.convert.to_incidence_matrix(H, sparse=True))

    numClasses = np.max(df_categories)
    x_0s = torch.zeros(H.num_nodes, numClasses)
    x_0s[torch.arange(H.num_nodes), torch.tensor(df_categories - 1)] = 1
    x_0s = x_0s.to_sparse_coo()
    # x_0s = torch.zeros((H.nodes, numClasses))
    # x_0s[torch.arange(len(df_categories)), df_categories - 1] = 1
    


elif dataset == "zoo":
    df = pd.read_csv('data/zoo/zoo.data')
    y = [i - 1 for i in df["type"].to_list()]

    numClasses = np.max(y) + 1
    oneHotY = np.zeros((len(y), numClasses))
    oneHotY[np.arange(len(y)), y] = 1

    y = oneHotY

    num_nodes = df.shape[0]
    ## remove the animal names column
    df_bool_matrix = df.drop(columns=["animal","legs","type"]).values.tolist()

    df_bool_matrix = list(map(lambda x: list(map(int, x)), df_bool_matrix))
    df_neg_bool_matrix = list(map(lambda x: list(map(lambda y: -y + 1, x)), df_bool_matrix))

    ## process legs
    leg_values = [0,2,4,5,6,8]
    df_legs = pd.DataFrame({"%d legs" % i : df["legs"] == i for i in leg_values}, index=df.index)
    
    incidence_1 = np.concatenate([df_bool_matrix, df_neg_bool_matrix, df_legs], axis=1)
    H = xgi.Hypergraph(incidence_1)
    H.cleanup(isolates=True, in_place=True)
    x_0s = np.zeros(shape=(num_nodes,0))
    
    # pos_hypergraph = xgi.convert.from_incidence_matrix(df_bool_matrix)

    incidence_1 = torch.from_numpy(incidence_1).to_sparse()
    # calculate incidence matrix
    # incidence_1 = coo_array(np.array(df_nodes))



#%%
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

def anchor_positional_encoding(hypergraph: xgi.Hypergraph, anchor_nodes, iterations):
    incidence_matrix = xgi.convert.to_incidence_matrix(hypergraph)
    transition_mat = coo2Sparse(hypergraph_random_walk(incidence_matrix))
    
    ary = []
    if len(anchor_nodes) == 0:
        print("No anchor nodes provided, returning zero positional encoding")
        return torch.zeros(incidence_matrix.shape[0], 0)
    
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

#%%
def arnoldi_encoding(hypergraph : xgi.Hypergraph, k : int, smallestOnly = True):
    if k == 0: 
        return np.zeros((hypergraph.num_nodes,0))
    
    laplacian = xgi.linalg.laplacian_matrix.normalized_hypergraph_laplacian(
        hypergraph, 
        sparse=True, 
        index=False
    )

    n = laplacian.shape[0]
    
    DETERMINISM_EIGS_L = np.random.rand(n, k)
    DETERMINISM_EIGS_S = np.random.rand(n, k)
    
    # 2. Run LOBPCG
    # largest=False targets the smallest eigenvalues
    # tol=1e-2 is "loose" but sufficient for clustering/embeddings
    kLargest = torch.from_numpy(np.zeros((hypergraph.num_nodes,0)))
    if not smallestOnly:
        k = k//2
        kLargest = torch.from_numpy(np.real(scipy.sparse.linalg.lobpcg(laplacian, DETERMINISM_EIGS_L, largest=True, tol=1e-2)[1]))
        
    kSmallest = torch.from_numpy(np.real(scipy.sparse.linalg.lobpcg(laplacian, DETERMINISM_EIGS_S, largest=False, tol=1e-2)[1]))


    return torch.cat((kLargest, kSmallest), axis=1).to_sparse()

# %% Actual ML stuff
# Where do we go from here?
# 1. Need to perform a baseline hypergraph transformer without positional encoding
    # I don't know how to input variable sized data into a transformer

class SNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers):
        super().__init__()
        
        # Initial input layer
        layers = [
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU()
        ]
        
        # Stack hidden layers
        for _ in range(n_layers):
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            layers.append(nn.ReLU())
        
        # Register layers as a Sequential block
        self.net = nn.Sequential(*layers)

        self.hyperedge_classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, incidence_matrix):
        nodeEmbeddings = self.net(x)

        edgeEmbeddings = torch.mm(incidence_matrix.t(), nodeEmbeddings)

        return self.hyperedge_classifier(edgeEmbeddings)

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
        Level of the task. Either "graph", "node", or "edge.
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
        self.edge_pool = task_level == "edge"

    def forward(self, x_0, incidence_1):
        # Base model
        x_0, x_1 = self.base_model(x_0, incidence_1)

        # Pool over all nodes in the hypergraph
        x = None
        if self.out_pool:
            x = torch.max(x_0, dim=0)[0] 
        elif self.edge_pool:
            x = x_1
        else:
            x = x_0

        return self.linear(x)

# %%
# Hyperparameters

use_arnoldis = False
eigenvectors = 4

use_random_walk_pe = False
anchorNodes = 0
numWalks = 0


num_epochs = 1000

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

# %%
## add encodings
print("Adding encodings...")

x = x_0s
if use_arnoldis:
    arnoldis = arnoldi_encoding(hypergraph=H, k=eigenvectors)
    x_0s = torch.cat([x_0s, arnoldis], axis=1)

if use_random_walk_pe:
    rw_pe = anchor_positional_encoding(
        hypergraph=H,
        anchor_nodes=random.sample(range(incidence_1.shape[0]), k=anchorNodes),
        iterations=numWalks
    )
    x_0s = torch.cat([x_0s, rw_pe], dim=1)

# %% 

x, incidence_1, y = (
    x_0s.to_dense().float().to(device),
    incidence_1.to_sparse().float().to(device),
    y.to(device),
)


# Base model hyperparameters
in_channels = x.shape[1]
hidden_channels = 128

heads = 4
n_layers = 1
mlp_num_layers = 2

# TODO: BEWAREEEEEE
task_level = "edge" #"graph" if out_channels == 1 else "node"

out_channels = y.shape[1]


# %%

print("Creating model")

model = SNetwork(
    in_channels=in_channels,
    hidden_channels=hidden_channels,
    out_channels=out_channels,
    n_layers=n_layers
    # mlp_num_layers=mlp_num_layers,
    # task_level=task_level,
).to(device)

# Optimizer and loss
opt = torch.optim.Adam(model.parameters(), lr=0.001)

# Categorial cross-entropy loss
loss_fn = torch.nn.CrossEntropyLoss()


# Accuracy
def acc_fn(y, y_hat):
    return (y.argmax(dim=1) == y_hat.argmax(dim=1)).float().mean()


# %%
# create masking for training
TRAIN_TEST_SPLIT_RATIO = 0.8
trainSize = int(len(df)*TRAIN_TEST_SPLIT_RATIO)
train_mask = [True]*int(trainSize) + [False]*int(len(df)-trainSize)
random.shuffle(train_mask)
test_mask = [not x for x in train_mask]

test_interval = 1
epoch_loss = []
train_accuracy = []
test_accuracy = []
print("Starting training")
begin = time.perf_counter()
for epoch_i in range(1, num_epochs + 1):
    model.train()

    opt.zero_grad()

    # Extract edge_index from sparse incidence matrix
    y_hat = model(x, incidence_1)
    loss = loss_fn(y_hat[train_mask], y[train_mask])

    loss.backward()
    opt.step()
    epoch_loss.append(loss.item())
    train_accuracy.append(acc_fn(y_hat[train_mask], y[train_mask]))

    if epoch_i % test_interval == 0:
        model.eval()

        y_hat = model(x, incidence_1)
        loss = loss_fn(y_hat[train_mask], y[train_mask])

        print(f"Epoch: {epoch_i} ")
        print(
            f"Train_loss: {np.mean(epoch_loss):.4f}, acc: {acc_fn(y_hat[train_mask], y[train_mask]):.4f}",
            flush=True,
        )
        test_accuracy.append(acc_fn(y_hat[test_mask], y[test_mask]))

        loss = loss_fn(y_hat[test_mask], y[test_mask])
        print(
            f"Test_loss: {loss:.4f}, Test_acc: {acc_fn(y_hat[test_mask], y[test_mask]):.4f}",
            flush=True,
        )

print(f"Total took {time.perf_counter() - begin} seconds.")
model.eval()
y_hat = model(x, incidence_1)
y_hat = y_hat.argmax(dim=1).cpu().numpy()
cm = sklearn.metrics.confusion_matrix(y.argmax(dim=1).cpu(),y_hat)
# confusion_matrixes.append(cm)

# %%
from matplotlib import pyplot as plt
plt.title("Accuracy over Epochs\nRW (walk size %d, %d anchors) PEs" % (numWalks, anchorNodes))

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim((0, 1))
plt.plot(np.arange(num_epochs, step=test_interval), [x.cpu() for x in test_accuracy], label="Test")
plt.plot([x.cpu() for x in train_accuracy], label="Train")
plt.legend()

disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("RW+Arnoldi-PE confusion matrix")
plt.show()

# %%

print('''{
    "eigenvectors" : %d,
    "anchorNodes" : %d,
    "walks" : %d,     
    "testAcc" : ''' % (eigenvectors if use_arnoldis else -1, anchorNodes if use_random_walk_pe else -1, numWalks if use_random_walk_pe else -1), end="")
print("[" , end="")
for i, v in enumerate([x.cpu() for x in test_accuracy]):
    if i < len(test_accuracy) - 1:
        print("%f, " % (v) , end="")
    else:
        print("%f]" % v)




