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

import pickle
import os.path as osp
import copy

import sys

import time
import uuid
import json

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
    if not isinstance(coo, scipy.sparse.coo_array):
        coo = coo.tocoo()
    indices = np.vstack((coo.row, coo.col))
    values = coo.data
    shape = coo.shape

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    return torch.sparse_coo_tensor(i, v, torch.Size(shape))

dataset = "walmart"

if dataset == "cora":
    # from AllSet loading
    def getCora():
        coraPath = "data/cocitation/cora/"
        print(f'Loading hypergraph dataset from hyperGCN: {dataset}')

        # first load node features:
        with open(coraPath + 'features.pickle', 'rb') as f:
            features = pickle.load(f)
            features = features.todense()

        with open(coraPath + 'labels.pickle', 'rb') as f:
            labels = pickle.load(f)

        num_nodes, feature_dim = features.shape
        assert num_nodes == len(labels)
        print(f'number of nodes:{num_nodes}, feature dimension: {feature_dim}')

        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)

        # The last, load hypergraph.
        with open(coraPath + 'hypergraph.pickle', 'rb') as f:
            # hypergraph in hyperGCN is in the form of a dictionary.
            # { hyperedge: [list of nodes in the he], ...}
            hypergraph = pickle.load(f)

        print(f'number of hyperedges: {len(hypergraph)}')

        # exploration
        maxEdgeId = max(hypergraph)
        for i in range(num_nodes):
            hypergraph[maxEdgeId+i] = {i}
        H = xgi.Hypergraph(hypergraph)
        # H.cleanup(isolates=True, connected=False, relabel=False)

        return H, features, labels

    H, x_0s, y_indices = getCora()
    incidence_1 = coo2Sparse(xgi.convert.to_incidence_matrix(H, sparse=True))

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

    #y = oneHotY
    y_indices = torch.tensor(df_triptype_labels) - 1  # 0-based indices
    #y = torch.Tensor(y)

    df_n = pd.read_csv('data/walmart/walmart-nodes.tsv', sep='\t')
    df_categories = np.array(df_n["category"], dtype=np.int64)
    df_nodes = df_n["node_id"].to_list()

    node_cat_map = dict(zip(df_nodes, df_categories))

    df = df_he

    # Nodes are not ordered by default in xgi so need to make sure 
    H = xgi.Hypergraph()
    
    H.add_nodes_from(df_nodes)
    for he in df_hyper_edges:
        H.add_edge(he)
    
    isolates = H.nodes.isolates()
    H.remove_nodes_from(isolates)

    incidence_1, node_idx_map, edge_idx_map = xgi.convert.to_incidence_matrix(H, sparse=True, index=True)
    incidence_1 = coo2Sparse(incidence_1)

    # 1. Align Node Features (You were already doing this mostly correctly)
    final_node_order = [node_idx_map[i] for i in range(len(node_idx_map))]
    df_categories = np.array([node_cat_map[n] for n in final_node_order])
    # ... create x_0s based on this ...

    # 2. Align Edge Labels (CRITICAL MISSING STEP)
    # edge_idx_map maps { xgi_edge_id : matrix_column_index }
    # We need the reverse: for matrix column 0, which df index was it?
    sorted_edge_indices = sorted(edge_idx_map.items(), key=lambda item: item[1])
    original_df_indices = [k for k, v in sorted_edge_indices]

    # Reorder y to match the matrix columns
    y_aligned = np.array(df_triptype_labels)[original_df_indices] 
    y_indices = torch.tensor(y_aligned) - 1

    numCategories = np.max(df_categories)
    x_0s = torch.zeros(H.num_nodes, numCategories)
    x_0s[torch.arange(H.num_nodes), torch.tensor(df_categories - 1)] = 1
    x_0s = x_0s.to_sparse_coo()
    # x_0s = torch.zeros((H.nodes, numClasses))
    # x_0s[torch.arange(len(df_categories)), df_categories - 1] = 1
    


elif dataset == "zoo":
    df = pd.read_csv('data/zoo/zoo.data')
    y_indices = torch.LongTensor([i - 1 for i in df["type"].to_list()])

    # numClasses = np.max(y) + 1
    # oneHotY = np.zeros((len(y), numClasses))
    # oneHotY[np.arange(len(y)), y] = 1

    # y = oneHotY

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
    x_0s = torch.zeros(size=(num_nodes,0))
    
    # pos_hypergraph = xgi.convert.from_incidence_matrix(df_bool_matrix)

    incidence_1 = torch.from_numpy(incidence_1).to_sparse()
    # calculate incidence matrix
    # incidence_1 = coo_array(np.array(df_nodes))


#%%
def compute_hyperedges_medoids(hypergraph: xgi.Hypergraph, node_features: torch.Tensor, top_k: int = 1):
    medoids = []
    
    if isinstance(node_features, torch.Tensor):
        features = node_features.cpu().numpy() if node_features.is_cuda else node_features.numpy()
    else:
        features = node_features
    
    # Create mapping from xgi node IDs to feature indices
    node_list = sorted(list(hypergraph.nodes))
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    
    # Iterate through hyperedges
    for edge_id in hypergraph.edges:
        edge_nodes = hypergraph.edges.members(edge_id)
        if len(edge_nodes) == 0:
            continue
        
        # Convert edge nodes to indices
        edge_node_indices = [node_to_idx[node] for node in edge_nodes if node in node_to_idx]
        
        if len(edge_node_indices) == 0:
            continue
        
        # Get features for nodes in hyperedge
        edge_features = features[edge_node_indices]
        
        # Compute pairwise distances for nodes in hyperedge
        distances = []
        for i, feat_i in enumerate(edge_features):
            dist_sum = np.sum(np.linalg.norm(edge_features - feat_i, axis=1))
            distances.append((dist_sum, edge_node_indices[i]))
        
        # Sort by distance then node index
        distances.sort(key=lambda x: (x[0], x[1]))
        
        # Get top-k medoids for hyperedge
        for i in range(min(top_k, len(distances))):
            medoid_idx = distances[i][1]
            medoids.append(medoid_idx)
    
    # Remove duplicates
    seen = set()
    unique_medoids = []
    for medoid in medoids:
        if medoid not in seen:
            seen.add(medoid)
            unique_medoids.append(medoid)
    
    return unique_medoids

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

    kLargest = torch.from_numpy(np.zeros((hypergraph.num_nodes,0)))
    if not smallestOnly:
        k = k//2
        DETERMINISM_EIGS_L = np.random.rand(n, k)
        kLargest = torch.from_numpy(np.real(scipy.sparse.linalg.lobpcg(laplacian, DETERMINISM_EIGS_L, largest=True, tol=1e-4)[1]))

    DETERMINISM_EIGS_S = np.random.rand(n, k)
    
    kSmallest = torch.from_numpy(np.real(scipy.sparse.linalg.lobpcg(laplacian, DETERMINISM_EIGS_S, largest=False, tol=1e-4)[1]))

    return torch.cat((kLargest, kSmallest), axis=1).to_sparse()
    
    
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

# Model saving parameters
save_model = False
model_output_dir = None

num_epochs = 200

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
        elif sys.argv[i].startswith("-o"):
            save_model = True
            i += 1
            model_output_dir = sys.argv[i] if i < len(sys.argv) else "."
        i += 1

# %%
## add encodings
print("Adding encodings...")

# Convert to dense for concatenation if sparse
x_0s = x_0s.to_dense() if x_0s.is_sparse else x_0s
x = x_0s

if use_arnoldis:
    arnoldis = arnoldi_encoding(hypergraph=H, k=eigenvectors)
    arnoldis_dense = arnoldis.to_dense() if arnoldis.is_sparse else arnoldis
    x_0s = torch.cat([x_0s, arnoldis_dense], dim=1)

if use_random_walk_pe:
    # Compute hyperedge medoids to use as anchor nodes
    print(f"Computing hyperedge medoids for anchor node selection...")
    all_medoids = compute_hyperedges_medoids(hypergraph=H, node_features=x_0s, top_k=1)
    print(f"Found {len(all_medoids)} unique medoids from {len(H.edges)} hyperedges")
    
    # Select anchorNodes medoids (deterministic: take first anchorNodes)
    if len(all_medoids) >= anchorNodes:
        selected_medoids = all_medoids[:anchorNodes]
    else:
        # If we have fewer medoids than requested, use all of them
        selected_medoids = all_medoids
        print(f"Warning: Only {len(selected_medoids)} medoids available, using all of them instead of {anchorNodes}")
    
    print(f"Using {len(selected_medoids)} medoids as anchor nodes for random walk")
    rw_pe = anchor_positional_encoding(
        hypergraph=H,
        anchor_nodes=selected_medoids,
        iterations=numWalks
    )
    rw_pe_dense = rw_pe.to_dense() if rw_pe.is_sparse else rw_pe
    x_0s = torch.cat([x_0s, rw_pe_dense], dim=1)

# x_0s = torch.reshape(x_0s, (1, -1))
# %% 

x, incidence_1, y = (
    x_0s.to_dense().float().to(device),
    incidence_1.to_sparse().float().to(device),
    y_indices.to(device),
)


# Base model hyperparameters
in_channels = x.shape[1]
hidden_channels = 128

heads = 4
n_layers = 1
mlp_num_layers = 2

# TODO: BEWAREEEEEE
task_level = "edge" if dataset == "walmart" else "node"

# out_channels = numClasses # WALMART
out_channels = (max(y_indices) + 1) # * H.num_nodes

print(x[0])
print(x[1])
# %%

print("Creating model")
model = Network(
    in_channels=in_channels,
    hidden_channels=hidden_channels,
    out_channels=out_channels,
    task_level=task_level
).to(device)

# Optimizer and loss
opt = torch.optim.Adam(model.parameters(), lr=0.001)

# Categorial cross-entropy loss
loss_fn = torch.nn.CrossEntropyLoss()


# Accuracy
def acc_fn(y_hat, y):
    return (y == y_hat.argmax(dim=1)).float().mean()


# %%
# create masking for training
TRAIN_TEST_SPLIT_RATIO = 0.8
# For walmart, labels are per hyperedge (edge-level task)
# For cora/zoo, labels are per node (node-level task)
if dataset == "walmart":
    outsize = len(y_indices)  # Number of hyperedges
else:
    outsize = H.num_nodes  # Number of nodes
trainSize = int(outsize*TRAIN_TEST_SPLIT_RATIO)
train_mask = [True]*int(trainSize) + [False]*int(outsize-trainSize)
random.shuffle(train_mask)
test_mask = [not x for x in train_mask]

test_interval = 1
epoch_loss = []
train_accuracy = []
test_accuracy = []
best_test_acc = 0.0
best_test_acc_epoch = 0
best_test_acc_epoch200 = 0.0
best_model_state = None
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
    train_accuracy.append(acc_fn(y_hat[train_mask], y[train_mask]).item())

    if epoch_i % test_interval == 0:
        model.eval()

        y_hat = model(x, incidence_1)
        loss = loss_fn(y_hat[train_mask], y[train_mask])

        print(f"Epoch: {epoch_i} ")
        print(
            f"Train_loss: {np.mean(epoch_loss):.4f}, acc: {acc_fn(y_hat[train_mask], y[train_mask]):.4f}",
            flush=True,
        )
        current_test_acc = acc_fn(y_hat[test_mask], y[test_mask])
        test_accuracy.append(current_test_acc)
        
        # Update best test accuracy
        test_acc_value = current_test_acc.item() if torch.is_tensor(current_test_acc) else current_test_acc
        if test_acc_value > best_test_acc:
            best_test_acc = test_acc_value
            best_test_acc_epoch = epoch_i
            # Save best model state (deep copy)
            best_model_state = copy.deepcopy(model.state_dict())
        
        # Track best accuracy up to epoch 200
        if epoch_i <= 200:
            if test_acc_value > best_test_acc_epoch200:
                best_test_acc_epoch200 = test_acc_value

        loss = loss_fn(y_hat[test_mask], y[test_mask])
        print(
            f"Test_loss: {loss:.4f}, Test_acc: {current_test_acc:.4f}",
            flush=True,
        )

print(f"Total took {time.perf_counter() - begin} seconds.")
model.eval()
y_hat = model(x, incidence_1)
y_hat = y_hat.argmax(dim=1).cpu().numpy()
cm = sklearn.metrics.confusion_matrix(y.cpu(),y_hat)
# confusion_matrixes.append(cm)

# # %%
# from matplotlib import pyplot as plt
# plt.title("Accuracy over Epochs\nRW (walk size %d, %d anchors) PEs" % (numWalks, anchorNodes))

# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.ylim((0, 1))
# plt.plot(np.arange(num_epochs, step=test_interval), [x.cpu() for x in test_accuracy], label="Test")
# plt.plot([x.cpu() for x in train_accuracy], label="Train")
# plt.legend()

# disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot()
# plt.title("RW+Arnoldi-PE confusion matrix")
# plt.show()

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

# Save best test accuracy and args to file
# Create filename based on args
filename_parts = [f"results_{dataset}"]
if use_arnoldis:
    filename_parts.append(f"a{eigenvectors}")
if use_random_walk_pe:
    filename_parts.append(f"w{anchorNodes}_{numWalks}")
filename_parts.append(f"e{num_epochs}")
filename_parts.append(uuid.uuid4().hex[:4])  # Short UUID (4 chars)
random_filename = "_".join(filename_parts) + ".json"
results = {
    "best_test_acc": float(best_test_acc),
    "best_test_acc_epoch": int(best_test_acc_epoch),
    "best_test_acc_epoch200": float(best_test_acc_epoch200),
    "args": {
        "use_arnoldis": use_arnoldis,
        "eigenvectors": eigenvectors if use_arnoldis else -1,
        "use_random_walk_pe": use_random_walk_pe,
        "anchorNodes": anchorNodes if use_random_walk_pe else -1,
        "numWalks": numWalks if use_random_walk_pe else -1,
        "num_epochs": num_epochs,
        "command_line": " ".join(sys.argv)
    },
    "all_test_acc": [float(x.cpu().item() if torch.is_tensor(x) else x) for x in test_accuracy],
    "total_time_seconds": time.perf_counter() - begin
}

# Determine output directory
output_dir = model_output_dir if model_output_dir else "."
os.makedirs(output_dir, exist_ok=True)

# Save results JSON
output_json_path = os.path.join(output_dir, random_filename)
with open(output_json_path, 'w') as f:
    json.dump(results, f, indent=2)

# Save model if requested
model_filename = None
if save_model and best_model_state is not None:
    model_filename = random_filename.replace('.json', '.pt')
    model_path = os.path.join(output_dir, model_filename)
    torch.save({
        'model_state_dict': best_model_state,
        'best_test_acc': best_test_acc,
        'best_test_acc_epoch': best_test_acc_epoch,
        'best_test_acc_epoch200': best_test_acc_epoch200,
        'args': results['args']
    }, model_path)
    print(f"Model saved to: {model_path}")

print(f"\nBest test accuracy: {best_test_acc:.4f} at epoch {best_test_acc_epoch}")
print(f"Best test accuracy (up to epoch 200): {best_test_acc_epoch200:.4f}")
print(f"Results saved to: {output_json_path}")




