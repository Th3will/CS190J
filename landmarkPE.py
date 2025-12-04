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
    if not isinstance(coo, scipy.sparse.coo_array):
        coo = coo.tocoo()
    indices = np.vstack((coo.row, coo.col))
    values = coo.data
    shape = coo.shape

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    return torch.sparse_coo_tensor(i, v, torch.Size(shape))

dataset = "zoo"

if dataset == "cora":
    # from AllSet loading
    def getCora():
        coraPath = "data/cocitation/cora/"
        print(f'Loading hypergraph dataset from hyperGCN: {dataset}')

        # first load node features:
        with open(coraPath + 'features.pickle', 'rb') as f:
            features = pickle.load(f)
            features = features.todense()[:, :0]

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

# %% Actual ML stuff
# Where do we go from here?
# 1. Need to perform a baseline hypergraph transformer without positional encoding
    # I don't know how to input variable sized data into a transformer

class LinearSelfAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        assert dim % heads == 0, "Hidden dimension must be divisible by heads."
        
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V projections
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x shape: (Batch, N, Dim)
        b, n, d = x.shape
        h = self.heads
        
        # 1. Project Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)
        # Shapes are now (Batch, Heads, N, Head_Dim)

        # 2. Apply Feature Map (The "Kernel Trick")
        # ELU + 1 is a common approximation for Softmax that keeps values positive
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # 3. Linear Attention Computation
        # Standard: softmax(Q @ K.T) @ V  -> Complexity O(N^2)
        # Linear:   Q @ (K.T @ V)         -> Complexity O(N)
        
        # kv = (K.T @ V). We sum over N here.
        # Shape: (Batch, Heads, Head_Dim, Head_Dim)
        kv = torch.einsum('bhnd,bhne->bhde', k, v)
        
        # z = 1 / (Q @ K.sum(dim=1)) - Normalization factor
        z = 1 / (torch.einsum('bhnd,bhd->bhn', q, k.sum(dim=2)) + 1e-6)
        
        # out = (Q @ kv) * z
        out = torch.einsum('bhnd,bhde->bhne', q, kv) * z.unsqueeze(-1)
        
        # 4. Reshape back
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.to_out(out)

class LinearTransformerBlock(nn.Module):
    """
    Replaces a standard TransformerEncoderLayer but uses Linear Attention
    """
    def __init__(self, dim, heads, dropout=0.1, mlp_dim=None):
        super().__init__()
        mlp_dim = mlp_dim or dim * 4
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LinearSelfAttention(dim, heads=heads, dropout=dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Residual connections are handled here
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class STransformer2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, nhead=4):
        super().__init__()
        
        self.embedding = nn.Linear(in_channels, hidden_channels)
        
        # REPLACED: Standard Transformer -> Stack of Linear Blocks
        self.layers = nn.ModuleList([
            LinearTransformerBlock(dim=hidden_channels, heads=nhead)
            for _ in range(n_layers)
        ])
        
        self.fc_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, incidence_1):
        # x input: (N, In_Channels)
        
        x = self.embedding(x) # (N, Hidden)
        
        # Unsqueeze batch dim: (1, N, Hidden)
        x = x.unsqueeze(0) 
        
        # Apply Linear Transformer Layers
        for layer in self.layers:
            x = layer(x)
            
        x = x.squeeze(0) # Back to (N, Hidden)

        # Message Passing (Node -> Hyperedge aggregation)
        # hyperedge_degrees = torch.sparse.sum(incidence_1, dim=0).to_dense()
        # hyperedge_degrees = hyperedge_degrees.clamp(min=1).unsqueeze(1) 

        # x = torch.mm(incidence_1.t(), x) # WALMART
        # x = x / hyperedge_degrees 

        return self.fc_out(x)

class GraphTransformer(nn.Module):
    def __init__(self, in_dim, d_model, nhead, num_layers, dim_feedforward=512, dropout=0.1):
        """
        Args:
            in_dim: Original feature dimension (e.g., 1433).
            d_model: Internal dimension for the transformer (must be divisible by nhead).
            nhead: Number of attention heads.
            num_layers: Number of transformer encoder layers.
        """
        super().__init__()
        
        # 1. Projection: Map irregular input dims (like 1433) to d_model
        self.project_in = nn.Linear(in_dim, d_model)
        
        # 2. Transformer Encoder
        # batch_first=True expects input: (Batch_Size, Seq_Len/Nodes, Features)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Output Projection (Optional: project back to class logits or keep as embeddings)
        # self.project_out = nn.Linear(d_model, num_classes) # Uncomment if needed

    def forward(self, x, mask=None, src_key_padding_mask=None):
        """
        x: Input tensor of shape (Batch, Nodes, in_dim) or (Nodes, in_dim)
        mask: Optional mask for self-attention (e.g., adjacency constraints)
        src_key_padding_mask: Optional mask for ignoring padding nodes in batch
        """
        # Ensure batch dimension exists
        if x.dim() == 2:
            x = x.unsqueeze(0) # (1, Nodes, Features)

        # Project features
        x = self.project_in(x)
        
        # Apply Transformer
        # Output shape: (Batch, Nodes, d_model)
        out = self.transformer_encoder(x, mask=mask, src_key_padding_mask=src_key_padding_mask)
        
        return out

class ConciseTransformer(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        nhead: int, 
        num_encoder_layers: int, 
        num_decoder_layers: int, 
        output_dim: int, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: The number of expected features in the encoder/decoder inputs (embedding size).
            nhead: The number of heads in the multiheadattention models.
            num_encoder_layers: The number of sub-encoder-layers in the encoder.
            num_decoder_layers: The number of sub-decoder-layers in the decoder.
            output_dim: The dimension of the final output (e.g., vocab size).
        """
        super().__init__()
        
        # Native PyTorch Transformer. 
        # Note: This module does NOT add positional encodings, satisfying your constraint.
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # Expects (batch, seq_len, features)
        )
        
        # Final projection layer to map back to vocabulary/output size
        self.fc_out = nn.Linear(d_model, output_dim)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def forward(
        self, 
        src: torch.Tensor, 
        tgt: torch.Tensor, 
        src_mask: torch.Tensor = None, 
        tgt_mask: torch.Tensor = None,
        src_padding_mask: torch.Tensor = None, 
        tgt_padding_mask: torch.Tensor = None
    ):
        """
        Args:
            src: Source sequence (Batch, Seq_Len, d_model) - Includes Positional Encoding
            tgt: Target sequence (Batch, Seq_Len, d_model) - Includes Positional Encoding
            tgt_mask: Causal mask for decoder (usually required)
            padding_masks: Boolean masks where True indicates padding to be ignored.
        """
        
        # Pass through the main Transformer (Encoder + Decoder)
        out = self.transformer(
            src=src,
            tgt=tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        # Project to output dimension
        return self.fc_out(out)
    
class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers):
        super().__init__()
        layers = []
        
        # Handle the single layer case specifically
        if n_layers == 1:
            layers.append(nn.Linear(in_channels, out_channels))
        else:
            # Input layer
            layers.append(nn.Linear(in_channels, hidden_channels))
            layers.append(nn.ReLU())
            
            # Hidden layers
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_channels, hidden_channels))
                layers.append(nn.ReLU())
            
            # Output layer
            layers.append(nn.Linear(hidden_channels, out_channels))
            
        self.net = nn.Sequential(*layers)

    def forward(self, x, incidence):
        return self.net(x)
    
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
    x_0s = torch.cat([x_0s, arnoldis.to_dense()], dim=1)

if use_random_walk_pe:
    rw_pe = anchor_positional_encoding(
        hypergraph=H,
        anchor_nodes=random.sample(range(incidence_1.shape[0]), k=anchorNodes),
        iterations=numWalks
    )
    x_0s = torch.cat([x_0s, rw_pe.to_dense()], dim=1)

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
task_level = "node" #"graph" if out_channels == 1 else "node"

# out_channels = numClasses # WALMART
out_channels = (max(y_indices) + 1) # * H.num_nodes

print(x[0])
print(x[1])
# %%

print("Creating model")
model = GraphTransformer(
    in_dim=in_channels,
    d_model=128,
    nhead=4,
    num_layers=2
)
# model = ConciseTransformer(
#     d_model=in_channels,
#     nhead=1,
#     num_encoder_layers=2,
#     num_decoder_layers=0,
#     output_dim=out_channels
# )
# model = STransformer2(
#     in_channels=in_channels,
#     hidden_channels=hidden_channels,
#     out_channels=out_channels,
#     n_layers=n_layers
#     # mlp_num_layers=mlp_num_layers,
#     # task_level=task_level,
# ).to(device)

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
# outsize = len(df) # WALMART
outsize = H.num_nodes
trainSize = int(outsize*TRAIN_TEST_SPLIT_RATIO)
train_mask = [True]*int(trainSize) + [False]*int(outsize-trainSize)
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
    y_hat = model(x)
    loss = loss_fn(y_hat[0][train_mask], y[train_mask])

    loss.backward()
    opt.step()
    epoch_loss.append(loss.item())
    train_accuracy.append(acc_fn(y_hat[0][train_mask], y[train_mask]).item())

    if epoch_i % test_interval == 0:
        model.eval()

        y_hat = model(x)
        loss = loss_fn(y_hat[0][train_mask], y[train_mask])

        print(f"Epoch: {epoch_i} ")
        print(
            f"Train_loss: {np.mean(epoch_loss):.4f}, acc: {acc_fn(y_hat[0][train_mask], y[train_mask]):.4f}",
            flush=True,
        )
        test_accuracy.append(acc_fn(y_hat[0][test_mask], y[test_mask]))

        loss = loss_fn(y_hat[0][test_mask], y[test_mask])
        print(
            f"Test_loss: {loss:.4f}, Test_acc: {acc_fn(y_hat[0][test_mask], y[test_mask]):.4f}",
            flush=True,
        )

print(f"Total took {time.perf_counter() - begin} seconds.")
model.eval()
y_hat = model(x)
y_hat = y_hat[0].argmax(dim=1).cpu().numpy()
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




