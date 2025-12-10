import torch
import torch.nn as nn
import torch.nn.functional as F

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