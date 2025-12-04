#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task (b): Clustering of graph embeddings.

- Embeddings: Graph2Vec, NetLSD (robust dense-eigs impl), GIN (supervised encoder -> embeddings)
- Datasets: MUTAG, ENZYMES, IMDB-MULTI (configurable via CLI)
- Clustering: KMeans, SpectralClustering
- Metrics: ARI (primary), Silhouette (secondary)
- Visuals: t-SNE and UMAP (colored by gold labels)
- Compatibility: patches for SciPy errstate and UMAP↔sklearn check_array mismatch

Outputs:
  - report/tables/clustering_eval.csv          (per run / per seed)
  - report/tables/clustering_eval_agg.csv      (mean/std over seeds)
  - report/tables/clustering_eval_top.csv      (best by ARI per dataset)
  - report/figures/*_{tsne,umap}.png           (per dataset/method/dim)
"""

<<<<<<< Updated upstream
# ---------------- Headless plotting + compat patches ----------------
=======
# ============================================================================
# ENVIRONMENT SETUP & COMPATIBILITY PATCHES
# ============================================================================

>>>>>>> Stashed changes
import matplotlib
matplotlib.use("Agg")  # Headless mode for plotting (no display needed)

<<<<<<< Updated upstream
# Patch SciPy.errstate if missing (seen in some NumPy/SciPy mixes)
import numpy as _np
import scipy as _sp
if not hasattr(_sp, "errstate"):
    _sp.errstate = _np.errstate

# Patch UMAP's internal check_array to ignore ensure_all_finite for older sklearn
try:
    import inspect
    from sklearn.utils import validation as _suv
    import umap.umap_ as _umap_mod
    if "ensure_all_finite" not in inspect.signature(_suv.check_array).parameters:
        _orig = _suv.check_array
        def _check_array_wrapper(*args, ensure_all_finite=None, **kwargs):
            return _orig(*args, **kwargs)
        _umap_mod.check_array = _check_array_wrapper
except Exception:
    pass
# --------------------------------------------------------------------

import os, time, argparse, json, warnings
warnings.filterwarnings("ignore")
=======
import os, time, argparse, json, warnings, inspect
warnings.filterwarnings("ignore")  # Suppress library warnings
>>>>>>> Stashed changes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from contextlib import contextmanager

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.manifold import TSNE
<<<<<<< Updated upstream
=======
from sklearn.preprocessing import StandardScaler

# ============================================================================
# COMPATIBILITY PATCHES (same as Task 1)
# ============================================================================

# PATCH 1: Fix missing scipy.errstate in some environments
# NetworkX expects scipy.errstate to exist, but some builds don't have it
import numpy as _np
import scipy as _sp
if not hasattr(_sp, "errstate"):
    _sp.errstate = _np.errstate  # Borrow from NumPy

# PATCH 2: Fix UMAP compatibility with older scikit-learn
# Newer UMAP calls sklearn's check_array with 'ensure_all_finite' parameter
# that older sklearn versions don't support
import sklearn.utils.validation as _suv
import sklearn.utils as _su

if "ensure_all_finite" not in inspect.signature(_suv.check_array).parameters:
    _orig_check_array = _suv.check_array

    def _wrapped_check_array(*args, ensure_all_finite=None, **kwargs):
        # Ignore the unsupported parameter
        return _orig_check_array(*args, **kwargs)

    # Patch both locations where UMAP might import from
    _suv.check_array = _wrapped_check_array
    _su.check_array  = _wrapped_check_array

# Now safe to import UMAP after patches
>>>>>>> Stashed changes
try:
    from umap import UMAP
except Exception:
    from umap.umap_ import UMAP

<<<<<<< Updated upstream
=======
# Import graph libraries after patches
>>>>>>> Stashed changes
import networkx as nx
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool

<<<<<<< Updated upstream
from karateclub import Graph2Vec  # NetLSD replaced by robust local impl below
=======
from karateclub import Graph2Vec  # Unsupervised graph embedding
>>>>>>> Stashed changes

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================

OUT_DIR_TABLES = "report/tables"
OUT_DIR_FIGS   = "report/figures"
os.makedirs(OUT_DIR_TABLES, exist_ok=True)
os.makedirs(OUT_DIR_FIGS,   exist_ok=True)
os.makedirs("report/logs",  exist_ok=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ds_labels(ds):
    """Extract integer labels from PyTorch Geometric dataset."""
    return np.array([int(g.y) for g in ds])

def infer_num_node_features(ds):
    """Return node feature size; if missing, fall back to degree one-hot up to min(10, max degree)."""
    x0 = getattr(ds[0], "x", None)
    if x0 is not None and x0 is not False:
        return ds.num_node_features
    # Create synthetic degree feature later in collate; here just signal 1 (degree scalar)
    return 1

def attach_degree_as_feature(graph):
<<<<<<< Updated upstream
    """If a graph has no x, create a single degree feature column."""
=======
    """
    Add node degree as a feature if graph has no node attributes.
    
    Some datasets (e.g., IMDB-MULTI) have no node features. GIN requires
    node features, so we create a single feature: node degree.
    
    Args:
        graph: PyG Data object
        
    Returns:
        graph: Same graph with x attribute added (if it was missing)
    """
>>>>>>> Stashed changes
    if getattr(graph, "x", None) is None:
        # Convert to NetworkX to easily compute degrees
        G = to_networkx(graph, to_undirected=True)
        deg = np.array([d for _, d in G.degree()], dtype=np.float32)
        # Add as single-column feature matrix
        graph.x = torch.from_numpy(deg).view(-1, 1)
    return graph

def to_nx_with_labels(ds_slice):
<<<<<<< Updated upstream
    """Convert PyG graphs to undirected NetworkX + categorical node labels (degree) for Graph2Vec."""
=======
    """
    Convert PyG graphs to NetworkX format with discrete node labels.
    
    Graph2Vec requires each node to have a discrete 'label' attribute.
    We use node degree as the label (standard practice).
    
    Args:
        ds_slice: List of PyG graph objects
        
    Returns:
        List of NetworkX graphs with 'label' attribute on each node
    """
>>>>>>> Stashed changes
    Gs = []
    for g in ds_slice:
        G = to_networkx(g, to_undirected=True)
        degs = dict(G.degree())
        # Assign degree as discrete label for each node
        for n in G.nodes():
            G.nodes[n]["label"] = int(degs[n])
        Gs.append(G)
    return Gs

@contextmanager
def timed(name="block"):
    """
    Simple timer context manager.
    
    Usage:
        with timed("my_function"):
            # your code
        # Prints: [my_function] time=1.23s
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(f"[{name}] time={time.perf_counter()-t0:.2f}s")

# ============================================================================
# EMBEDDING METHODS: UNSUPERVISED (Graph2Vec, NetLSD)
# ============================================================================

<<<<<<< Updated upstream
# ---------------- Embeddings: Graph2Vec & NetLSD ----------------
def embed_graph2vec(ds_slice, dim=128, seed=0, epochs=20, wl_iterations=2, min_count=5):
=======
def embed_graph2vec(ds_slice, dim=128, seed=0, epochs=20,
                    wl_iterations=2, min_count=5):
    """
    Generate Graph2Vec embeddings.
    
    Graph2Vec is an unsupervised method that:
    1. Extracts substructures using Weisfeiler-Lehman (WL) kernel
    2. Treats each graph as a "document" and substructures as "words"
    3. Applies doc2vec (similar to Word2Vec) to learn embeddings
    
    Args:
        ds_slice: List of PyG graphs
        dim: Embedding dimension
        seed: Random seed for reproducibility
        epochs: Number of training epochs for doc2vec
        wl_iterations: Depth of WL kernel (how far to look for patterns)
        min_count: Minimum frequency for substructure to be included
        
    Returns:
        X: Embedding matrix [num_graphs, dim]
    """
    # Convert to NetworkX with node labels
>>>>>>> Stashed changes
    Gs = to_nx_with_labels(ds_slice)
    
    with timed("Graph2Vec"):
<<<<<<< Updated upstream
        model = Graph2Vec(dimensions=dim, wl_iterations=wl_iterations, epochs=epochs,
                          seed=seed, workers=1, min_count=min_count)
=======
        # Initialize and train Graph2Vec model
        model = Graph2Vec(
            dimensions=dim,
            wl_iterations=wl_iterations,  # Substructure depth
            epochs=epochs,
            seed=seed,
            workers=1,  # Single-threaded for reproducibility
            min_count=min_count,
        )
>>>>>>> Stashed changes
        model.fit(Gs)
        X = model.get_embedding()
    return X

# Robust NetLSD (dense eigendecomposition; no ARPACK k<=0 issue)
def _netlsd_signature_dense(G, times):
<<<<<<< Updated upstream
=======
    """
    Compute NetLSD heat trace signature using dense eigendecomposition.
    
    NetLSD (Network Laplacian Spectral Descriptor) characterizes graphs
    through heat diffusion. The heat trace measures how heat spreads over
    the graph at different time scales.
    
    Math:
    - Normalized Laplacian: L = I - D^(-1/2) * A * D^(-1/2)
    - Eigenvalues: λ₁, λ₂, ..., λₙ (from L)
    - Heat trace at time t: h(t) = Σᵢ exp(-t·λᵢ)
    
    This signature captures both local and global graph structure.
    
    Args:
        G: NetworkX graph
        times: Array of diffusion times (log-spaced is typical)
        
    Returns:
        Array of heat trace values [one per time point]
    """
>>>>>>> Stashed changes
    n = G.number_of_nodes()
    if n == 0:
        return np.zeros_like(times)
    
    # Compute normalized Laplacian (symmetric matrix)
    L = nx.normalized_laplacian_matrix(G).astype(float).toarray()
<<<<<<< Updated upstream
    lam = np.linalg.eigvalsh(L)                      # symmetric -> stable
    return np.exp(-np.outer(times, lam)).sum(axis=1) # heat trace

def embed_netlsd(ds_slice, dim=128, pca_seed=0, n_times=256, t_min=1e-2, t_max=1e2):
=======
    
    # Get eigenvalues (always real for symmetric matrices)
    lam = np.linalg.eigvalsh(L)
    
    # Compute heat trace: h(t) = sum_i exp(-t * lambda_i)
    # Using outer product to vectorize: [times, eigenvalues] → [times]
    return np.exp(-np.outer(times, lam)).sum(axis=1)

def embed_netlsd(ds_slice, dim=128, pca_seed=0,
                 n_times=256, t_min=1e-2, t_max=1e2):
    """
    Generate NetLSD embeddings with PCA compression.
    
    Process:
    1. Compute heat trace signatures at multiple time scales
    2. Each graph → vector of heat traces [n_times dimensions]
    3. Apply PCA to compress to desired dimension
    
    Why PCA? Raw signatures are high-dimensional (n_times=256).
    PCA keeps most important information in fewer dimensions.
    
    Args:
        ds_slice: List of PyG graphs
        dim: Target embedding dimension after PCA
        pca_seed: Random seed for PCA
        n_times: Number of time points for heat trace
        t_min, t_max: Time range (log-spaced from t_min to t_max)
        
    Returns:
        X: Embedding matrix [num_graphs, dim]
    """
    # Logarithmically-spaced time points (captures multiple scales)
>>>>>>> Stashed changes
    times = np.logspace(np.log10(t_min), np.log10(t_max), num=n_times)
    
    # Convert to NetworkX
    Gs = [to_networkx(g, to_undirected=True) for g in ds_slice]
    
    with timed("NetLSD"):
        # Compute heat trace signature for each graph
        sigs = [_netlsd_signature_dense(G, times) for G in Gs]
<<<<<<< Updated upstream
        X = np.vstack(sigs)  # [n_graphs, n_times]
=======
        X = np.vstack(sigs)  # Stack into matrix [num_graphs, n_times]
        
        # Apply PCA if target dimension differs from raw signature size
>>>>>>> Stashed changes
        if dim != X.shape[1]:
            X = PCA(n_components=dim, random_state=pca_seed).fit_transform(X)
    
    return X

# ============================================================================
# EMBEDDING METHOD: SUPERVISED (GIN)
# ============================================================================

class GINEncoder(nn.Module):
    """
    Graph Isomorphism Network (GIN) encoder.
    
    GIN is a powerful Graph Neural Network that:
    1. Performs message passing between neighbors
    2. Aggregates information with learnable MLPs
    3. Pools node features to get graph-level embedding
    4. Adds classification head for supervised training
    
    Architecture:
        Node features → GIN layers (message passing) → Graph pooling → Embedding
                                                                    ↓
                                                            Classification head
    
    We train GIN with labels (supervised), then extract the graph embedding
    from the penultimate layer (before the classification head).
    """
    
    def __init__(self, in_dim, hidden=64, layers=3, dropout=0.2, n_classes=2):
        """
        Args:
            in_dim: Input feature dimension (e.g., 1 for degree-only features)
            hidden: Hidden layer dimension (also the embedding dimension)
            layers: Number of GIN layers (depth of message passing)
            dropout: Dropout probability for regularization
            n_classes: Number of classes (for the classification head)
        """
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        
        # Build GIN layers
        h = in_dim
        for _ in range(layers):
            # Each GIN layer has an MLP that processes node features
            mlp = nn.Sequential(
                nn.Linear(h, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
            )
            self.layers.append(GINConv(mlp))  # GIN convolution with this MLP
            h = hidden
<<<<<<< Updated upstream
        self.head = nn.Linear(hidden, n_classes)  # used only during training
=======
        
        # Classification head (used during training, ignored for embeddings)
        self.head = nn.Linear(hidden, n_classes)
>>>>>>> Stashed changes

    def forward(self, data):
        """
        Forward pass through GIN.
        
        Args:
            data: PyG batch with attributes:
                - x: Node features [num_nodes, in_dim]
                - edge_index: Graph connectivity [2, num_edges]
                - batch: Batch assignment for each node [num_nodes]
        
        Returns:
            logits: Classification predictions [num_graphs, n_classes]
            g: Graph embeddings [num_graphs, hidden] ← This is what we extract!
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Message passing through GIN layers
        for conv in self.layers:
<<<<<<< Updated upstream
            x = conv(x, edge_index)
            x = F.relu(x)
=======
            x = conv(x, edge_index)  # GIN convolution
            x = torch.relu(x)
>>>>>>> Stashed changes
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph-level pooling (aggregate node features → graph feature)
        g = global_mean_pool(x, batch)  # [num_graphs, hidden] ← EMBEDDING!
        
        # Classification head (for training supervision)
        logits = self.head(g)
        
        return logits, g

def train_gin_get_embeddings(ds_slice, dim=128, seed=0, hidden=64, layers=3, dropout=0.2,
                             epochs=30, batch_size=64, lr=1e-3, device="cpu"):
    """
<<<<<<< Updated upstream
    Train a small supervised GIN to learn an encoder, then return graph embeddings (penultimate layer).
    Embedding dimension returned is `hidden`. If `dim` != hidden, we apply PCA to `dim`.
=======
    Train GIN with supervision, then extract graph embeddings.
    
    Process:
    1. Prepare graphs (add degree features if needed)
    2. Train GIN model with labels (supervised learning)
    3. Extract embeddings from penultimate layer
    4. Apply PCA if hidden dimension ≠ target dimension
    
    Why train GIN for clustering task?
    - Tests if supervised training (which sees labels) produces better
      embeddings for unsupervised clustering than methods that never see labels
    - Expected: GIN should cluster better (it "cheated" by seeing labels)
    
    Args:
        ds_slice: List of PyG graphs
        dim: Target embedding dimension (after optional PCA)
        seed: Random seed
        hidden: Hidden dimension of GIN (before PCA)
        layers: Number of GIN layers
        dropout: Dropout probability
        epochs: Training epochs
        batch_size: Mini-batch size for training
        lr: Learning rate
        device: 'cpu' or 'cuda'
        
    Returns:
        X: Graph embeddings [num_graphs, dim]
>>>>>>> Stashed changes
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

<<<<<<< Updated upstream
    # Ensure each graph has features
=======
    # Ensure all graphs have node features (use degree if missing)
>>>>>>> Stashed changes
    graphs = [attach_degree_as_feature(g.clone()) for g in ds_slice]
    
    # Determine number of classes and input dimension
    num_classes = len(np.unique([int(g.y) for g in graphs]))
    in_dim = graphs[0].x.size(-1)

<<<<<<< Updated upstream
    model = GINEncoder(in_dim, hidden=hidden, layers=layers, dropout=dropout, n_classes=num_classes).to(device)
=======
    # Initialize GIN model
    model = GINEncoder(
        in_dim,
        hidden=hidden,
        layers=layers,
        dropout=dropout,
        n_classes=num_classes,
    ).to(device)

    # Optimizer and data loader
>>>>>>> Stashed changes
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)

    # ===== TRAINING PHASE =====
    with timed("GIN-train"):
        model.train()
        for _ in range(epochs):
            for batch in loader:
                batch = batch.to(device)
                
                # Forward pass
                logits, _ = model(batch)
                
                # Compute loss (cross-entropy)
                loss = F.cross_entropy(logits, batch.y.view(-1))
                
                # Backward pass
                opt.zero_grad()
                loss.backward()
                opt.step()

<<<<<<< Updated upstream
    # Extract embeddings
    with timed("GIN-embed"):
        model.eval()
        emb_all = []
        loader_eval = DataLoader(graphs, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for batch in loader_eval:
                batch = batch.to(device)
                _, g = model(batch)
                emb_all.append(g.cpu())
        X = torch.cat(emb_all, dim=0).numpy()  # [n_graphs, hidden]
=======
    # ===== EMBEDDING EXTRACTION PHASE =====
    with timed("GIN-embed"):
        model.eval()  # Set to evaluation mode
        chunks = []
        eval_loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():  # No gradients needed
            for batch in eval_loader:
                batch = batch.to(device)
                _, g = model(batch)  # Get embeddings (ignore logits)
                chunks.append(g.cpu())
        
        # Concatenate all embeddings
        X = torch.cat(chunks, dim=0).numpy()  # [num_graphs, hidden]
>>>>>>> Stashed changes

    # Apply PCA if target dimension differs from hidden dimension
    if dim != X.shape[1]:
        X = PCA(n_components=dim, random_state=seed).fit_transform(X)
    return X

# ============================================================================
# CLUSTERING & EVALUATION
# ============================================================================

<<<<<<< Updated upstream
# ---------------- Clustering & metrics ----------------
def cluster_and_score(X, y, n_clusters, seed, algo="kmeans"):
    if algo == "kmeans":
        model = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
        pred = model.fit_predict(X)
=======
def preprocess_for_clustering(X):
    """
    Standardize embeddings before clustering.
    
    WHY THIS IS CRITICAL:
    Different embedding dimensions can have vastly different scales:
    
    Example WITHOUT standardization:
        Feature 0: range [0.1, 0.2]    (small scale)
        Feature 1: range [100, 200]    (large scale)
        Feature 2: range [0.001, 0.002] (tiny scale)
    
    Distance calculation:
        ||x - y||² ≈ (Δf₁)² + (Δf₂)² + (Δf₃)²
                   ≈ 0.01² + 100² + 0.001²
                   ≈ 10,000  (dominated by feature 1!)
    
    After StandardScaler (zero mean, unit variance):
        All features: mean=0, std=1
        Distance: all features contribute equally
    
    This makes clustering fair across different embedding methods and dimensions.
    
    Args:
        X: Raw embedding matrix [num_graphs, dim]
        
    Returns:
        X_scaled: Standardized embedding matrix (mean=0, std=1 per feature)
    """
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def cluster_and_score(X, y, n_clusters, seed, algo="kmeans"):
    """
    Apply clustering and compute evaluation metrics.
    
    Steps:
    1. Standardize embeddings (critical for distance-based methods)
    2. Apply clustering algorithm
    3. Compute ARI (match with true labels)
    4. Compute Silhouette (internal cluster quality)
    
    Args:
        X: Raw embedding matrix [num_graphs, dim]
        y: True labels [num_graphs]
        n_clusters: Number of clusters (usually = number of true classes)
        seed: Random seed
        algo: "kmeans" or "spectral"
        
    Returns:
        dict with:
            - ari: Adjusted Rand Index (how well clusters match true labels)
            - silhouette: Silhouette score (cluster compactness & separation)
            - labels: Predicted cluster assignments
            - X_proc: Standardized embeddings (for visualization)
    """
    # Step 1: Standardize
    X_proc = preprocess_for_clustering(X)

    # Step 2: Apply clustering
    if algo == "kmeans":
        # K-Means: Partition into k clusters by minimizing within-cluster variance
        model = KMeans(
            n_clusters=n_clusters,
            n_init=20,  # Run 20 times with different initializations (robustness)
            random_state=seed
        )
        pred = model.fit_predict(X_proc)
        
>>>>>>> Stashed changes
    elif algo == "spectral":
        # Spectral Clustering: Build similarity graph, then cluster eigenvectors
        # Can find non-convex clusters (more flexible than K-Means)
        model = SpectralClustering(
            n_clusters=n_clusters,
<<<<<<< Updated upstream
            assign_labels="kmeans",
            affinity="rbf",       # robust on compact embeddings
            random_state=seed
=======
            assign_labels="kmeans",  # Final assignment via k-means on eigenvectors
            affinity="rbf",  # Gaussian (RBF) similarity
            random_state=seed,
>>>>>>> Stashed changes
        )
        pred = model.fit_predict(X)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    # Step 3: Compute ARI (Adjusted Rand Index)
    # ARI = 1.0 → Perfect match with true labels
    # ARI = 0.0 → No better than random
    # ARI < 0.0 → Worse than random
    ari = adjusted_rand_score(y, pred)
<<<<<<< Updated upstream
    try:
        # silhouette requires at least 2 clusters present
=======

    # Step 4: Compute Silhouette Score
    # Silhouette = +1 → Perfect clusters (tight and well-separated)
    # Silhouette =  0 → Overlapping clusters
    # Silhouette = -1 → Wrong cluster assignments
    sil = np.nan
    try:
        # Need at least 2 clusters for silhouette
>>>>>>> Stashed changes
        if len(np.unique(pred)) > 1:
            sil = silhouette_score(X, pred)
        else:
            sil = np.nan
    except Exception:
        sil = np.nan

    return {
        "ari": float(ari),
        "silhouette": float(np.nan if np.isnan(sil) else sil),
<<<<<<< Updated upstream
        "labels": pred
=======
        "labels": pred,  # Cluster assignments
        "X_proc": X_proc,  # Standardized embeddings (for t-SNE/UMAP)
>>>>>>> Stashed changes
    }

# ============================================================================
# VISUALIZATION: t-SNE & UMAP
# ============================================================================

<<<<<<< Updated upstream
# ---------------- Visualization ----------------
def scatter_2d(X2, y, title, outpath):
    fig = plt.figure(figsize=(5,5))
=======
def scatter_2d(X2, labels, title, outpath):
    """
    Create a 2D scatter plot colored by labels.
    
    Args:
        X2: 2D coordinates [num_points, 2]
        labels: Color for each point (can be true labels or cluster assignments)
        title: Plot title
        outpath: Where to save the figure
    """
    fig = plt.figure(figsize=(5, 5))
>>>>>>> Stashed changes
    ax = fig.add_subplot(111)
    ax.scatter(X2[:,0], X2[:,1], c=y, s=16)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

<<<<<<< Updated upstream
def plot_tsne_umap(X, y, title_prefix, out_prefix, tsne_seed=0, umap_seed=0):
    # t-SNE
    T = TSNE(n_components=2, random_state=tsne_seed, init="pca",
             perplexity=min(30, max(5, len(X)//10)))
    Xt = T.fit_transform(X)
    scatter_2d(Xt, y, f"{title_prefix} — t-SNE", f"{out_prefix}_tsne.png")

    # UMAP
    n_neighbors = min(15, max(2, len(X)-1))
    U = UMAP(n_components=2, random_state=umap_seed, n_neighbors=n_neighbors, min_dist=0.1)
    Xu = U.fit_transform(X)
    scatter_2d(Xu, y, f"{title_prefix} — UMAP", f"{out_prefix}_umap.png")
=======
def plot_tsne_umap(X_proc, y_true, y_clusters, title_prefix, out_prefix,
                   tsne_seed=0, umap_seed=0):
    """
    Generate t-SNE and UMAP visualizations for embeddings.
    
    Creates 4 plots:
    1. t-SNE colored by TRUE labels  → Shows if same-class graphs are close
    2. t-SNE colored by CLUSTER labels → Shows what clustering found
    3. UMAP colored by TRUE labels
    4. UMAP colored by CLUSTER labels
    
    Comparing (1) vs (2): If similar, clustering worked well (high ARI)
    
    t-SNE vs UMAP:
    - t-SNE: Preserves local structure (neighbors stay neighbors)
    - UMAP: Preserves both local AND global structure (better overall view)
    
    Args:
        X_proc: Standardized embeddings [num_graphs, dim]
        y_true: True class labels
        y_clusters: Predicted cluster assignments
        title_prefix: Prefix for plot titles
        out_prefix: Prefix for output filenames
        tsne_seed, umap_seed: Random seeds for reproducibility
    """
    
    # ===== t-SNE VISUALIZATION =====
    try:
        # t-SNE: Nonlinear dimensionality reduction (high-dim → 2D)
        T = TSNE(
            n_components=2,
            random_state=tsne_seed,
            init="pca",  # Initialize with PCA (faster convergence)
            perplexity=min(30, max(5, len(X_proc)//10)),  # Adaptive perplexity
        )
        Xt = T.fit_transform(X_proc)

        # Plot 1: Colored by TRUE labels
        scatter_2d(
            Xt,
            y_true,
            f"{title_prefix} — t-SNE (true)",
            f"{out_prefix}_tsne_true.png",
        )
        
        # Plot 2: Colored by CLUSTER assignments
        scatter_2d(
            Xt,
            y_clusters,
            f"{title_prefix} — t-SNE (clusters)",
            f"{out_prefix}_tsne_clusters.png",
        )
    except Exception as e:
        print(f"[warn] t-SNE plot failed: {e}")

    # ===== UMAP VISUALIZATION =====
    try:
        # UMAP: Modern alternative to t-SNE (often better global structure)
        n_neighbors = min(15, max(2, len(X_proc) - 1))
        U = UMAP(
            n_components=2,
            random_state=umap_seed,
            n_neighbors=n_neighbors,
            min_dist=0.1,
        )
        Xu = U.fit_transform(X_proc)

        # Plot 3: Colored by TRUE labels
        scatter_2d(
            Xu,
            y_true,
            f"{title_prefix} — UMAP (true)",
            f"{out_prefix}_umap_true.png",
        )
        
        # Plot 4: Colored by CLUSTER assignments
        scatter_2d(
            Xu,
            y_clusters,
            f"{title_prefix} — UMAP (clusters)",
            f"{out_prefix}_umap_clusters.png",
        )
    except Exception as e:
        print(f"[warn] UMAP plot failed: {e}")
>>>>>>> Stashed changes

# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

<<<<<<< Updated upstream
# ---------------- Runner ----------------
def run(datasets, methods, dims, seeds, plot_policy="first_seed",
        out_csv=f"{OUT_DIR_TABLES}/clustering_eval.csv",
        gin_hidden=64, gin_layers=3, gin_dropout=0.2, gin_epochs=30, gin_batch=64, device="cpu"):
=======
def run(
    datasets,
    methods,
    dims,
    seeds,
    plot_policy="first_seed",
    out_csv=f"{OUT_DIR_TABLES}/clustering_eval.csv",
    gin_hidden=64,
    gin_layers=3,
    gin_dropout=0.2,
    gin_epochs=30,
    gin_batch=64,
    device="cpu",
):
>>>>>>> Stashed changes
    """
    Run the complete clustering experiment.
    
    For each combination of (dataset, seed, method, dimension):
    1. Generate embeddings
    2. Apply K-Means clustering
    3. Apply Spectral clustering
    4. Compute ARI and Silhouette for both
    5. Optionally generate t-SNE/UMAP visualizations
    6. Save results progressively
    
    Args:
        datasets: List of dataset names (e.g., ["MUTAG", "ENZYMES"])
        methods: List of embedding methods (e.g., ["graph2vec", "netlsd", "gin"])
        dims: List of embedding dimensions (e.g., [32, 64, 128])
        seeds: List of random seeds for reproducibility
        plot_policy: 'none' | 'first_seed' | 'all' - controls visualization frequency
        out_csv: Path to output CSV file
        gin_hidden: Hidden dimension for GIN
        gin_layers: Number of GIN layers
        gin_dropout: Dropout probability for GIN
        gin_epochs: Training epochs for GIN
        gin_batch: Batch size for GIN
        device: 'cpu' or 'cuda'
        
    Returns:
        DataFrame with all results
    """
    rows = []
    for ds_name in datasets:
        print(f"\n=== Dataset: {ds_name} ===")
        
        # Load dataset
        ds = TUDataset(root="data", name=ds_name)
        y = ds_labels(ds)  # True labels
        graphs = [ds[i] for i in range(len(ds))]
        n_clusters = len(np.unique(y))  # Use true number of classes for k
        
        first_seed = min(seeds) if len(seeds) else 0

        for seed in seeds:
            for method in methods:
                for dim in dims:
                    print(f"\n--- {method} | dim={dim} | seed={seed} ---")
<<<<<<< Updated upstream
                    # Embeddings
=======

                    # ===== STEP 1: GENERATE EMBEDDINGS =====
>>>>>>> Stashed changes
                    if method.lower() == "graph2vec":
                        X = embed_graph2vec(graphs, dim=dim, seed=seed, epochs=20)
                    elif method.lower() == "netlsd":
                        X = embed_netlsd(graphs, dim=dim, pca_seed=seed)
                    elif method.lower() == "gin":
                        X = train_gin_get_embeddings(
                            graphs, dim=dim, seed=seed,
                            hidden=gin_hidden, layers=gin_layers, dropout=gin_dropout,
                            epochs=gin_epochs, batch_size=gin_batch, device=device
                        )
                    else:
                        raise ValueError(f"Unknown method: {method}")

<<<<<<< Updated upstream
                    # Clustering
=======
                    # ===== STEP 2: CLUSTERING & METRICS =====
                    
                    # K-Means clustering
>>>>>>> Stashed changes
                    res_km = cluster_and_score(X, y, n_clusters, seed, algo="kmeans")
                    
                    # Spectral clustering
                    res_sp = cluster_and_score(X, y, n_clusters, seed, algo="spectral")

                    # Prepare result rows
                    row_km = dict(
                        dataset=ds_name, method=method, dim=dim, seed=seed, algo="kmeans",
                        ari=round(res_km["ari"], 4),
                        silhouette=float(np.nan if np.isnan(res_km["silhouette"]) else round(res_km["silhouette"], 4)),
                        n_graphs=len(ds), n_clusters=n_clusters
                    )
                    
                    row_sp = dict(
                        dataset=ds_name, method=method, dim=dim, seed=seed, algo="spectral",
                        ari=round(res_sp["ari"], 4),
                        silhouette=float(np.nan if np.isnan(res_sp["silhouette"]) else round(res_sp["silhouette"], 4)),
                        n_graphs=len(ds), n_clusters=n_clusters
                    )
<<<<<<< Updated upstream
                    print("KMeans:", json.dumps(row_km))
                    print("Spectral:", json.dumps(row_sp))
                    rows.extend([row_km, row_sp])

                    # Plots
=======

                    # Print results for monitoring
                    print("KMeans:", json.dumps(row_km))
                    print("Spectral:", json.dumps(row_sp))

                    # Append both clustering results
                    rows.extend([row_km, row_sp])

                    # ===== STEP 3: VISUALIZATION (OPTIONAL) =====
                    # Generate plots based on policy:
                    # - 'none': skip all plots (fastest)
                    # - 'first_seed': only plot for first seed (default)
                    # - 'all': plot for every seed (most thorough)
>>>>>>> Stashed changes
                    do_plots = (
                        plot_policy == "all" or
                        (plot_policy == "first_seed" and seed == first_seed)
                    )
                    
                    if do_plots:
                        prefix = f"{OUT_DIR_FIGS}/{ds_name}_{method}_d{dim}"
                        title  = f"{ds_name} | {method} (d={dim})"
<<<<<<< Updated upstream
                        try:
                            plot_tsne_umap(X, y, title, prefix)
                        except Exception as e:
                            print(f"[warn] plot failed: {e}")

                    # Progressive write
=======
                        
                        # Use K-Means results for visualization
                        # (could also use Spectral, but one is enough)
                        plot_tsne_umap(
                            X_proc=res_km["X_proc"],  # Standardized embeddings
                            y_true=y,  # True class labels
                            y_clusters=res_km["labels"],  # K-Means assignments
                            title_prefix=title,
                            out_prefix=prefix,
                            tsne_seed=seed,
                            umap_seed=seed,
                        )

                    # ===== STEP 4: PROGRESSIVE SAVE =====
                    # Save after each configuration so we don't lose results
                    # if the script crashes or is interrupted
>>>>>>> Stashed changes
                    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # Final DataFrame
    df = pd.DataFrame(rows)
    print(f"\nSaved per-run clustering results to {out_csv}")
    return df

<<<<<<< Updated upstream
def aggregate_and_rank(df,
                       out_csv_agg=f"{OUT_DIR_TABLES}/clustering_eval_agg.csv",
                       out_csv_top=f"{OUT_DIR_TABLES}/clustering_eval_top.csv"):
=======
# ============================================================================
# AGGREGATION & RANKING
# ============================================================================

def aggregate_and_rank(
    df,
    out_csv_agg=f"{OUT_DIR_TABLES}/clustering_eval_agg.csv",
    out_csv_top=f"{OUT_DIR_TABLES}/clustering_eval_top.csv",
):
    """
    Aggregate results across random seeds and identify best configurations.
    
    Creates two summary tables:
    1. Aggregated results: mean ± std for each configuration
    2. Top configurations: best method/dim/algo per dataset (by ARI)
    
    Args:
        df: Raw results DataFrame from run()
        out_csv_agg: Path for aggregated results
        out_csv_top: Path for top configurations
        
    Returns:
        agg: Aggregated DataFrame
        top_df: Top configurations DataFrame
    """
>>>>>>> Stashed changes
    if df.empty:
        print("No rows to aggregate.")
        return df, None

<<<<<<< Updated upstream
    agg = df.groupby(["dataset", "method", "dim", "algo"]).agg(
        ari_mean=("ari", "mean"),   ari_std=("ari", "std"),
        sil_mean=("silhouette", "mean"), sil_std=("silhouette", "std"),
        n_runs=("ari", "count"),
    ).reset_index()
=======
    # ===== AGGREGATION: COMPUTE MEAN & STD ACROSS SEEDS =====
    agg = (
        df.groupby(["dataset", "method", "dim", "algo"])
        .agg(
            ari_mean=("ari", "mean"),
            ari_std=("ari", "std"),
            sil_mean=("silhouette", "mean"),
            sil_std=("silhouette", "std"),
            n_runs=("ari", "count"),  # How many seeds were used
        )
        .reset_index()
    )

    # Fill NaN std with 0.0 (happens when only 1 seed is used)
    # This makes tables cleaner (0.0 instead of NaN)
    agg["ari_std"] = agg["ari_std"].fillna(0.0)
    agg["sil_std"] = agg["sil_std"].fillna(0.0)
>>>>>>> Stashed changes

    # Save aggregated results
    agg.to_csv(out_csv_agg, index=False)
    print(f"Saved aggregated results to {out_csv_agg}")

<<<<<<< Updated upstream
    # Best by ARI per dataset (tie-break by Silhouette)
=======
    # ===== RANKING: FIND BEST CONFIGURATION PER DATASET =====
    # Best = highest ARI (primary metric)
    # If tie, use Silhouette as tiebreaker
>>>>>>> Stashed changes
    tops = []
    for ds in agg["dataset"].unique():
        sub = agg[agg["dataset"] == ds]
        
        # Sort by ARI (descending), then Silhouette (descending)
        best = sub.sort_values(
            ["ari_mean", "sil_mean"],
            ascending=False
        ).head(1)  # Take top row
        
        tops.append(best)
        print(f"\nBest separation for {ds}:")
        print(best.to_string(index=False))

    # Combine best configurations into one table
    top_df = pd.concat(tops, ignore_index=True)
    top_df.to_csv(out_csv_top, index=False)
    print(f"Saved top-by-ARI table to {out_csv_top}")
    
    return agg, top_df

# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def parse_args():
<<<<<<< Updated upstream
    p = argparse.ArgumentParser(description="Task (b): Clustering of graph embeddings (Graph2Vec, NetLSD, GIN)")
    p.add_argument("--datasets", nargs="+", default=["MUTAG", "ENZYMES", "IMDB-MULTI"])
    p.add_argument("--methods",  nargs="+", default=["graph2vec", "netlsd", "gin"])
    p.add_argument("--dims",     nargs="+", type=int, default=[32, 64, 128])
    p.add_argument("--seeds",    nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--plot_policy", choices=["none", "first_seed", "all"], default="first_seed",
                   help="Save t-SNE/UMAP plots per (dataset, method, dim).")
    # GIN options
    p.add_argument("--gin_hidden", type=int, default=64)
    p.add_argument("--gin_layers", type=int, default=3)
    p.add_argument("--gin_dropout", type=float, default=0.2)
    p.add_argument("--gin_epochs", type=int, default=30)
    p.add_argument("--gin_batch", type=int, default=64)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()

def main():
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
=======
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Task (b): Clustering of graph embeddings (Graph2Vec, NetLSD, GIN)"
    )
    
    # Dataset selection
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["MUTAG", "ENZYMES", "IMDB-MULTI"],
        help="Datasets to evaluate"
    )
    
    # Embedding methods
    p.add_argument(
        "--methods",
        nargs="+",
        default=["graph2vec", "netlsd", "gin"],
        help="Embedding methods to compare"
    )
    
    # Embedding dimensions
    p.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=[32, 64, 128],
        help="Embedding dimensions to test"
    )
    
    # Random seeds
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Random seeds for reproducibility"
    )
    
    # Visualization policy
    p.add_argument(
        "--plot_policy",
        choices=["none", "first_seed", "all"],
        default="first_seed",
        help="Save t-SNE/UMAP plots per (dataset, method, dim)."
    )
    
    # GIN hyperparameters
    p.add_argument("--gin_hidden", type=int, default=64,
                   help="Hidden dimension for GIN encoder")
    p.add_argument("--gin_layers", type=int, default=3,
                   help="Number of GIN layers")
    p.add_argument("--gin_dropout", type=float, default=0.2,
                   help="Dropout probability for GIN")
    p.add_argument("--gin_epochs", type=int, default=30,
                   help="Training epochs for GIN")
    p.add_argument("--gin_batch", type=int, default=64,
                   help="Batch size for GIN training")
    p.add_argument("--device", type=str, default="cpu",
                   help="Device for GIN training (cpu or cuda)")
    
    return p.parse_args()

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Main execution function.
    
    Workflow:
    1. Parse command-line arguments
    2. Run clustering experiments
    3. Aggregate results across seeds
    4. Identify best configurations
    5. Save all outputs
    """
    # Keep environment predictable (avoid user site-packages)
    os.environ.setdefault("PYTHONNOUSERSITE", "1")

    # Parse arguments
>>>>>>> Stashed changes
    args = parse_args()

    # ===== RUN EXPERIMENTS =====
    df = run(
        datasets=args.datasets,
        methods=args.methods,
        dims=args.dims,
        seeds=args.seeds,
        plot_policy=args.plot_policy,
        out_csv=f"{OUT_DIR_TABLES}/clustering_eval.csv",
        gin_hidden=args.gin_hidden,
        gin_layers=args.gin_layers,
        gin_dropout=args.gin_dropout,
        gin_epochs=args.gin_epochs,
        gin_batch=args.gin_batch,
        device=args.device,
    )
<<<<<<< Updated upstream
    aggregate_and_rank(df)

if __name__ == "__main__":
    main()

=======

    # ===== AGGREGATE & RANK =====
    aggregate_and_rank(df)

    print("\n" + "="*60)
    print("CLUSTERING EVALUATION COMPLETE!")
    print("="*60)
    print(f"\nOutputs saved to:")
    print(f"  - Tables: {OUT_DIR_TABLES}/")
    print(f"  - Figures: {OUT_DIR_FIGS}/")
    print("\nKey files:")
    print("  - clustering_eval.csv       : Per-run results")
    print("  - clustering_eval_agg.csv   : Aggregated (mean ± std)")
    print("  - clustering_eval_top.csv   : Best configurations")
    print("  - *_tsne_true.png           : t-SNE by true labels")
    print("  - *_tsne_clusters.png       : t-SNE by cluster assignments")
    print("  - *_umap_*.png              : UMAP versions")

if __name__ == "__main__":
    main()


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
BASIC USAGE:
------------
# Quick test with default parameters
python final_task2.py

# Custom datasets and methods
python final_task2.py \
  --datasets MUTAG ENZYMES \
  --methods graph2vec netlsd

# Test multiple dimensions
python final_task2.py \
  --dims 32 64 128 256

# More seeds for statistical confidence
python final_task2.py \
  --seeds 0 1 2 3 4

# Generate plots for all seeds (slower)
python final_task2.py \
  --plot_policy all

# Skip plots entirely (faster)
python final_task2.py \
  --plot_policy none

# Use GPU for GIN training
python final_task2.py \
  --device cuda

# Custom GIN architecture
python final_task2.py \
  --gin_hidden 128 \
  --gin_layers 5 \
  --gin_epochs 50


COMPLETE EVALUATION:
-------------------
python final_task2.py \
  --datasets MUTAG ENZYMES IMDB-MULTI \
  --methods graph2vec netlsd gin \
  --dims 32 64 128 \
  --seeds 0 1 2 3 4 \
  --plot_policy first_seed \
  --gin_hidden 64 \
  --gin_layers 3 \
  --gin_epochs 30 \
  --device cpu

"""
>>>>>>> Stashed changes
