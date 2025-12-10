#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task (b): Clustering of graph embeddings.

Embeddings: Graph2Vec, NetLSD, GIN
Datasets: MUTAG, ENZYMES, IMDB-MULTI (configurable via CLI)
Clustering: KMeans, SpectralClustering
Metrics: ARI (main), Silhouette (secondary)
Visuals:
  - t-SNE (colored by true class and by k-means clusters)
  - UMAP (colored by true class and by k-means clusters)

Outputs:
  report/tables/clustering_eval.csv
  report/tables/clustering_eval_agg.csv
  report/tables/clustering_eval_top.csv
  report/figures/*_{tsne,umap}_true.png
  report/figures/*_{tsne,umap}_clusters.png
"""

# ---------------- Headless plotting ----------------
import matplotlib
matplotlib.use("Agg")

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

import os, time, argparse, json, warnings, inspect
warnings.filterwarnings("ignore")  # Suppress library warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from contextlib import contextmanager

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
# ENVIRONMENT PATCHES (compatibility fixes)
# ------------------------------------------------------------------

# 1. Some SciPy builds don't expose scipy.errstate but NetworkX calls it.
#    We'll create scipy.errstate = numpy.errstate BEFORE importing networkx.
import numpy as _np
import scipy as _sp
if not hasattr(_sp, "errstate"):
    _sp.errstate = _np.errstate  # patch for older / weird SciPy builds

# 2. Some sklearn versions don't support ensure_all_finite in check_array,
#    but newer umap-learn will call it with that kwarg. We'll monkeypatch
#    sklearn.utils.validation.check_array *and* sklearn.utils.check_array
#    BEFORE importing umap.
import sklearn.utils.validation as _suv
import sklearn.utils as _su

if "ensure_all_finite" not in inspect.signature(_suv.check_array).parameters:
    _orig_check_array = _suv.check_array

    def _wrapped_check_array(*args, ensure_all_finite=None, **kwargs):
        # Ignore ensure_all_finite if old sklearn doesn't know it.
        return _orig_check_array(*args, **kwargs)

    _suv.check_array = _wrapped_check_array
    _su.check_array  = _wrapped_check_array

# Now it's safe to import umap
try:
    from umap import UMAP
except Exception:
    from umap.umap_ import UMAP

# Import graph libraries after patches
import networkx as nx
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool

from karateclub import Graph2Vec  # Unsupervised graph embedding

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================

OUT_DIR_TABLES = "report/tables"
OUT_DIR_FIGS   = "report/figures"
OUT_DIR_LOGS   = "report/logs"
os.makedirs(OUT_DIR_TABLES, exist_ok=True)
os.makedirs(OUT_DIR_FIGS,   exist_ok=True)
os.makedirs(OUT_DIR_LOGS,   exist_ok=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ds_labels(ds):
    """Extract integer labels from PyTorch Geometric dataset."""
    return np.array([int(g.y) for g in ds])

def attach_degree_as_feature(graph):
    """
    Add node degree as a feature if graph has no node attributes.
    
    Some datasets (e.g., IMDB-MULTI) have no node features. GIN requires
    node features, so we create a single feature: node degree.
    
    Args:
        graph: PyG Data object
        
    Returns:
        graph: Same graph with x attribute added (if it was missing)
    """
    if getattr(graph, "x", None) is None:
        # Convert to NetworkX to easily compute degrees
        G = to_networkx(graph, to_undirected=True)
        deg = np.array([d for _, d in G.degree()], dtype=np.float32)
        # Add as single-column feature matrix
        graph.x = torch.from_numpy(deg).view(-1, 1)
    return graph

def to_nx_with_labels(ds_slice):
    """
    Convert PyG graphs to NetworkX format with discrete node labels.
    
    Graph2Vec requires each node to have a discrete 'label' attribute.
    We use node degree as the label (standard practice).
    
    Args:
        ds_slice: List of PyG graph objects
        
    Returns:
        List of NetworkX graphs with 'label' attribute on each node
    """
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
    Gs = to_nx_with_labels(ds_slice)
    
    with timed("Graph2Vec"):
        # Initialize and train Graph2Vec model
        model = Graph2Vec(
            dimensions=dim,
            wl_iterations=wl_iterations,  # Substructure depth
            epochs=epochs,
            seed=seed,
            workers=1,  # Single-threaded for reproducibility
            min_count=min_count,
        )
        model.fit(Gs)
        X = model.get_embedding()
    return X

def _netlsd_signature_dense(G, times):
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
    n = G.number_of_nodes()
    if n == 0:
        return np.zeros_like(times)
    
    # Compute normalized Laplacian (symmetric matrix)
    L = nx.normalized_laplacian_matrix(G).astype(float).toarray()
    
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
    times = np.logspace(np.log10(t_min), np.log10(t_max), num=n_times)
    
    # Convert to NetworkX
    Gs = [to_networkx(g, to_undirected=True) for g in ds_slice]
    
    with timed("NetLSD"):
        # Compute heat trace signature for each graph
        sigs = [_netlsd_signature_dense(G, times) for G in Gs]
        X = np.vstack(sigs)  # Stack into matrix [num_graphs, n_times]
        
        # Apply PCA if target dimension differs from raw signature size
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
                nn.Linear(h, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.layers.append(GINConv(mlp))  # GIN convolution with this MLP
            h = hidden
        
        # Classification head (used during training, ignored for embeddings)
        self.head = nn.Linear(hidden, n_classes)

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
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph-level pooling (aggregate node features → graph feature)
        g = global_mean_pool(x, batch)  # [num_graphs, hidden] ← EMBEDDING!
        
        # Classification head (for training supervision)
        logits = self.head(g)
        
        return logits, g

def train_gin_get_embeddings(
    ds_slice,
    dim=128,
    seed=0,
    hidden=64,
    layers=3,
    dropout=0.2,
    epochs=30,
    batch_size=64,
    lr=1e-3,
    device="cpu",
):
    """
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
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Ensure all graphs have node features (use degree if missing)
    graphs = [attach_degree_as_feature(g.clone()) for g in ds_slice]
    
    # Determine number of classes and input dimension
    num_classes = len(np.unique([int(g.y) for g in graphs]))
    in_dim = graphs[0].x.size(-1)

    # Initialize GIN model
    model = GINEncoder(
        in_dim,
        hidden=hidden,
        layers=layers,
        dropout=dropout,
        n_classes=num_classes,
    ).to(device)

    # Optimizer and data loader
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

    # ===== EMBEDDING EXTRACTION PHASE =====
    with timed("GIN-embed"):
        model.eval()
        chunks = []
        eval_loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for batch in eval_loader:
                batch = batch.to(device)
                _, g = model(batch)
                chunks.append(g.cpu())
        X = torch.cat(chunks, dim=0).numpy()  # [num_graphs, hidden]

    # Apply PCA if target dimension differs from hidden dimension
    if dim != X.shape[1]:
        X = PCA(n_components=dim, random_state=seed).fit_transform(X)

    return X


def cluster_and_score(X, y, n_clusters, seed, algo="kmeans"):
    """
    Apply clustering and compute evaluation metrics.
    
    
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

    if algo == "kmeans":
        model = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
        pred = model.fit_predict(X_proc)
    elif algo == "spectral":
        # Spectral Clustering: Build similarity graph, then cluster eigenvectors
        # Can find non-convex clusters (more flexible than K-Means)
        model = SpectralClustering(
            n_clusters=n_clusters,
            assign_labels="kmeans",  # Final assignment via k-means on eigenvectors
            affinity="rbf",  # Gaussian (RBF) similarity
            random_state=seed,
        )
        pred = model.fit_predict(X_proc)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    ari = adjusted_rand_score(y, pred)

    sil = np.nan
    try:
        # Need at least 2 clusters for silhouette
        if len(np.unique(pred)) > 1:
            sil = silhouette_score(X_proc, pred)
    except Exception:
        sil = np.nan

    return {
        "ari": float(ari),
        "silhouette": float(np.nan if np.isnan(sil) else sil),
        "labels": pred,
        "X_proc": X_proc,
    }

# ============================================================================
# VISUALIZATION: t-SNE & UMAP
# ============================================================================

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
    ax = fig.add_subplot(111)
    ax.scatter(X2[:, 0], X2[:, 1], c=labels, s=16)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_tsne_umap(X_proc, y_true, y_clusters, title_prefix, out_prefix,
                   tsne_seed=0, umap_seed=0):
    """
    Saves four plots:
      {out_prefix}_tsne_true.png
      {out_prefix}_tsne_clusters.png
      {out_prefix}_umap_true.png
      {out_prefix}_umap_clusters.png
    If UMAP fails, you'll still get the t-SNE plots.
    """

    # --- t-SNE ---
    try:
        T = TSNE(
            n_components=2,
            random_state=tsne_seed,
            init="pca",
            perplexity=min(30, max(5, len(X_proc)//10)),
        )
        Xt = T.fit_transform(X_proc)

        scatter_2d(
            Xt,
            y_true,
            f"{title_prefix} — t-SNE (true)",
            f"{out_prefix}_tsne_true.png",
        )
        scatter_2d(
            Xt,
            y_clusters,
            f"{title_prefix} — t-SNE (clusters)",
            f"{out_prefix}_tsne_clusters.png",
        )
    except Exception as e:
        print(f"[warn] t-SNE plot failed: {e}")

    # --- UMAP ---
    try:
        n_neighbors = min(15, max(2, len(X_proc) - 1))
        U = UMAP(
            n_components=2,
            random_state=umap_seed,
            n_neighbors=n_neighbors,
            min_dist=0.1,
        )
        Xu = U.fit_transform(X_proc)

        scatter_2d(
            Xu,
            y_true,
            f"{title_prefix} — UMAP (true)",
            f"{out_prefix}_umap_true.png",
        )
        scatter_2d(
            Xu,
            y_clusters,
            f"{title_prefix} — UMAP (clusters)",
            f"{out_prefix}_umap_clusters.png",
        )
    except Exception as e:
        print(f"[warn] UMAP plot failed: {e}")


# ---------------- Runner ----------------
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
    """
    plot_policy: 'none' | 'first_seed' | 'all'
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

                    # ===== STEP 1: GENERATE EMBEDDINGS =====
                    if method.lower() == "graph2vec":
                        X = embed_graph2vec(
                            graphs,
                            dim=dim,
                            seed=seed,
                            epochs=20,
                        )
                    elif method.lower() == "netlsd":
                        X = embed_netlsd(
                            graphs,
                            dim=dim,
                            pca_seed=seed,
                        )
                    elif method.lower() == "gin":
                        X = train_gin_get_embeddings(
                            graphs,
                            dim=dim,
                            seed=seed,
                            hidden=gin_hidden,
                            layers=gin_layers,
                            dropout=gin_dropout,
                            epochs=gin_epochs,
                            batch_size=gin_batch,
                            device=device,
                        )
                    else:
                        raise ValueError(f"Unknown method: {method}")

                    # ===== STEP 2: CLUSTERING & METRICS =====
                    
                    # K-Means clustering
                    res_km = cluster_and_score(X, y, n_clusters, seed, algo="kmeans")
                    
                    # Spectral clustering
                    res_sp = cluster_and_score(X, y, n_clusters, seed, algo="spectral")

                    # Prepare result rows
                    row_km = dict(
                        dataset=ds_name,
                        method=method,
                        dim=dim,
                        seed=seed,
                        algo="kmeans",
                        ari=res_km["ari"],
                        silhouette=(
                            None
                            if np.isnan(res_km["silhouette"])
                            else float(res_km["silhouette"])
                        ),
                        n_graphs=len(ds),
                        n_clusters=n_clusters,
                    )
                    
                    row_sp = dict(
                        dataset=ds_name,
                        method=method,
                        dim=dim,
                        seed=seed,
                        algo="spectral",
                        ari=res_sp["ari"],
                        silhouette=(
                            None
                            if np.isnan(res_sp["silhouette"])
                            else float(res_sp["silhouette"])
                        ),
                        n_graphs=len(ds),
                        n_clusters=n_clusters,
                    )

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
                    do_plots = (
                        plot_policy == "all"
                        or (plot_policy == "first_seed" and seed == first_seed)
                    )
                    
                    if do_plots:
                        prefix = f"{OUT_DIR_FIGS}/{ds_name}_{method}_d{dim}"
                        title  = f"{ds_name} | {method} (d={dim})"
                        plot_tsne_umap(
                            X_proc=res_km["X_proc"],
                            y_true=y,
                            y_clusters=res_km["labels"],
                            title_prefix=title,
                            out_prefix=prefix,
                            tsne_seed=seed,
                            umap_seed=seed,
                        )

                    # ===== STEP 4: PROGRESSIVE SAVE =====
                    # Save after each configuration so we don't lose results
                    # if the script crashes or is interrupted
                    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # Final DataFrame
    df = pd.DataFrame(rows)
    print(f"\nSaved per-run clustering results to {out_csv}")
    return df

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
    if df.empty:
        print("No rows to aggregate.")
        return df, None

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


    # Save aggregated results
    agg.to_csv(out_csv_agg, index=False)
    print(f"Saved aggregated results to {out_csv_agg}")

    # ===== RANKING: FIND BEST CONFIGURATION PER DATASET =====
    # Best = highest ARI (primary metric)
    # If tie, use Silhouette as tiebreaker
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




TROUBLESHOOTING:
---------------
1. "Out of memory" with GIN:
   → Reduce --gin_batch (e.g., to 32)
   → Reduce --gin_hidden (e.g., to 32)
   → Use fewer --gin_layers

2. Very low ARI across all methods:
   → Dataset may have inherently overlapping classes
   → Try increasing embedding dimension
   → Check if dataset has clear class structure

3. Plots failing:
   → t-SNE/UMAP can fail on very small datasets
   → Reduce perplexity (edit code if needed)
   → Use --plot_policy none to skip

4. Slow execution:
   → Reduce number of seeds
   → Use smaller dimensions
   → Skip plots with --plot_policy none
   → For GIN, reduce --gin_epochs
"""
