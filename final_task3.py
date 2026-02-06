#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task (c): Stability Analysis for Graph Embedding Methods

PURPOSE:
- Evaluate how robust graph embeddings are to perturbations
- Test stability under two types of noise:
  1. Edge perturbations (add/remove edges)
  2. Attribute perturbations (shuffle node features)
- Compare stability across different embedding methods and dimensions

KEY QUESTIONS:
- "How much do embeddings change when the graph is perturbed?"
- "Which embedding methods are more stable/robust?"
- "How does perturbation affect downstream classification performance?"

WHY THIS MATTERS:
- Real-world graphs are noisy (missing edges, errors in features)
- Stable embeddings → More reliable predictions
- Understanding robustness helps choose appropriate methods

METRICS:
1. Embedding Stability:
   - Cosine similarity (higher = more stable)
   - L2 drift (lower = more stable)
2. Classification Degradation:
   - ΔAccuracy, ΔF1, ΔAUC (how much performance drops)

WORKFLOW:
1. Generate clean embeddings for each graph
2. Apply perturbations at different levels
3. Generate perturbed embeddings
4. Measure:
   a. Embedding drift (cosine sim, L2 distance)
   b. Classification performance change
5. Plot stability curves across perturbation levels

CLEAN PLOTTING VERSION:
- Small, focused plots per (dataset × perturbation × method × metric)
- One line per dimension for easy comparison
- Separate plots for each metric to avoid clutter
"""

# ============================================================================
# ENVIRONMENT SETUP & COMPATIBILITY PATCHES
# ============================================================================

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for headless plotting

# PATCH 1: Fix missing scipy.errstate in some environments
# Some SciPy builds don't expose scipy.errstate, but NetworkX needs it
import numpy as _np
import scipy as _sp
if not hasattr(_sp, "errstate"):
    _sp.errstate = _np.errstate  # Borrow from NumPy

# PATCH 2: Fix UMAP compatibility with older scikit-learn versions
# Newer UMAP calls check_array with 'ensure_all_finite' kwarg that old sklearn doesn't support
try:
    import inspect
    from sklearn.utils import validation as _suv
    import umap.umap_ as _umap_mod
    if "ensure_all_finite" not in inspect.signature(_suv.check_array).parameters:
        _orig = _suv.check_array
        def _check_array_wrapper(*args, ensure_all_finite=None, **kwargs):
            return _orig(*args, **kwargs)  # Ignore the new parameter
        _umap_mod.check_array = _check_array_wrapper
except Exception:
    pass

# ============================================================================
# IMPORTS
# ============================================================================

import os, argparse, json, warnings, time, random
warnings.filterwarnings("ignore")  # Suppress scientific library warnings

import numpy as np
import pandas as pd
import psutil  # For memory profiling
import matplotlib.pyplot as plt

from contextlib import contextmanager
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_networkx, add_self_loops
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GINConv, global_mean_pool

import networkx as nx
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from karateclub import Graph2Vec  # Graph2Vec embedding library

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================

OUT_DIR_TABLES = "final_report3/tables"
OUT_DIR_FIGS   = "final_report3/figures"
os.makedirs(OUT_DIR_TABLES, exist_ok=True)
os.makedirs(OUT_DIR_FIGS,   exist_ok=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed: int):
    """
    Set all random seeds for reproducibility.
    
    Ensures consistent results across runs by fixing:
    - Python's random module
    - NumPy's random generator
    - PyTorch's random operations
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def ds_labels(ds):
    """
    Extract integer labels from PyTorch Geometric dataset.
    
    Args:
        ds: PyG dataset
        
    Returns:
        labels: NumPy array of integer labels
    """
    return np.array([int(g.y) for g in ds])

def ensure_node_features(graphs: List[Data]) -> List[Data]:
    """
    Ensure all graphs have node features.
    
    If node features (x) are missing, use node degree as a simple 1D feature.
    This is essential for GIN which requires node attributes.
    
    Args:
        graphs: List of PyG Data objects
        
    Returns:
        graphs: Same list with x attribute added where missing
    """
    out = []
    for g in graphs:
        if getattr(g, "x", None) is None:
            # Compute node degrees
            deg = torch.bincount(g.edge_index[0], minlength=g.num_nodes).float().view(-1, 1)
            # Create new graph with degree as features
            g = Data(x=deg, edge_index=g.edge_index, y=g.y, num_nodes=g.num_nodes)
        out.append(g)
    return out

@contextmanager
def timed(name="block"):
    """
    Context manager to measure execution time.
    
    Usage:
        with timed("my_operation"):
            # your code here
        # Prints: [my_operation] X.XXs
    
    Args:
        name: Label for the timed block
    """
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"[{name}] {dt:.2f}s")

# ============================================================================
# PERTURBATION FUNCTIONS
# ============================================================================

def perturb_edges(g: Data, level: float, seed: int) -> Data:
    """
    Perturb graph by adding and removing edges.
    
    ALGORITHM:
    1. Convert edge list to undirected edge set
    2. Remove ~level fraction of existing edges
    3. Add ~level fraction of random new edges
    4. Convert back to PyG format
    
    This simulates:
    - Missing edges in real data
    - Spurious edges due to noise
    - Uncertainty in graph structure
    
    Args:
        g: Original PyG Data object
        level: Perturbation intensity (0.0 to 1.0)
               0.5 means ~50% of edges are affected
        seed: Random seed for reproducibility
        
    Returns:
        g_perturbed: New graph with perturbed edges
    """
    set_seed(seed)
    n = g.num_nodes
    E = g.edge_index.t().tolist()
    
    # Create undirected edge set (remove duplicates, self-loops)
    und = set(tuple(sorted(e)) for e in E if e[0] != e[1])
    m = len(und)
    if m == 0:
        return g.clone()

    # Number of edges to modify
    k = max(1, int(level * m))

    # STEP 1: Remove k random edges
    rem = random.sample(list(und), min(k, m))
    for e in rem:
        if e in und:
            und.remove(e)

    # STEP 2: Add k random new edges
    possible = set()
    attempts = 0
    while len(possible) < k and attempts < 20 * k:
        u = random.randrange(n)
        v = random.randrange(n)
        if u == v:  # Skip self-loops
            attempts += 1
            continue
        e = tuple(sorted((u, v)))
        if e not in und:  # Only add new edges
            possible.add(e)
        attempts += 1
    und.update(possible)

    # STEP 3: Build new edge_index (both directions for undirected)
    if len(und) == 0:
        ei = torch.zeros((2, 0), dtype=torch.long)
    else:
        u, v = zip(*und)
        # Add both (u,v) and (v,u) for undirected graph
        ei = torch.tensor([list(u) + list(v), list(v) + list(u)], dtype=torch.long)

    return Data(x=g.x.clone(), edge_index=ei, y=g.y, num_nodes=n)

def perturb_attrs(g: Data, level: float, seed: int) -> Data:
    """
    Perturb graph by shuffling node features.
    
    ALGORITHM:
    1. Select ~level fraction of nodes
    2. Randomly permute their feature vectors
    3. Leave other nodes unchanged
    
    This simulates:
    - Measurement noise in node attributes
    - Errors in feature extraction
    - Missing or corrupted features
    
    Args:
        g: Original PyG Data object
        level: Perturbation intensity (0.0 to 1.0)
               1.0 means all node features are shuffled
        seed: Random seed for reproducibility
        
    Returns:
        g_perturbed: New graph with perturbed node features
    """
    set_seed(seed)
    x = g.x.clone()
    n = x.size(0)
    
    # Number of nodes to perturb
    k = max(1, int(level * n))
    
    # Select random nodes to perturb
    idx = np.arange(n)
    np.random.shuffle(idx)
    take = idx[:k]
    
    # Permute features among selected nodes
    perm = take.copy()
    np.random.shuffle(perm)
    x[take] = x[perm]
    
    return Data(x=x, edge_index=g.edge_index.clone(), y=g.y, num_nodes=g.num_nodes)

# ============================================================================
# EMBEDDING METHODS
# ============================================================================

def to_nx_with_labels(ds_slice):
    """
    Convert PyG graphs to NetworkX format and add node labels.
    
    Graph2Vec requires discrete node labels. Here we use node degree.
    
    Args:
        ds_slice: List of PyG graph objects
        
    Returns:
        List of NetworkX graphs with 'label' attribute on each node
    """
    Gs = []
    for g in ds_slice:
        G = to_networkx(g, to_undirected=True)
        degs = dict(G.degree())
        for n in G.nodes:
            G.nodes[n]["label"] = int(degs[n])  # Assign degree as label
        Gs.append(G)
    return Gs

def embed_graph2vec(graphs: List[Data], dim: int, seed: int):
    """
    Generate Graph2Vec embeddings.
    
    Graph2Vec learns graph-level embeddings by treating graphs as "documents"
    and their substructures (from Weisfeiler-Lehman) as "words", then applies
    doc2vec-style training.
    
    Process:
    1. Extract WL substructures from each graph
    2. Treat substructures as vocabulary
    3. Learn embeddings via skip-gram
    
    Args:
        graphs: List of PyG graphs
        dim: Embedding dimension
        seed: Random seed
        
    Returns:
        X: Embedding matrix [num_graphs, dim]
    """
    Gs = to_nx_with_labels(graphs)
    with timed("graph2vec"):
        model = Graph2Vec(
            dimensions=dim,
            wl_iterations=2,  # Depth of WL kernel
            epochs=20,        # Training epochs
            seed=seed,
            workers=1,
            min_count=5       # Minimum frequency for substructures
        )
        model.fit(Gs)
        X = model.get_embedding()
    return X

def _netlsd_signature_dense(G, times):
    """
    Compute NetLSD heat trace signature using dense eigendecomposition.
    
    NetLSD (Network Laplacian Spectral Descriptor) characterizes a graph
    through the heat trace of its normalized Laplacian at different time scales.
    
    Heat trace at time t: h(t) = Σᵢ exp(-t · λᵢ)
    where λᵢ are the eigenvalues of the normalized Laplacian.
    
    The heat trace captures:
    - Small t: Local structure (triangles, small motifs)
    - Large t: Global structure (communities, diameter)
    
    Args:
        G: NetworkX graph
        times: Array of diffusion times
        
    Returns:
        Array of heat trace values (one per time)
    """
    n = G.number_of_nodes()
    if n == 0:
        return np.zeros_like(times)

    # Compute normalized Laplacian (symmetric)
    L = nx.normalized_laplacian_matrix(G).astype(float).toarray()
    
    # Eigenvalues (always real for symmetric L)
    lam = np.linalg.eigvalsh(L)

    # Heat trace: h(t) = sum_i exp(-t * lambda_i)
    return np.exp(-np.outer(times, lam)).sum(axis=1)

def embed_netlsd(graphs: List[Data], dim: int, seed: int):
    """
    Generate NetLSD embeddings with PCA compression.
    
    Process:
    1. Compute heat trace signatures at logarithmically-spaced times
    2. Stack into matrix [num_graphs, n_times]
    3. Apply PCA to reduce to desired dimension
    
    Args:
        graphs: List of PyG graphs
        dim: Target embedding dimension
        seed: Random seed for PCA
        
    Returns:
        X: Embedding matrix [num_graphs, dim]
    """
    # Logarithmically-spaced time points (cover multiple scales)
    times = np.logspace(-2, 2, 256)
    Gs = [to_networkx(g, to_undirected=True) for g in graphs]
    
    with timed("netlsd"):
        # Compute heat trace signatures for all graphs
        sigs = [_netlsd_signature_dense(G, times) for G in Gs]
        X = np.vstack(sigs)  # [num_graphs, 256]
        
        # PCA compression to target dimension
        if dim != X.shape[1]:
            X = PCA(n_components=dim, random_state=seed).fit_transform(X)
    
    return X

# ============================================================================
# GIN MODEL (SUPERVISED EMBEDDING)
# ============================================================================

class GINSmall(nn.Module):
    """
    Lightweight GIN encoder for graph classification.
    
    GIN (Graph Isomorphism Network) is as powerful as the Weisfeiler-Lehman
    test for distinguishing graphs.
    
    Architecture:
    - Stack of GIN convolution layers (message passing)
    - Global mean pooling (graph → vector)
    - Classification head
    
    Used for:
    1. Supervised graph classification
    2. Extracting graph embeddings (penultimate layer)
    """
    
    def __init__(self, in_dim, hidden=64, layers=3, n_classes=2, dropout=0.2):
        """
        Initialize GIN model.
        
        Args:
            in_dim: Input feature dimension
            hidden: Hidden layer dimension (= embedding dimension)
            layers: Number of GIN layers
            n_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = dropout
        self.mlps = nn.ModuleList()
        self.convs = nn.ModuleList()

        h = hidden
        # First MLP/conv (input → hidden)
        self.mlps.append(nn.Sequential(nn.Linear(in_dim, h), nn.ReLU(), nn.Linear(h, h)))
        self.convs.append(GINConv(self.mlps[0]))
        
        # Remaining layers (hidden → hidden)
        for _ in range(layers - 1):
            mlp = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
            self.mlps.append(mlp)
            self.convs.append(GINConv(mlp))

        # Classification head
        self.lin = nn.Linear(h, n_classes)

    def forward(self, x, edge_index, batch):
        """
        Forward pass through GIN.
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge connections [2, num_edges]
            batch: Batch assignment [num_nodes]
            
        Returns:
            out: Classification logits [batch_size, n_classes]
            g: Graph embeddings [batch_size, hidden] (penultimate layer)
        """
        h = x
        # Message passing layers
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Aggregate to graph-level embedding
        g = global_mean_pool(h, batch)   # [batch_size, hidden]
        
        # Classification
        out = self.lin(g)
        return out, g

def train_gin_embed(graphs: List[Data], dim: int, seed: int,
                    epochs=30, batch_size=64, lr=1e-3, layers=3, dropout=0.2):
    """
    Train GIN encoder and extract graph embeddings.
    
    Training process:
    1. Train GIN for graph classification (supervised)
    2. Extract embeddings from penultimate layer
    3. Use these embeddings for downstream tasks
    
    Args:
        graphs: List of PyG graphs
        dim: Hidden dimension (= embedding dimension)
        seed: Random seed
        epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        layers: Number of GIN layers
        dropout: Dropout probability
        
    Returns:
        X: Embedding matrix [num_graphs, dim]
    """
    set_seed(seed)
    graphs = ensure_node_features(graphs)
    in_dim = graphs[0].x.size(1)
    n_classes = int(torch.stack([g.y for g in graphs]).max()) + 1

    # Create data loader
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = GINSmall(
        in_dim,
        hidden=dim,
        layers=layers,
        n_classes=n_classes,
        dropout=dropout
    )
    
    # Optimizer and loss
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    with timed(f"gin(h={dim})_train"):
        for _ in range(epochs):
            for batch in loader:
                opt.zero_grad()
                logits, _ = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(logits, batch.y)
                loss.backward()
                opt.step()

    # Extract embeddings (evaluation mode)
    model.eval()
    X = []
    with torch.no_grad():
        for g in DataLoader(graphs, batch_size=batch_size, shuffle=False):
            logits, emb = model(g.x, g.edge_index, g.batch)
            X.append(emb)
    X = torch.cat(X, dim=0).cpu().numpy()
    return X

def get_embeddings(method: str, graphs: List[Data], dim: int, seed: int) -> np.ndarray:
    """
    Unified interface for all embedding methods.
    
    Routes to appropriate embedding function based on method name.
    
    Args:
        method: Embedding method name ("graph2vec", "netlsd", or "gin")
        graphs: List of PyG graphs
        dim: Embedding dimension
        seed: Random seed
        
    Returns:
        X: Embedding matrix [num_graphs, dim]
        
    Raises:
        ValueError: If method name is unknown
    """
    method = method.lower()
    if method == "graph2vec":
        return embed_graph2vec(graphs, dim, seed)
    if method == "netlsd":
        return embed_netlsd(graphs, dim, seed)
    if method == "gin":
        return train_gin_embed(graphs, dim, seed, epochs=30, layers=3)
    raise ValueError(f"Unknown method: {method}")

# ============================================================================
# EVALUATION METRICS
# ============================================================================

def auc_any(y_true, scores, classes):
    """
    Compute ROC-AUC for binary or multiclass problems.
    
    Handles both:
    - Binary classification (2 classes)
    - Multiclass classification (>2 classes) using One-vs-Rest (OvR) macro average
    
    Args:
        y_true: True labels
        scores: Classifier scores (1D for binary, 2D for multiclass)
        classes: Unique class labels
        
    Returns:
        float: AUC score, or NaN if computation fails
    """
    y_true = np.asarray(y_true)
    classes = np.asarray(classes)
    try:
        if len(classes) == 2:
            # Binary classification
            if scores.ndim == 1:
                return roc_auc_score(y_true, scores)
            else:
                pos = 1 if scores.shape[1] > 1 else 0
                return roc_auc_score(y_true, scores[:, pos])
        else:
            # Multiclass: use One-vs-Rest (OvR) with macro averaging
            Y = label_binarize(y_true, classes=classes)
            return roc_auc_score(Y, scores, average="macro", multi_class="ovr")
    except Exception:
        return np.nan

def eval_clfs(X, y, seed):
    """
    Train classifiers and compute all metrics.
    
    Trains two classifiers:
    1. Linear SVM (fast, interpretable)
    2. MLP (nonlinear, more expressive)
    
    Computes metrics:
    - Accuracy: Overall correctness
    - F1 (macro): Average per-class F1 score
    - AUC: Area under ROC curve
    
    Args:
        X: Feature matrix [num_samples, dim]
        y: Labels
        seed: Random seed
        
    Returns:
        metrics: Dict with acc_svm, f1_svm, auc_svm, acc_mlp, f1_mlp, auc_mlp
    """
    classes = np.unique(y)
    
    # CLASSIFIER 1: Linear SVM with scaling
    svm = make_pipeline(
        StandardScaler(with_mean=True),
        LinearSVC(dual=False, random_state=seed)
    )
    svm.fit(X, y)
    yhat = svm.predict(X)
    s_score = svm.decision_function(X) if hasattr(svm, "decision_function") else None
    acc_svm = accuracy_score(y, yhat)
    f1_svm  = f1_score(y, yhat, average="macro")
    auc_svm = auc_any(y, s_score, classes)

    # CLASSIFIER 2: MLP with one hidden layer
    mlp = make_pipeline(
        StandardScaler(with_mean=True),
        MLPClassifier(
            hidden_layer_sizes=(128,),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=800,
            tol=1e-4,
            random_state=seed
        )
    )
    mlp.fit(X, y)
    yhat2 = mlp.predict(X)
    s_score2 = mlp.predict_proba(X) if hasattr(mlp, "predict_proba") else None
    acc_mlp = accuracy_score(y, yhat2)
    f1_mlp  = f1_score(y, yhat2, average="macro")
    auc_mlp = auc_any(y, s_score2, classes)

    return dict(
        acc_svm=acc_svm, f1_svm=f1_svm, auc_svm=auc_svm,
        acc_mlp=acc_mlp, f1_mlp=f1_mlp, auc_mlp=auc_mlp
    )

def emb_stability(X_clean, X_pert):
    """
    Measure embedding stability between clean and perturbed graphs.
    
    Two complementary metrics:
    1. Cosine similarity (higher = more stable)
       - Measures angular distance
       - Range: [-1, 1], typically [0, 1] for embeddings
       - Invariant to scaling
    
    2. L2 distance (lower = more stable)
       - Measures Euclidean distance
       - Sensitive to magnitude changes
       - Absolute difference
    
    Args:
        X_clean: Clean embeddings [num_graphs, dim]
        X_pert: Perturbed embeddings [num_graphs, dim]
        
    Returns:
        mean_cos: Mean cosine similarity (higher is better)
        mean_l2: Mean L2 distance (lower is better)
    """
    # Row-wise cosine similarity
    cs = np.diag(cosine_similarity(X_clean, X_pert))
    
    # Row-wise L2 distance
    l2 = np.linalg.norm(X_clean - X_pert, axis=1)
    
    return float(np.mean(cs)), float(np.mean(l2))

# ============================================================================
# PLOTTING HELPERS (CLEAN VERSION)
# ============================================================================

def _styled_fig_suptitle(fig, title):
    """
    Apply consistent styling to figure titles.
    
    Args:
        fig: Matplotlib figure
        title: Title text
    """
    fig.suptitle(title, fontsize=14, y=0.92, fontweight="bold")
    fig.subplots_adjust(top=0.8)

def _plot_with_shaded_std(ax, levels, means, stds, label):
    """
    Plot line with shaded standard deviation region.
    
    Creates:
    - Solid line for mean values
    - Shaded region for mean ± std
    
    Args:
        ax: Matplotlib axis
        levels: X-axis values (perturbation levels)
        means: Mean values at each level
        stds: Standard deviations at each level
        label: Legend label
        
    Returns:
        line: Line2D object for legend
    """
    line, = ax.plot(
        levels,
        means,
        marker="o",
        markersize=5,
        linewidth=2,
        alpha=0.9,
        label=label,
    )
    if len(levels) > 1:
        ax.fill_between(
            levels,
            np.array(means) - np.nan_to_num(stds),
            np.array(means) + np.nan_to_num(stds),
            alpha=0.15
        )
    return line

def _plot_metric_per_classifier_method(
    df,
    dataset,
    perturb_type,
    clf_key,          # "svm" or "mlp"
    metric_key,       # "acc", "f1", "auc"
    ylabel,
    outpath
):
    """
    Generate focused plot for one metric, one classifier, one method.
    
    PLOT STRUCTURE:
    - X-axis: Perturbation level
    - Y-axis: Δ metric (change from clean baseline)
    - Lines: One per embedding dimension
    - Shaded regions: Standard deviation across seeds
    
    This creates small, readable plots instead of cluttered multi-panel figures.
    
    Args:
        df: DataFrame filtered to one method
        dataset: Dataset name (for title)
        perturb_type: "edges" or "attrs"
        clf_key: "svm" or "mlp"
        metric_key: "acc", "f1", or "auc"
        ylabel: Y-axis label
        outpath: Where to save figure
    """
    method_name = df["method"].iloc[0]
    levels = sorted(df["level"].unique())

    fig, ax = plt.subplots(figsize=(5.5, 4))

    _styled_fig_suptitle(
        fig,
        f"{dataset} — Δ{metric_key.upper()} ({clf_key.upper()}) vs. perturb level ({perturb_type})\n{method_name}"
    )

    legend_handles = []
    legend_labels  = []

    # Plot one line per dimension
    for dim in sorted(df["dim"].unique()):
        sub = df[df["dim"] == dim]
        means = [sub[sub.level == lv][f"delta_{metric_key}_{clf_key}"].mean() for lv in levels]
        stds  = [sub[sub.level == lv][f"delta_{metric_key}_{clf_key}"].std()  for lv in levels]

        h = _plot_with_shaded_std(
            ax,
            levels,
            means,
            stds,
            label=f"d={dim}"
        )
        legend_handles.append(h)
        legend_labels.append(f"d={dim}")

    # Reference line at 0 (no change)
    ax.axhline(0, lw=1, ls="--", alpha=0.6)
    
    ax.set_xlabel("Perturbation level (relative)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, ls="--", alpha=0.3)

    ax.legend(
        legend_handles,
        legend_labels,
        loc="best",
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

def _plot_embed_stat_per_method(
    df,
    dataset,
    perturb_type,
    colname,          # "l2" or "cos"
    ylabel,
    title_metric_name,
    outpath
):
    """
    Generate plot for embedding stability metrics (classifier-agnostic).
    
    These metrics measure embedding drift, not classification performance.
    
    METRICS:
    - L2 drift: ||X_clean - X_pert||₂ (lower is better)
    - Cosine similarity: cos(X_clean, X_pert) (higher is better)
    
    Args:
        df: DataFrame filtered to one method
        dataset: Dataset name
        perturb_type: "edges" or "attrs"
        colname: Column name ("l2" or "cos")
        ylabel: Y-axis label
        title_metric_name: Metric name for title
        outpath: Where to save figure
    """
    method_name = df["method"].iloc[0]
    levels = sorted(df["level"].unique())

    fig, ax = plt.subplots(figsize=(5.5, 4))

    _styled_fig_suptitle(
        fig,
        f"{dataset} — {title_metric_name} vs. perturb level ({perturb_type})\n{method_name}"
    )

    legend_handles = []
    legend_labels  = []

    # Plot one line per dimension
    for dim in sorted(df["dim"].unique()):
        sub = df[df["dim"] == dim]
        means = [sub[sub.level == lv][colname].mean() for lv in levels]
        stds  = [sub[sub.level == lv][colname].std()  for lv in levels]

        h = _plot_with_shaded_std(
            ax,
            levels,
            means,
            stds,
            label=f"d={dim}"
        )
        legend_handles.append(h)
        legend_labels.append(f"d={dim}")

    ax.set_xlabel("Perturbation level (relative)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, ls="--", alpha=0.3)

    ax.legend(
        legend_handles,
        legend_labels,
        loc="best",
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def run_one_dataset(ds_name, methods, dims, seeds, levels_edges, levels_attrs):
    """
    Run stability analysis for one dataset.
    
    WORKFLOW:
    1. Load dataset
    2. For each (seed, method, dim):
       a. Generate clean embeddings
       b. Train classifiers on clean data
       c. For each perturbation level:
          - Apply perturbations
          - Generate perturbed embeddings
          - Measure embedding drift
          - Measure classification degradation
    3. Return results DataFrame
    
    Args:
        ds_name: Dataset name
        methods: List of embedding methods
        dims: List of embedding dimensions
        seeds: List of random seeds
        levels_edges: Edge perturbation levels
        levels_attrs: Attribute perturbation levels
        
    Returns:
        df: DataFrame with all results
    """
    print(f"\n=== Dataset: {ds_name} ===")
    
    # Load dataset
    ds = TUDataset(root="data", name=ds_name)
    graphs = ensure_node_features([ds[i] for i in range(len(ds))])
    y = ds_labels(ds)
    rows = []

    for seed in seeds:
        set_seed(seed)
        # Cache for clean embeddings and metrics
        # Key: (method, dim) → Value: (embeddings, metrics)
        cache_clean = {}

        for method in methods:
            for dim in dims:
                # STEP 1: Generate clean embeddings
                X_clean = get_embeddings(method, graphs, dim, seed)
                metrics_clean = eval_clfs(X_clean, y, seed)
                cache_clean[(method, dim)] = (X_clean, metrics_clean)

                # STEP 2: Edge perturbations
                for lv in levels_edges:
                    # Apply perturbations to all graphs
                    pert = [perturb_edges(g, lv, seed + 123) for g in graphs]
                    
                    # Generate perturbed embeddings
                    Xp = get_embeddings(method, pert, dim, seed)
                    
                    # Evaluate on perturbed embeddings
                    metrics_p = eval_clfs(Xp, y, seed)
                    
                    # Measure embedding stability
                    coss, l2 = emb_stability(X_clean, Xp)

                    # Compute metric changes (Δ = perturbed - clean)
                    row = dict(
                        dataset=ds_name,
                        perturb="edges",
                        level=float(lv),
                        method=method,
                        dim=int(dim),
                        seed=int(seed),
                        cos=float(coss),  # Cosine similarity (higher = more stable)
                        l2=float(l2)      # L2 drift (lower = more stable)
                    )
                    for key in ["acc", "f1", "auc"]:
                        row[f"delta_{key}_svm"] = float(metrics_p[f"{key}_svm"] - metrics_clean[f"{key}_svm"])
                        row[f"delta_{key}_mlp"] = float(metrics_p[f"{key}_mlp"] - metrics_clean[f"{key}_mlp"])
                    rows.append(row)

                # STEP 3: Attribute perturbations
                for lv in levels_attrs:
                    # Apply perturbations to all graphs
                    pert = [perturb_attrs(g, lv, seed + 456) for g in graphs]
                    
                    # Generate perturbed embeddings
                    Xp = get_embeddings(method, pert, dim, seed)
                    
                    # Evaluate on perturbed embeddings
                    metrics_p = eval_clfs(Xp, y, seed)
                    
                    # Measure embedding stability
                    coss, l2 = emb_stability(X_clean, Xp)

                    # Compute metric changes
                    row = dict(
                        dataset=ds_name,
                        perturb="attrs",
                        level=float(lv),
                        method=method,
                        dim=int(dim),
                        seed=int(seed),
                        cos=float(coss),
                        l2=float(l2)
                    )
                    for key in ["acc", "f1", "auc"]:
                        row[f"delta_{key}_svm"] = float(metrics_p[f"{key}_svm"] - metrics_clean[f"{key}_svm"])
                        row[f"delta_{key}_mlp"] = float(metrics_p[f"{key}_mlp"] - metrics_clean[f"{key}_mlp"])
                    rows.append(row)

    df = pd.DataFrame(rows)
    return df

# ============================================================================
# AGGREGATION & PLOTTING
# ============================================================================

def aggregate_and_plot(df_all):
    """
    Aggregate results and generate all plots.
    
    Creates separate plots for:
    - Each dataset
    - Each perturbation type (edges vs attributes)
    - Each embedding method
    - Each metric (accuracy, F1, AUC)
    - Each classifier (SVM, MLP)
    - Embedding stability metrics (L2, cosine)
    
    This results in focused, readable plots instead of cluttered multi-panel figures.
    
    Args:
        df_all: Combined DataFrame from all datasets
    """
    if df_all.empty:
        print("No rows to plot.")
        return

    # Save raw results
    out_csv = os.path.join(OUT_DIR_TABLES, "stability_results.csv")
    df_all.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")

    # Generate plots for each combination
    for ds in df_all["dataset"].unique():
        sub_ds = df_all[df_all["dataset"] == ds]
        
        for perturb in ["edges", "attrs"]:
            subp = sub_ds[sub_ds["perturb"] == perturb]

            # Loop over each embedding method
            for method_name in subp["method"].unique():
                subm = subp[subp["method"] == method_name]

                # --- CLASSIFICATION METRICS ---
                # Create separate plots for each (classifier, metric) combination
                for clf_key in ["svm", "mlp"]:
                    for metric_key, ylabel in [
                        ("acc", "ΔAccuracy"),
                        ("f1",  "ΔF1"),
                        ("auc", "ΔAUC"),
                    ]:
                        outname = f"{ds}_{perturb}_{method_name}_{clf_key}_delta_{metric_key}.png"
                        outpath = os.path.join(OUT_DIR_FIGS, outname)
                        _plot_metric_per_classifier_method(
                            df=subm,
                            dataset=ds,
                            perturb_type=perturb,
                            clf_key=clf_key,
                            metric_key=metric_key,
                            ylabel=ylabel,
                            outpath=outpath
                        )

                # --- EMBEDDING STABILITY METRICS ---
                # These don't depend on classifier
                
                # L2 drift plot
                outname_l2 = f"{ds}_{perturb}_{method_name}_embed_drift_l2.png"
                _plot_embed_stat_per_method(
                    df=subm,
                    dataset=ds,
                    perturb_type=perturb,
                    colname="l2",
                    ylabel="Mean L2 drift",
                    title_metric_name="Embedding drift (mean L2)",
                    outpath=os.path.join(OUT_DIR_FIGS, outname_l2)
                )

                # Cosine similarity plot
                outname_cos = f"{ds}_{perturb}_{method_name}_embed_cosine.png"
                _plot_embed_stat_per_method(
                    df=subm,
                    dataset=ds,
                    perturb_type=perturb,
                    colname="cos",
                    ylabel="Mean cosine similarity",
                    title_metric_name="Embedding stability (mean cosine)",
                    outpath=os.path.join(OUT_DIR_FIGS, outname_cos)
                )

# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Task (c): Stability Analysis with clean per-method plots"
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["MUTAG", "ENZYMES", "IMDB-MULTI"],
        help="Datasets to analyze"
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=["graph2vec", "netlsd", "gin"],
        help="Embedding methods to evaluate"
    )
    p.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=[32, 64],
        help="Embedding dimensions to test"
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1],
        help="Random seeds for reproducibility"
    )
    p.add_argument(
        "--levels_edges",
        nargs="+",
        type=float,
        default=[0.5, 1.0],
        help="Relative fraction of edges to add/remove (0.5 = 50%%)"
    )
    p.add_argument(
        "--levels_attrs",
        nargs="+",
        type=float,
        default=[0.5, 1.0],
        help="Relative fraction of node features to shuffle (1.0 = 100%%)"
    )

    return p.parse_args()

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Main execution function.
    
    Workflow:
    1. Parse arguments
    2. Run stability analysis for each dataset
    3. Combine results
    4. Generate all plots
    """
    os.environ.setdefault("PYTHONNOUSERSITE", "1")  # Avoid user site-packages
    args = parse_args()

    all_rows = []
    for ds in args.datasets:
        df = run_one_dataset(
            ds_name=ds,
            methods=args.methods,
            dims=args.dims,
            seeds=args.seeds,
            levels_edges=args.levels_edges,
            levels_attrs=args.levels_attrs,
        )
        all_rows.append(df)

    # Combine all results
    df_all = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    
    # Generate plots
    aggregate_and_plot(df_all)

if __name__ == "__main__":
    main()