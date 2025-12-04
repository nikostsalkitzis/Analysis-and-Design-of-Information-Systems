#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task (c): Stability Analysis

PURPOSE:
- Test how ROBUST graph embeddings are to perturbations (noise)
- Compare embedding drift when graphs are modified
- Measure impact on downstream classification performance
- Identify which methods are most stable/reliable

KEY QUESTIONS:
1. If I slightly modify a graph, does the embedding change dramatically?
2. Does this change hurt classification performance?
3. Are some methods more robust than others?
4. Does embedding dimension affect stability?

WHY THIS MATTERS:
Real-world graphs have noise:
- Missing/false edges in social networks
- Measurement errors in molecular structures
- Noisy node features in biological networks
Stable embeddings → More reliable for production systems

WORKFLOW:
1. Generate embeddings for CLEAN graphs (baseline)
2. PERTURB graphs (add/remove edges OR shuffle node features)
3. Generate embeddings for PERTURBED graphs
4. Compare embeddings (L2 distance, Cosine similarity)
5. Compare classification performance (Δ Accuracy, Δ F1, Δ AUC)
6. Plot stability curves over perturbation levels
"""

# ============================================================================
# ENVIRONMENT SETUP & COMPATIBILITY PATCHES
# ============================================================================

import matplotlib
matplotlib.use("Agg")  # Headless plotting (no display needed)

# PATCH 1: Fix missing scipy.errstate
import numpy as _np
import scipy as _sp
if not hasattr(_sp, "errstate"):
    _sp.errstate = _np.errstate

# PATCH 2: Fix UMAP compatibility with older scikit-learn
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

# ============================================================================
# IMPORTS
# ============================================================================

import os, argparse, json, warnings, time, random
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import psutil
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

from karateclub import Graph2Vec

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================

# --------- Paths ---------
OUT_DIR_TABLES = "report/tables"
OUT_DIR_FIGS   = "report/figures"
os.makedirs(OUT_DIR_TABLES, exist_ok=True)
os.makedirs(OUT_DIR_FIGS,   exist_ok=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def ds_labels(ds):
    """Extract integer labels from PyTorch Geometric dataset."""
    return np.array([int(g.y) for g in ds])

def ensure_node_features(graphs: List[Data]) -> List[Data]:
    """
    Ensure all graphs have node features.
    
    Some datasets (e.g., IMDB-MULTI) don't have node attributes.
    We create a simple feature: node degree.
    
    Args:
        graphs: List of PyG Data objects
        
    Returns:
        graphs: Same list, but with x attribute added if missing
    """
    out = []
    for g in graphs:
        if getattr(g, "x", None) is None:
            # Use node degree as a 1D feature
            deg = torch.bincount(
                g.edge_index[0],
                minlength=g.num_nodes
            ).float().view(-1, 1)
            g = Data(x=deg, edge_index=g.edge_index, y=g.y, num_nodes=g.num_nodes)
        out.append(g)
    return out

@contextmanager
def timed(name="block"):
    """Simple timer context manager."""
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"[{name}] {dt:.2f}s")

# ============================================================================
# GRAPH PERTURBATION FUNCTIONS
# ============================================================================

def perturb_edges(g: Data, level: float, seed: int) -> Data:
    """
    Perturb graph by adding and removing edges.
    
    Process:
    1. Remove ~level fraction of existing edges (random)
    2. Add ~level fraction of new edges (random, avoiding existing)
    
    Example (level=0.5):
        Original: 100 edges
        Remove:   ~50 random edges
        Add:      ~50 new random edges
        Result:   Different graph with similar edge count
    
    WHY THIS MATTERS:
    - Simulates errors in graph construction
    - Tests if embedding relies on exact edge set
    - Real-world: Link prediction errors, sampling noise
    
    Args:
        g: PyG graph (Data object)
        level: Fraction of edges to modify (0.0 = none, 1.0 = all)
        seed: Random seed
        
    Returns:
        Perturbed graph (new Data object)
    """
    set_seed(seed)
    n = g.num_nodes
    
    # Convert edges to undirected set
    E = g.edge_index.t().tolist()
    und = set(tuple(sorted(e)) for e in E if e[0] != e[1])  # Remove self-loops
    m = len(und)
    
    if m == 0:
        return g.clone()  # Empty graph, nothing to perturb

    # Number of edges to modify
    k = max(1, int(level * m))

    # ===== STEP 1: REMOVE EDGES =====
    # Randomly select k edges to remove
    rem = random.sample(list(und), min(k, m))
    for e in rem:
        if e in und:
            und.remove(e)

    # ===== STEP 2: ADD EDGES =====
    # Generate k new edges (not already present)
    possible = set()
    attempts = 0
    while len(possible) < k and attempts < 20 * k:
        u = random.randrange(n); v = random.randrange(n)
        if u == v: 
            attempts += 1; continue
        e = tuple(sorted((u, v)))
        
        # Only add if not already in graph
        if e not in und:
            possible.add(e)
        
        attempts += 1
    
    # Add the new edges
    und.update(possible)

    # Build new edge_index (both directions)
    u, v = zip(*und) if und else ([], [])
    ei = torch.tensor([list(u)+list(v), list(v)+list(u)], dtype=torch.long)
    return Data(x=g.x.clone(), edge_index=ei, y=g.y, num_nodes=n)

def perturb_attrs(g: Data, level: float, seed: int) -> Data:
    """
    Perturb graph by shuffling node features.
    
    Process:
    1. Select ~level fraction of nodes randomly
    2. Shuffle their feature vectors among themselves
    
    Example (level=0.5):
        Original features:
            Node 0: [1.2, 0.5]
            Node 1: [0.3, 1.1]
            Node 2: [0.7, 0.9]
            Node 3: [1.5, 0.1]
        
        Select 50% (nodes 0, 2):
            Node 0 ← Node 2's features
            Node 2 ← Node 0's features
        
        Result:
            Node 0: [0.7, 0.9]  (changed)
            Node 1: [0.3, 1.1]  (unchanged)
            Node 2: [1.2, 0.5]  (changed)
            Node 3: [1.5, 0.1]  (unchanged)
    
    WHY THIS MATTERS:
    - Simulates measurement noise in node attributes
    - Tests if embedding relies heavily on exact feature values
    - Real-world: Sensor errors, incomplete data
    
    Args:
        g: PyG graph (Data object)
        level: Fraction of node features to shuffle (0.0 = none, 1.0 = all)
        seed: Random seed
        
    Returns:
        Perturbed graph (new Data object)
    """
    set_seed(seed)
    x = g.x.clone()
    n = x.size(0)
    
    # Number of nodes to perturb
    k = max(1, int(level * n))
    
    # Randomly select k nodes
    idx = np.arange(n)
    np.random.shuffle(idx)
    take = idx[:k]
    
    # Shuffle features among selected nodes
    perm = take.copy()
    np.random.shuffle(perm)
    x[take] = x[perm]  # Swap features
    
    # Create new graph with perturbed features
    return Data(
        x=x,
        edge_index=g.edge_index.clone(),
        y=g.y,
        num_nodes=g.num_nodes
    )

# ============================================================================
# EMBEDDING METHODS (same as previous tasks)
# ============================================================================

def to_nx_with_labels(ds_slice):
    """Convert PyG graphs to NetworkX with node labels (degree)."""
    Gs = []
    for g in ds_slice:
        G = to_networkx(g, to_undirected=True)
        degs = dict(G.degree())
        for n in G.nodes:
            G.nodes[n]["label"] = int(degs[n])
        Gs.append(G)
    return Gs

def embed_graph2vec(graphs: List[Data], dim: int, seed: int):
    """Generate Graph2Vec embeddings (unsupervised)."""
    Gs = to_nx_with_labels(graphs)
    with timed("graph2vec"):
        model = Graph2Vec(
            dimensions=dim,
            wl_iterations=2,
            epochs=20,
            seed=seed,
            workers=1,
            min_count=5
        )
        model.fit(Gs)
        X = model.get_embedding()
    return X

def _netlsd_signature_dense(G, times):
    """Compute NetLSD heat trace signature."""
    n = G.number_of_nodes()
    if n == 0:
        return np.zeros_like(times)
    
    L = nx.normalized_laplacian_matrix(G).astype(float).toarray()
    lam = np.linalg.eigvalsh(L)
    return np.exp(-np.outer(times, lam)).sum(axis=1)

def embed_netlsd(graphs: List[Data], dim: int, seed: int):
    """Generate NetLSD embeddings (unsupervised, spectral)."""
    times = np.logspace(-2, 2, 256)
    Gs = [to_networkx(g, to_undirected=True) for g in graphs]
    
    with timed("netlsd"):
        sigs = [_netlsd_signature_dense(G, times) for G in Gs]
        X = np.vstack(sigs)
        if dim != X.shape[1]:
            X = PCA(n_components=dim, random_state=seed).fit_transform(X)
    return X

# ===== GIN ENCODER (Supervised) =====

class GINSmall(nn.Module):
    """
    Lightweight GIN encoder for embeddings.
    
    Note: hidden dimension = target embedding dimension
    (No separate PCA step needed if they match)
    """
    def __init__(self, in_dim, hidden=64, layers=3, n_classes=2, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.mlps = nn.ModuleList()
        self.convs = nn.ModuleList()

        h = hidden
        # First layer
        self.mlps.append(nn.Sequential(
            nn.Linear(in_dim, h),
            nn.ReLU(),
            nn.Linear(h, h)
        ))
        self.convs.append(GINConv(self.mlps[0]))
        
        # Additional layers
        for _ in range(layers - 1):
            mlp = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
            self.mlps.append(mlp)
            self.convs.append(GINConv(mlp))

        # Classification head (for supervised training)
        self.lin = nn.Linear(h, n_classes)

    def forward(self, x, edge_index, batch):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        g = global_mean_pool(h, batch)  # Graph embedding
        out = self.lin(g)  # Classification logits
        return out, g

def train_gin_embed(graphs: List[Data], dim: int, seed: int,
                    epochs=30, batch_size=64, lr=1e-3, layers=3, dropout=0.2):
    """
    Train GIN with supervision, extract embeddings.
    
    Even though this is a clustering/stability task, we train GIN
    with labels to test if supervised embeddings are more robust.
    """
    set_seed(seed)
    graphs = ensure_node_features(graphs)
    in_dim = graphs[0].x.size(1)
    n_classes = int(torch.stack([g.y for g in graphs]).max()) + 1

    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
    model = GINSmall(in_dim, hidden=dim, layers=layers,
                     n_classes=n_classes, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Training
    model.train()
    with timed(f"gin(h={dim})_train"):
        for _ in range(epochs):
            for batch in loader:
                opt.zero_grad()
                logits, _ = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(logits, batch.y)
                loss.backward()
                opt.step()

    # Extract embeddings
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
    Unified interface for getting embeddings.
    
    Args:
        method: "graph2vec", "netlsd", or "gin"
        graphs: List of PyG graphs
        dim: Embedding dimension
        seed: Random seed
        
    Returns:
        X: Embedding matrix [num_graphs, dim]
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
    Compute AUC for binary or multiclass problems.
    Returns NaN if not computable.
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
            # Multiclass (One-vs-Rest)
            Y = label_binarize(y_true, classes=classes)
            return roc_auc_score(Y, scores, average="macro", multi_class="ovr")
    except Exception:
        return np.nan

def eval_clfs(X, y, seed):
    """
    Train classifiers and compute metrics.
    
    We train on the SAME embeddings we're testing (not train/test split)
    because we're measuring embedding quality, not generalization.
    
    Args:
        X: Embeddings [num_graphs, dim]
        y: Labels [num_graphs]
        seed: Random seed
        
    Returns:
        dict with metrics for both SVM and MLP
    """
    classes = np.unique(y)
    
    # ===== LINEAR SVM =====
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

    # ===== MLP =====
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
    Measure embedding stability between clean and perturbed versions.
    
    Two complementary metrics:
    1. Cosine similarity: Measures if direction is preserved
    2. L2 distance: Measures magnitude of change
    
    Args:
        X_clean: Clean embeddings [num_graphs, dim]
        X_pert: Perturbed embeddings [num_graphs, dim]
        
    Returns:
        cos: Mean cosine similarity (1.0 = identical direction)
        l2: Mean L2 distance (0.0 = identical)
    """
    # Cosine similarity (row-wise)
    cs = np.diag(cosine_similarity(X_clean, X_pert))
    
    # L2 distance (row-wise)
    l2 = np.linalg.norm(X_clean - X_pert, axis=1)
    
    return float(np.mean(cs)), float(np.mean(l2))

# ============================================================================
# PLOTTING HELPERS
# ============================================================================

def _styled_fig_suptitle(fig, title):
    # Put the title in suptitle with extra top margin to avoid clipping
    fig.suptitle(title, fontsize=16, y=0.98)
    fig.subplots_adjust(top=0.86)

def plot_delta_auc(df, dataset, perturb_type, outpath):
    fig, ax = plt.subplots(figsize=(9,5))
    _styled_fig_suptitle(fig, f"{dataset} — ΔAUC (perturbed − clean) vs. level ({perturb_type})")

    methods = df["method"].unique()
    levels  = sorted(df["level"].unique())

    for meth in methods:
        for dim in sorted(df[df.method==meth]["dim"].unique()):
            sub = df[(df.method==meth) & (df.dim==dim)]
            means = [sub[sub.level==lv]["delta_auc_mlp"].mean() for lv in levels]  # use MLP deltas (or swap to SVM)
            stds  = [sub[sub.level==lv]["delta_auc_mlp"].std()  for lv in levels]
            ax.plot(levels, means, marker="o", label=f"{meth} d={dim}")
            if len(levels) > 1:
                ax.fill_between(levels,
                                np.array(means) - np.nan_to_num(stds),
                                np.array(means) + np.nan_to_num(stds),
                                alpha=0.15)

    ax.axhline(0, lw=1, ls="--", alpha=0.6)
    ax.set_xlabel("Perturbation level (relative)")
    ax.set_ylabel("ΔAUC")
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(ncol=2, frameon=False, loc="best")
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_embed_drift(df, dataset, perturb_type, outpath):
    fig, ax = plt.subplots(figsize=(9,5))
    _styled_fig_suptitle(fig, f"{dataset} — Embedding drift (mean L2) vs. level ({perturb_type})")

    methods = df["method"].unique()
    levels  = sorted(df["level"].unique())

    for meth in methods:
        for dim in sorted(df[df.method==meth]["dim"].unique()):
            sub = df[(df.method==meth) & (df.dim==dim)]
            means = [sub[sub.level==lv]["l2"].mean() for lv in levels]
            stds  = [sub[sub.level==lv]["l2"].std()  for lv in levels]
            ax.plot(levels, means, marker="o", label=f"{meth} d={dim}")
            if len(levels) > 1:
                ax.fill_between(levels,
                                np.array(means) - np.nan_to_num(stds),
                                np.array(means) + np.nan_to_num(stds),
                                alpha=0.15)

    ax.set_xlabel("Perturbation level (relative)")
    ax.set_ylabel("Mean L2 drift")
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(ncol=2, frameon=False, loc="best")
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def run_one_dataset(ds_name, methods, dims, seeds, levels_edges, levels_attrs):
    """
    Run stability analysis for one dataset.
    
    Workflow:
    1. Load dataset
    2. For each (seed, method, dim):
       a. Generate CLEAN embeddings (baseline)
       b. Train classifiers, record metrics
       c. For each perturbation level:
          - Perturb graphs
          - Generate perturbed embeddings
          - Train classifiers, record metrics
          - Compute embedding drift
          - Compute performance drop (Δ)
    3. Return DataFrame with all results
    
    Args:
        ds_name: Dataset name (e.g., "MUTAG")
        methods: List of embedding methods
        dims: List of embedding dimensions
        seeds: List of random seeds
        levels_edges: List of edge perturbation levels
        levels_attrs: List of attribute perturbation levels
        
    Returns:
        DataFrame with stability results
    """
    print(f"\n=== Dataset: {ds_name} ===")
    
    # Load dataset
    ds = TUDataset(root="data", name=ds_name)
    graphs = ensure_node_features([ds[i] for i in range(len(ds))])
    y = ds_labels(ds)
    
    rows = []

    for seed in seeds:
        set_seed(seed)
        
        # Cache for clean embeddings (avoid recomputation)
        cache_clean = {}

        for method in methods:
            for dim in dims:
                print(f"\n--- {method} | dim={dim} | seed={seed} ---")
                
                # ===== STEP 1: CLEAN BASELINE =====
                X_clean = get_embeddings(method, graphs, dim, seed)
                metrics_clean = eval_clfs(X_clean, y, seed)
                cache_clean[(method, dim)] = (X_clean, metrics_clean)

                # ===== STEP 2: EDGE PERTURBATIONS =====
                for lv in levels_edges:
                    # Perturb graphs
                    pert = [perturb_edges(g, lv, seed+123) for g in graphs]
                    
                    # Get perturbed embeddings
                    Xp = get_embeddings(method, pert, dim, seed)
                    metrics_p = eval_clfs(Xp, y, seed)
                    
                    # Embedding stability
                    coss, l2 = emb_stability(X_clean, Xp)

                    row = dict(dataset=ds_name, perturb="edges", level=float(lv),
                               method=method, dim=int(dim), seed=int(seed),
                               cos=float(coss), l2=float(l2))
                    for key in ["acc","f1","auc"]:
                        row[f"delta_{key}_svm"] = float(metrics_p[f"{key}_svm"] - metrics_clean[f"{key}_svm"])
                        row[f"delta_{key}_mlp"] = float(metrics_p[f"{key}_mlp"] - metrics_clean[f"{key}_mlp"])
                    rows.append(row)

                # Attribute perturbations
                for lv in levels_attrs:
                    pert = [perturb_attrs(g, lv, seed+456) for g in graphs]
                    Xp = get_embeddings(method, pert, dim, seed)
                    metrics_p = eval_clfs(Xp, y, seed)
                    coss, l2 = emb_stability(X_clean, Xp)

                    row = dict(dataset=ds_name, perturb="attrs", level=float(lv),
                               method=method, dim=int(dim), seed=int(seed),
                               cos=float(coss), l2=float(l2))
                    for key in ["acc","f1","auc"]:
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
    Aggregate results and generate all stability plots.
    
    For each (dataset, perturbation type, method):
    - Plot embedding drift (L2, Cosine) vs perturbation level
    - Plot classification drops (Δ Acc, Δ F1, Δ AUC) vs perturbation level
    - Separate plots for SVM and MLP
    
    Total plots per configuration: 8
    - 2 embedding metrics (L2, Cosine)
    - 3 classification metrics × 2 classifiers (Acc, F1, AUC for SVM & MLP)
    
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

    # ===== GENERATE PLOTS =====
    # Loop over all combinations
    for ds in df_all["dataset"].unique():
        sub = df_all[df_all["dataset"] == ds]
        for perturb in ["edges", "attrs"]:
            subp = sub[sub["perturb"] == perturb]

            # ΔAUC (MLP) vs level
            plot_delta_auc(
                subp.rename(columns={"delta_auc_mlp": "delta_auc_mlp"}),
                ds, perturb,
                os.path.join(OUT_DIR_FIGS, f"{ds}_delta_auc_{perturb}.png")
            )

            # Embedding drift (L2) vs level
            plot_embed_drift(
                subp,
                ds, perturb,
                os.path.join(OUT_DIR_FIGS, f"{ds}_embed_drift_{perturb}.png")
            )

# --------- CLI ---------
def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Task (c): Stability Analysis with cleaner per-method plots"
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
        help="Embedding methods to test"
    )
    
    # Embedding dimensions
    p.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=[32, 64],
        help="Embedding dimensions to test"
    )
    
    # Random seeds
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1],
        help="Random seeds for reproducibility"
    )

    # Perturbation levels for edges
    p.add_argument(
        "--levels_edges",
        nargs="+",
        type=float,
        default=[0.5, 1.0],
        help="Relative fraction of edges to add/remove."
    )
    
    # Perturbation levels for attributes
    p.add_argument(
        "--levels_attrs",
        nargs="+",
        type=float,
        default=[0.5, 1.0],
        help="Relative fraction of node features to shuffle."
    )

    return p.parse_args()

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Main execution function.
    
    Workflow:
    1. Parse command-line arguments
    2. Run stability analysis for each dataset
    3. Combine results from all datasets
    4. Generate all stability plots
    """
    # Keep environment predictable
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    
    args = parse_args()

    # Run experiments for each dataset
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
    
    # Aggregate and plot
    aggregate_and_plot(df_all)

    print("\n" + "="*70)
    print("STABILITY ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nOutputs saved to:")
    print(f"  - Table: {OUT_DIR_TABLES}/stability_results.csv")
    print(f"  - Figures: {OUT_DIR_FIGS}/")
    print("\nPlot naming convention:")
    print("  {dataset}_{perturb}_{method}_{classifier}_delta_{metric}.png")
    print("  {dataset}_{perturb}_{method}_embed_{drift|cosine}.png")
    print("\nExample:")
    print("  MUTAG_edges_graph2vec_svm_delta_acc.png")
    print("  MUTAG_attrs_netlsd_embed_drift_l2.png")

if __name__ == "__main__":
    main()


# ============================================================================
# USAGE EXAMPLES & INTERPRETATION GUIDE
# ============================================================================

"""
BASIC USAGE:
-----------
# Quick test with defaults
python final_task3.py

# Custom configuration
python final_task3.py \
  --datasets MUTAG ENZYMES \
  --methods graph2vec netlsd \
  --dims 32 64 \
  --seeds 0 1 2 \
  --levels_edges 0.1 0.3 0.5 0.7 1.0 \
  --levels_attrs 0.1 0.3 0.5 0.7 1.0


COMPLETE EVALUATION:
-------------------
python final_task3.py \
  --datasets MUTAG ENZYMES IMDB-MULTI \
  --methods graph2vec netlsd gin \
  --dims 32 64 128 \
  --seeds 0 1 2 3 4 \
  --levels_edges 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
  --levels_attrs 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

"""
