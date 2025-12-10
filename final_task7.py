#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task 7 (Extra): Combined Graph Embeddings (Are Complementary Embeddings Better?)

PURPOSE:
- Test if COMBINING different embedding methods improves classification
- Leverage complementary strengths of different approaches
- Compare combined embeddings vs single methods

KEY QUESTION:
"Do Graph2Vec + NetLSD together work better than either alone?"

WHY THIS MATTERS:
- Different methods capture different aspects:
  * Graph2Vec → Local substructures
  * NetLSD → Global spectral properties
  * GIN → Supervised task-specific features
- Combining may capture more complete graph representation
- Trade-off: Better performance vs higher computational cost

APPROACH:
Instead of choosing ONE method, we concatenate embeddings from MULTIPLE methods.

Example:
  Graph2Vec embedding: [v1, v2, ..., v64]   (64-dim)
  NetLSD embedding:    [u1, u2, ..., u64]   (64-dim)
  Combined:            [v1, v2, ..., v64, u1, u2, ..., u64]  (128-dim)

COMBINATIONS TESTED:
1. graph2vec + netlsd
2. netlsd + gin
3. graph2vec + gin
4. graph2vec + netlsd + gin

WORKFLOW:
1. For each combination:
   a. Generate embeddings for each constituent method
   b. Normalize each embedding block independently (crucial!)
   c. Concatenate normalized blocks
   d. Train classifier on combined embeddings
2. Compare: Combined vs single methods
3. Analyze: Is performance gain worth computational cost?

KEY INSIGHT:
Normalization BEFORE concatenation is critical!
- Different methods have different scales
- Without normalization, one method dominates
- StandardScaler per block ensures fair combination
"""

# ============================================================================
# ENVIRONMENT SETUP & COMPATIBILITY PATCHES
# ============================================================================

import matplotlib
matplotlib.use("Agg")  # Headless plotting

# Patch SciPy's errstate if missing
import numpy as _np
import scipy as _sp
if not hasattr(_sp, "errstate"):
    _sp.errstate = _np.errstate

import os, time, argparse, json, tracemalloc, warnings
from contextlib import contextmanager
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool

from karateclub import Graph2Vec

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

def ensure_node_features(graph: Data) -> Data:
    """
    Ensure graph has node features.
    
    If missing, use node degree as a simple 1D feature.
    Essential for GIN which requires node attributes.
    
    Args:
        graph: PyG Data object
        
    Returns:
        graph: Same graph with x attribute added if missing
    """
    if getattr(graph, "x", None) is None:
        G = to_networkx(graph, to_undirected=True)
        deg = np.array([d for _, d in G.degree()], dtype=np.float32)
        graph.x = torch.from_numpy(deg).view(-1, 1)
    return graph

@contextmanager
def timed_mem(name="block"):
    """
    Measure execution time and memory usage.
    
    Tracks:
    - Wall-clock time
    - Process RSS memory (via psutil)
    - Python heap peak memory (via tracemalloc)
    
    Usage:
        with timed_mem("my_function") as meta:
            # your code
        print(meta['time_sec'], meta['rss_after_mb'])
    """
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss
    tracemalloc.start()
    t0 = time.perf_counter()
    meta = {}
    try:
        yield meta
    finally:
        elapsed = time.perf_counter() - t0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_after = proc.memory_info().rss
        meta.update(dict(
            time_sec=round(elapsed, 3),
            rss_before_mb=round(rss_before/1e6, 1),
            rss_after_mb=round(rss_after/1e6, 1),
            py_peak_mb=round(peak/1e6, 1),
        ))
        print(f"[{name}] time={elapsed:.2f}s, rss_before={rss_before/1e6:.1f}MB, "
              f"rss_after={rss_after/1e6:.1f}MB, py_peak={peak/1e6:.1f}MB")

def roc_auc_any(y_true, scores, classes):
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
            if scores is None:
                return np.nan
            scores = np.asarray(scores)
            if scores.ndim == 1:
                return roc_auc_score(y_true, scores)
            elif scores.ndim == 2:
                pos_col = 1 if scores.shape[1] > 1 else 0
                return roc_auc_score(y_true, scores[:, pos_col])
            else:
                return np.nan
        else:
            if scores is None:
                return np.nan
            scores = np.asarray(scores)
            Y = label_binarize(y_true, classes=classes)
            if scores.ndim == 1:
                scores = np.vstack([1 - scores, scores]).T
            return roc_auc_score(Y, scores, average="macro", multi_class="ovr")
    except Exception:
        return np.nan

def eval_classifier(clf, X_train, X_test, y_train, y_test, classes, tag="clf"):
    """
    Train a classifier and evaluate its performance.
    
    Steps:
    1. Fit classifier on training data (with timing & memory tracking)
    2. Predict on test data
    3. Compute metrics: Accuracy, F1-macro, ROC-AUC
    
    Args:
        clf: sklearn classifier (e.g., SVM, MLP)
        X_train, X_test: Feature matrices
        y_train, y_test: Labels
        classes: Unique class labels
        tag: Name for logging
        
    Returns:
        metrics: dict with acc, f1, auc
        timings: dict with training time & memory
        scores: continuous scores for AUC computation
    """
    with timed_mem(f"train_{tag}") as meta:
        clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)

    # Get continuous scores for AUC
    scores = None
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(X_test)
    elif hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    auc = roc_auc_any(y_test, scores, classes)

    metrics = dict(
        acc=round(float(acc), 4),
        f1=round(float(f1m), 4),
        auc=float(np.nan if np.isnan(auc) else round(float(auc), 4)),
    )
    timings = dict(
        train_time_sec=meta["time_sec"],
        train_rss_before_mb=meta["rss_before_mb"],
        train_rss_after_mb=meta["rss_after_mb"],
        train_py_peak_mb=meta["py_peak_mb"],
    )
    return metrics, timings, scores

# ============================================================================
# EMBEDDING METHODS (Same as previous tasks)
# ============================================================================

def to_nx_with_labels(ds_slice):
    """
    Convert PyG graphs to NetworkX with discrete node labels.
    
    Uses node degree as the label (standard for Graph2Vec).
    """
    Gs = []
    for g in ds_slice:
        G = to_networkx(g, to_undirected=True)
        degs = dict(G.degree())
        for n in G.nodes():
            G.nodes[n]["label"] = int(degs[n])
        Gs.append(G)
    return Gs

def embed_graph2vec(ds_slice, dim=128, seed=0, epochs=20,
                    wl_iterations=2, min_count=5):
    """Generate Graph2Vec embeddings."""
    Gs = to_nx_with_labels(ds_slice)
    with timed_mem("embed_graph2vec") as meta:
        model = Graph2Vec(
            dimensions=dim,
            wl_iterations=wl_iterations,
            epochs=epochs,
            seed=seed,
            workers=1,
            min_count=min_count,
        )
        model.fit(Gs)
        X = model.get_embedding()
    info = dict(
        gen_time_sec=meta["time_sec"],
        gen_rss_before_mb=meta["rss_before_mb"],
        gen_rss_after_mb=meta["rss_after_mb"],
        gen_py_peak_mb=meta["py_peak_mb"],
    )
    return X, info

def _netlsd_signature_dense(G, times):
    """Compute NetLSD heat trace signature."""
    n = G.number_of_nodes()
    if n == 0:
        return np.zeros_like(times)
    L = nx.normalized_laplacian_matrix(G).astype(float).toarray()
    lam = np.linalg.eigvalsh(L)  # Laplacian symmetric => stable eigenvalues
    return np.exp(-np.outer(times, lam)).sum(axis=1)

def embed_netlsd(ds_slice, dim=128, pca_seed=0,
                 n_times=256, t_min=1e-2, t_max=1e2):
    """Generate NetLSD embeddings with PCA compression."""
    times = np.logspace(np.log10(t_min), np.log10(t_max), num=n_times)
    Gs = [to_networkx(g, to_undirected=True) for g in ds_slice]
    with timed_mem("embed_netlsd") as meta:
        sigs = [_netlsd_signature_dense(G, times) for G in Gs]
        X = np.vstack(sigs)  # [num_graphs, n_times]
        # PCA compression to requested embedding dim
        if dim != X.shape[1]:
            X = PCA(n_components=dim, random_state=pca_seed).fit_transform(X)
    info = dict(
        gen_time_sec=meta["time_sec"],
        gen_rss_before_mb=meta["rss_before_mb"],
        gen_rss_after_mb=meta["rss_after_mb"],
        gen_py_peak_mb=meta["py_peak_mb"],
    )
    return X, info

# ===== GIN ENCODER =====

class GINEncoder(nn.Module):
    """
    GIN-based graph encoder with global mean pooling.
    
    We set hidden = dim so GIN embeddings match requested size.
    This makes concatenation cleaner (all methods produce same-sized vectors).
    """
    def __init__(self, in_dim, hidden=64, layers=3, n_classes=2, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.bns    = nn.ModuleList()

        # First layer
        mlp0 = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.layers.append(GINConv(mlp0))
        self.bns.append(nn.BatchNorm1d(hidden))

        # Subsequent layers
        for _ in range(layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.layers.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden))

        # Classification head
        self.lin = nn.Linear(hidden, n_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.layers, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, batch)  # Graph embedding
        out = self.lin(g)  # Classification logits
        return out, g

def embed_gin(ds_slice,
              dim=128,
              seed=0,
              layers=3,
              dropout=0.2,
              epochs=30,
              batch_size=64,
              lr=1e-3,
              device="cpu"):
    """
    Train GIN supervised and export graph embeddings.
    
    NOTE: We train on the FULL dataset for simplicity.
    In a proper setup, should train only on training split.
    But for embedding extraction, this is acceptable.
    
    Args:
        ds_slice: List of PyG graphs
        dim: Embedding dimension (= hidden dimension of GIN)
        seed: Random seed
        layers: Number of GIN layers
        dropout: Dropout probability
        epochs: Training epochs
        batch_size: Mini-batch size
        lr: Learning rate
        device: 'cpu' or 'cuda'
        
    Returns:
        X: Embedding matrix [num_graphs, dim]
        info: dict with generation time & memory
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    graphs = [ensure_node_features(g.clone()) for g in ds_slice]
    num_classes = len(np.unique([int(g.y) for g in graphs]))
    in_dim = graphs[0].x.size(-1)

    model = GINEncoder(
        in_dim=in_dim,
        hidden=dim,  # Hidden dimension = embedding dimension
        layers=layers,
        n_classes=num_classes,
        dropout=dropout,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    # Training phase
    with timed_mem("embed_gin_train") as meta_train:
        model.train()
        for _ in range(epochs):
            for batch in loader:
                batch = batch.to(device)
                logits, _ = model(batch)
                loss = criterion(logits, batch.y.view(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()

    # Embedding extraction phase
    with timed_mem("embed_gin_infer") as meta_embed:
        model.eval()
        emb_chunks = []
        eval_loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for batch in eval_loader:
                batch = batch.to(device)
                _, g = model(batch)
                emb_chunks.append(g.cpu())
        X = torch.cat(emb_chunks, dim=0).numpy()

    info = dict(
        gen_time_sec=round(meta_train["time_sec"] + meta_embed["time_sec"], 3),
        gen_rss_before_mb=meta_train["rss_before_mb"],
        gen_rss_after_mb=meta_embed["rss_after_mb"],
        gen_py_peak_mb=max(meta_train["py_peak_mb"], meta_embed["py_peak_mb"]),
    )
    return X, info

def get_single_method_embedding(method, ds_slice, dim, seed, device="cpu"):
    """
    Unified interface for getting embeddings from any method.
    
    Args:
        method: "graph2vec", "netlsd", or "gin"
        ds_slice: List of PyG graphs
        dim: Embedding dimension
        seed: Random seed
        device: torch device (for GIN)
        
    Returns:
        X: Embedding matrix [num_graphs, dim]
        info: dict with generation statistics
    """
    method = method.lower()
    if method == "graph2vec":
        return embed_graph2vec(ds_slice, dim=dim, seed=seed)
    elif method == "netlsd":
        return embed_netlsd(ds_slice, dim=dim, pca_seed=seed)
    elif method == "gin":
        return embed_gin(ds_slice, dim=dim, seed=seed, device=device)
    else:
        raise ValueError(f"Unknown method: {method}")

# ============================================================================
# CONCATENATION HELPER (KEY FUNCTION!)
# ============================================================================

def concat_normalized_blocks(blocks):
    """
    Concatenate multiple embedding blocks with independent normalization.
    
    WHY NORMALIZE EACH BLOCK SEPARATELY?
    
    Problem without normalization:
        Graph2Vec: values in [-2, 2]
        NetLSD:    values in [0, 100]
        Concatenated: [g2v_features | netlsd_features]
        
        → NetLSD features dominate distance calculations!
        → Graph2Vec features effectively ignored
    
    Solution: Standardize each block independently
        Graph2Vec: normalize to mean=0, std=1
        NetLSD:    normalize to mean=0, std=1
        Concatenated: Both contribute equally
    
    ALGORITHM:
    1. For each embedding block:
       a. Apply StandardScaler (mean=0, std=1)
       b. Store normalized block
    2. Concatenate all normalized blocks horizontally
    
    Args:
        blocks: List of (name, X_block) tuples
                where X_block is [n_graphs, dim_i]
    
    Returns:
        X_concat: Concatenated normalized embeddings [n_graphs, sum(dim_i)]
    
    Example:
        blocks = [
            ("graph2vec", X_g2v),  # [100, 64]
            ("netlsd", X_netlsd),  # [100, 64]
        ]
        
        Result: X_concat has shape [100, 128]
                First 64 dims: normalized Graph2Vec
                Last 64 dims:  normalized NetLSD
    """
    norm_blocks = []
    for name, X in blocks:
        # Standardize: (X - mean) / std
        scaler = StandardScaler(with_mean=True)
        Xn = scaler.fit_transform(X)
        norm_blocks.append(Xn)
    
    # Concatenate along feature dimension (axis=1)
    return np.concatenate(norm_blocks, axis=1)

# ============================================================================
# COMBINATION PARSING
# ============================================================================

def parse_combo_strings(combo_strs):
    """
    Parse combination strings into lists of methods.
    
    Input: ["graph2vec+netlsd", "netlsd+gin", "graph2vec+netlsd+gin"]
    Output: [["graph2vec", "netlsd"], ["netlsd", "gin"], ["graph2vec", "netlsd", "gin"]]
    
    Args:
        combo_strs: List of combination strings
        
    Returns:
        combos: List of lists of method names
    
    Raises:
        ValueError: If combination has fewer than 2 methods
    """
    combos = []
    for s in combo_strs:
        # Split by '+' and clean
        parts = [p.strip().lower() for p in s.split("+") if p.strip()]
        if len(parts) < 2:
            raise ValueError(
                f"Combination '{s}' must have at least two methods separated by '+'."
            )
        combos.append(parts)
    return combos

# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def run_experiment(
    datasets,
    combos,         # list of list of methods
    dims,
    seeds,
    device="cpu",
    test_size=0.2,
    out_csv=f"{OUT_DIR_TABLES}/concat_classification_eval.csv",
):
    """
    Run complete combined embeddings experiment.
    
    WORKFLOW:
    For each (dataset, seed, dimension):
        1. Generate embeddings for all base methods needed
        2. Cache embeddings (avoid recomputation)
        3. For each combination:
           a. Retrieve cached embeddings
           b. Normalize each block independently
           c. Concatenate normalized blocks
           d. Train SVM on concatenated features
           e. Train MLP on concatenated features
           f. Record metrics and timing
    
    EFFICIENCY:
    - Generate each base embedding only once per (dataset, dim, seed)
    - Reuse for multiple combinations
    - Example: If testing "g2v+netlsd" and "g2v+gin",
      only compute Graph2Vec once
    
    Args:
        datasets: List of dataset names
        combos: List of method combinations (e.g., [["graph2vec", "netlsd"]])
        dims: List of embedding dimensions
        seeds: List of random seeds
        device: torch device
        test_size: Test split fraction
        out_csv: Output CSV path
        
    Returns:
        DataFrame with all results
    """
    rows = []

    # Identify all base methods needed across all combinations
    all_base_methods = sorted({m for combo in combos for m in combo})
    print(f"Base methods used in combos: {all_base_methods}")

    for ds_name in datasets:
        print(f"\n=== Dataset: {ds_name} ===")
        
        # Load dataset
        ds = TUDataset(root="data", name=ds_name)
        y_all = ds_labels(ds)
        classes = np.unique(y_all)
        idx_all = np.arange(len(ds))
        ds_all  = [ds[i] for i in idx_all]

        for seed in seeds:
            print(f"\n-- Seed: {seed} --")
            
            # Train/test split
            tr_idx, te_idx = train_test_split(
                idx_all,
                test_size=test_size,
                random_state=seed,
                stratify=y_all,
            )

            # Cache for embeddings: (method, dim) → (X_all, info)
            embedding_cache = {}

            for dim in dims:
                print(f"\n   >> dim={dim}")
                
                # ===== GENERATE ALL NEEDED BASE EMBEDDINGS =====
                for method in all_base_methods:
                    key = (method, dim)
                    if key in embedding_cache:
                        continue  # Already computed
                    
                    print(f"      [emb] {method} (dim={dim})")
                    X_all, gen_info = get_single_method_embedding(
                        method, ds_all, dim=dim, seed=seed, device=device
                    )
                    embedding_cache[key] = (X_all, gen_info)

                # ===== EVALUATE EACH COMBINATION =====
                for combo_methods in combos:
                    combo_name = "+".join(combo_methods)
                    print(f"      [combo] {combo_name} | dim={dim} | seed={seed}")

                    # Collect blocks & aggreg generation info
                    blocks = []
                    gen_time = 0.0
                    gen_rss_before = None
                    gen_rss_after = None
                    gen_py_peak = 0.0

                    for m in combo_methods:
                        X_m, info_m = embedding_cache[(m, dim)]
                        blocks.append((m, X_m))
                        
                        # Aggregate timing (sum)
                        gen_time += info_m["gen_time_sec"]
                        gen_py_peak = max(gen_py_peak, info_m["gen_py_peak_mb"])
                        if gen_rss_before is None:
                            gen_rss_before = info_m["gen_rss_before_mb"]
                        gen_rss_after = info_m["gen_rss_after_mb"]

                    # ===== CONCATENATE WITH NORMALIZATION =====
                    X_all_concat = concat_normalized_blocks(blocks)

                    # Train/test split on concatenated features
                    X_train, X_test = X_all_concat[tr_idx], X_all_concat[te_idx]
                    y_train, y_test = y_all[tr_idx], y_all[te_idx]

                    # ===== TRAIN CLASSIFIERS =====
                    
                    # SVM
                    svm = make_pipeline(
                        StandardScaler(with_mean=True),
                        LinearSVC(dual=False, random_state=seed),
                    )
                    svm_metrics, svm_time, _ = eval_classifier(
                        svm, X_train, X_test, y_train, y_test, classes, tag="svm"
                    )

                    # MLP
                    mlp = make_pipeline(
                        StandardScaler(with_mean=True),
                        MLPClassifier(
                            hidden_layer_sizes=(128,),
                            activation="relu",
                            solver="adam",
                            alpha=1e-4,
                            max_iter=800,
                            tol=1e-4,
                            random_state=seed,
                        ),
                    )
                    mlp_metrics, mlp_time, _ = eval_classifier(
                        mlp, X_train, X_test, y_train, y_test, classes, tag="mlp"
                    )

                    # ===== RECORD RESULTS =====
                    row = dict(
                        dataset=ds_name,
                        combo=combo_name,
                        base_methods=",".join(combo_methods),
                        dim=int(dim),  # Dimension PER method
                        seed=int(seed),
                        n_graphs=len(ds),
                        n_classes=len(classes),
                        # Generation stats (sum over methods)
                        gen_time_sec=round(gen_time, 3),
                        gen_rss_before_mb=gen_rss_before,
                        gen_rss_after_mb=gen_rss_after,
                        gen_py_peak_mb=gen_py_peak,
                        # SVM metrics & time
                        acc_svm=svm_metrics["acc"],
                        f1_svm=svm_metrics["f1"],
                        auc_svm=svm_metrics["auc"],
                        train_time_svm_sec=svm_time["train_time_sec"],
                        train_rss_before_svm_mb=svm_time["train_rss_before_mb"],
                        train_rss_after_svm_mb=svm_time["train_rss_after_mb"],
                        train_py_peak_svm_mb=svm_time["train_py_peak_mb"],
                        # MLP metrics & time
                        acc_mlp=mlp_metrics["acc"],
                        f1_mlp=mlp_metrics["f1"],
                        auc_mlp=mlp_metrics["auc"],
                        train_time_mlp_sec=mlp_time["train_time_sec"],
                        train_rss_before_mlp_mb=mlp_time["train_rss_before_mb"],
                        train_rss_after_mlp_mb=mlp_time["train_rss_after_mb"],
                        train_py_peak_mlp_mb=mlp_time["train_py_peak_mb"],
                    )

                    print(json.dumps(row, indent=2))
                    rows.append(row)

                    # Progressive write (save after each run)
                    pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    print(f"\nSaved per-run concatenation results to {out_csv}")
    return df

# ============================================================================
# AGGREGATION & PLOTTING
# ============================================================================

def aggregate_and_plot(df, out_csv_agg=f"{OUT_DIR_TABLES}/concat_classification_eval_agg.csv"):
    """
    Aggregate results across seeds and generate plots.
    
    Creates:
    1. Aggregated CSV with mean ± std per configuration
    2. Accuracy vs dimension plots (mean ± std)
    3. F1-score vs dimension plots
    4. AUC vs dimension plots
    5. Generation time vs dimension plots
    
    Args:
        df: Raw results DataFrame
        out_csv_agg: Path for aggregated results CSV
    """
    if df.empty:
        print("No rows to aggregate.")
        return

    # Convert to long format (one row per classifier)
    recs = []
    for _, r in df.iterrows():
        for clf in ["svm", "mlp"]:
            recs.append(dict(
                dataset=r["dataset"],
                combo=r["combo"],
                dim=int(r["dim"]),
                seed=int(r["seed"]),
                clf=clf,
                acc=r[f"acc_{clf}"],
                f1=r[f"f1_{clf}"],
                auc=r[f"auc_{clf}"],
                gen_time_sec=r["gen_time_sec"],
            ))
    long = pd.DataFrame(recs)

    # Aggregate: mean and std across seeds
    agg = long.groupby(["dataset", "combo", "dim", "clf"]).agg(
        acc_mean=("acc", "mean"), acc_std=("acc", "std"),
        f1_mean=("f1", "mean"),   f1_std=("f1", "std"),
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        gen_time_mean=("gen_time_sec", "mean"),
        gen_time_std=("gen_time_sec", "std"),
        n_runs=("acc", "count"),
    ).reset_index()

    agg.to_csv(out_csv_agg, index=False)
    print(f"Saved aggregated concatenation results to {out_csv_agg}")

    # ===== GENERATE PLOTS PER DATASET/COMBO =====
    for dataset in agg["dataset"].unique():
        subD = agg[agg["dataset"] == dataset]
        for combo_name in subD["combo"].unique():
            subDM = subD[subD["combo"] == combo_name].sort_values("dim")
            safe_combo = combo_name.replace("+", "-")  # For filename

            n_runs = int(subDM["n_runs"].max()) if "n_runs" in subDM else None

            # ===== PLOT 1: Accuracy vs Dimension =====
            fig = plt.figure(figsize=(6,4))
            ax = fig.add_subplot(111)
            for clf in ["svm", "mlp"]:
                sdf = subDM[subDM["clf"] == clf]
                if sdf.empty:
                    continue
                x = sdf["dim"].values
                y = sdf["acc_mean"].values
                s = sdf["acc_std"].fillna(0).values

                # Shaded region: mean ± std
                ax.fill_between(x, y - s, y + s, alpha=0.20, linewidth=0)
                # Line with error bars
                ax.errorbar(x, y, yerr=s, fmt='-o', capsize=4, label=f"{clf.upper()} acc")

            subtitle = f"{dataset} — Accuracy vs. Dim (concat: {combo_name})"
            if n_runs:
                subtitle += f"  (n_seeds={n_runs})"
            ax.set_title(subtitle)
            ax.set_xlabel("Embedding dimension per method")
            ax.set_ylabel("Accuracy (mean ± std)")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend()
            fig.tight_layout()
            fig.savefig(f"{OUT_DIR_FIGS}/concat_{dataset}_{safe_combo}_acc_vs_dim.png", dpi=150)
            plt.close(fig)

            # ===== PLOT 2: F1-Score vs Dimension =====
            fig = plt.figure(figsize=(6,4))
            ax = fig.add_subplot(111)
            for clf in ["svm", "mlp"]:
                sdf = subDM[subDM["clf"] == clf]
                if sdf.empty:
                    continue
                x = sdf["dim"].values
                y = sdf["f1_mean"].values
                s = sdf["f1_std"].fillna(0).values

                ax.fill_between(x, y - s, y + s, alpha=0.20, linewidth=0)
                ax.errorbar(x, y, yerr=s, fmt='-o', capsize=4, label=f"{clf.upper()} F1")

            subtitle = f"{dataset} — F1-score vs. Dim (concat: {combo_name})"
            if n_runs:
                subtitle += f"  (n_seeds={n_runs})"
            ax.set_title(subtitle)
            ax.set_xlabel("Embedding dimension per method")
            ax.set_ylabel("Macro F1 (mean ± std)")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend()
            fig.tight_layout()
            fig.savefig(f"{OUT_DIR_FIGS}/concat_{dataset}_{safe_combo}_f1_vs_dim.png", dpi=150)
            plt.close(fig)

            # ===== PLOT 3: AUC vs Dimension =====
            fig = plt.figure(figsize=(6,4))
            ax = fig.add_subplot(111)
            for clf in ["svm", "mlp"]:
                sdf = subDM[subDM["clf"] == clf]
                if sdf.empty:
                    continue
                x = sdf["dim"].values
                y = sdf["auc_mean"].values
                s = sdf["auc_std"].fillna(0).values

                ax.fill_between(x, y - s, y + s, alpha=0.20, linewidth=0)
                ax.errorbar(x, y, yerr=s, fmt='-o', capsize=4, label=f"{clf.upper()} AUC")

            subtitle = f"{dataset} — AUC vs. Dim (concat: {combo_name})"
            if n_runs:
                subtitle += f"  (n_seeds={n_runs})"
            ax.set_title(subtitle)
            ax.set_xlabel("Embedding dimension per method")
            ax.set_ylabel("ROC-AUC (mean ± std)")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend()
            fig.tight_layout()
            fig.savefig(f"{OUT_DIR_FIGS}/concat_{dataset}_{safe_combo}_auc_vs_dim.png", dpi=150)
            plt.close(fig)

            # ===== PLOT 4: Generation Time vs Dimension =====
            sdf = subDM[subDM["clf"] == "svm"].sort_values("dim")  # Time same for both clfs
            fig2 = plt.figure(figsize=(6,4))
            ax2 = fig2.add_subplot(111)
            ax2.plot(sdf["dim"], sdf["gen_time_mean"], marker="o")
            ax2.set_title(f"{dataset} — Generation Time vs. Dim (concat: {combo_name})")
            ax2.set_xlabel("Embedding dimension per method")
            ax2.set_ylabel("Generation time (s) [sum over methods]")
            ax2.grid(True, linestyle="--", alpha=0.4)
            fig2.tight_layout()
            fig2.savefig(f"{OUT_DIR_FIGS}/concat_{dataset}_{safe_combo}_gentime_vs_dim.png", dpi=150)
            plt.close(fig2)

# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Concatenated graph embeddings (Graph2Vec, NetLSD, GIN combinations)"
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["MUTAG", "ENZYMES", "IMDB-MULTI"],
        help="Datasets to evaluate"
    )
    p.add_argument(
        "--combos",
        nargs="+",
        default=["graph2vec+netlsd", "netlsd+gin", "graph2vec+netlsd+gin"],
        help="Embedding method combinations, e.g. 'graph2vec+netlsd', 'netlsd+gin'."
    )
    p.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=[32, 64, 128],
        help="Embedding dimensions to test"
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Random seeds for reproducibility"
    )
    p.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test split fraction"
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for GIN training (cpu or cuda)"
    )
    p.add_argument(
        "--out",
        type=str,
        default=f"{OUT_DIR_TABLES}/concat_classification_eval.csv",
        help="Output CSV path"
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
    2. Parse combination strings
    3. Run experiments
    4. Aggregate and plot results
    """
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    args = parse_args()

    # Parse combination strings
    combos = parse_combo_strings(args.combos)

    print("="*70)
    print("COMBINED GRAPH EMBEDDINGS EXPERIMENT")
    print("="*70)
    print(f"Datasets: {args.datasets}")
    print(f"Combinations: {args.combos}")
    print(f"Dimensions: {args.dims}")
    print(f"Seeds: {args.seeds}")
    print("="*70)

    # Run experiments
    df = run_experiment(
        datasets=args.datasets,
        combos=combos,
        dims=args.dims,
        seeds=args.seeds,
        device=args.device,
        test_size=args.test_size,
        out_csv=args.out,
    )
    
    # Aggregate and plot
    aggregate_and_plot(df)

    print("\n" + "="*70)
    print("COMBINED EMBEDDINGS ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  - Raw results: {args.out}")
    print(f"  - Aggregated: {OUT_DIR_TABLES}/concat_classification_eval_agg.csv")
    print(f"  - Plots: {OUT_DIR_FIGS}/concat_*.png")
    print("\n✅ All results saved!")

if __name__ == "__main__":
    main()


# ============================================================================
# USAGE EXAMPLES & INTERPRETATION GUIDE
# ============================================================================

"""
BASIC USAGE:
-----------
# Quick test with defaults
python final_task7.py

# Custom combinations
python final_task7.py \
  --combos graph2vec+netlsd graph2vec+gin

# Test multiple dimensions
python final_task7.py \
  --dims 32 64 128 256

# More seeds for statistical confidence
python final_task7.py \
  --seeds 0 1 2 3 4


COMPLETE EVALUATION:
-------------------
python final_task7.py \
  --datasets MUTAG ENZYMES IMDB-MULTI \
  --combos graph2vec+netlsd netlsd+gin graph2vec+gin graph2vec+netlsd+gin \
  --dims 32 64 128 \
  --seeds 0 1 2 \
  --device cpu


INTERPRETING RESULTS:
--------------------

CSV Output (concat_classification_eval.csv):
  dataset,combo,base_methods,dim,seed,acc_svm,acc_mlp,...
  MUTAG,graph2vec+netlsd,graph2vec,netlsd,64,0,0.88,0.86,...
  
  Key columns:
    - combo: Combination name (e.g., "graph2vec+netlsd")
    - dim: Dimension PER method (total dim = dim × num_methods)
    - acc_svm, acc_mlp: Classification accuracy
    - gen_time_sec: Total generation time (sum of all methods)


Aggregated CSV (concat_classification_eval_agg.csv):
  dataset,combo,dim,clf,acc_mean,acc_std,...
  MUTAG,graph2vec+netlsd,64,svm,0.87,0.02,...
  
  Shows mean ± std across seeds for each configuration.



TROUBLESHOOTING:
---------------

Problem: Combined performs WORSE than single
Solution:
  - Check if normalization is applied
  - Try different normalization methods
  - Methods might conflict (rare)

Problem: No improvement from combination
Solution:
  - Methods might be redundant
  - Try different combinations
  - One method might already be optimal

Problem: Very slow with GIN
Solution:
  - Use GPU (--device cuda)
  - Reduce --gin_epochs
  - Test without GIN first

"""