#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task (d): Cross-Dataset Transferability Analysis

PURPOSE:
- Evaluate how well graph embeddings transfer across different datasets
- Test generalization ability of embedding methods
- Identify which methods produce more universal representations

KEY QUESTION:
"Do embeddings learned on one dataset work well on another?"

WHY THIS MATTERS:
- Real-world scenarios: Limited labeled data in target domain
- Transfer learning: Leverage knowledge from one dataset to another
- Robustness: Good embeddings should capture universal graph properties
- Domain adaptation: Understanding cross-dataset performance helps deployment

TRANSFER LEARNING SETUP:
Source Dataset → Train Embeddings + Classifier
Target Dataset → Apply same Embeddings → Test Classifier
Goal: High target performance = Good transferability

COMPARISON:
- Within-dataset performance: Train and test on same dataset (baseline)
- Cross-dataset performance: Train on source, test on target
- Δ metrics: Performance drop when transferring

WORKFLOW:
1. For each source dataset:
   a. Generate embeddings
   b. Train classifier on source
   c. Evaluate on source (within-dataset baseline)
2. For each target dataset:
   a. Generate embeddings using same method
   b. Apply source classifier
   c. Evaluate on target (cross-dataset transfer)
3. Compare: Δ = within_performance - transfer_performance

METRICS:
- Accuracy, F1 (macro), AUC
- Δ metrics: Performance degradation during transfer
- Lower Δ = Better transferability

VISUALIZATIONS:
1. Heatmaps: Source → Target performance matrix (includes diagonal)
2. Barplots: Cross-dataset only (src ≠ tgt), grouped by direction
3. Scatter: Within vs Transfer performance (cross-dataset only)

METHODS COMPARED:
- Graph2Vec: Unsupervised, substructure-based
- NetLSD: Spectral, based on Laplacian heat trace
- GIN: Supervised, neural message passing
"""

# ============================================================================
# ENVIRONMENT SETUP & COMPATIBILITY PATCHES
# ============================================================================

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for headless plotting

import os, argparse, warnings, random
warnings.filterwarnings("ignore")  # Suppress scientific library warnings

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from karateclub import Graph2Vec  # Graph2Vec embedding library

# PATCH: Fix missing scipy.errstate for NetLSD stability
import scipy as sp
if not hasattr(sp, "errstate"):
    sp.errstate = np.errstate

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================

OUT_DIR_TABLES = "final_report4/tables"
OUT_DIR_FIGS   = "final_report4/figures"
os.makedirs(OUT_DIR_TABLES, exist_ok=True)
os.makedirs(OUT_DIR_FIGS, exist_ok=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed):
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
        ds: PyG dataset or list of graphs
        
    Returns:
        labels: NumPy array of integer labels
    """
    return np.array([int(g.y) for g in ds])

def ensure_node_features(graphs):
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
            # Compute node degrees as features
            deg = torch.bincount(
                g.edge_index[0],
                minlength=g.num_nodes
            ).float().view(-1, 1)
            g = Data(x=deg, edge_index=g.edge_index, y=g.y, num_nodes=g.num_nodes)
        out.append(g)
    return out

def auc_any(y_true, scores, classes):
    """
    Compute ROC-AUC for binary or multiclass problems.
    
    Handles both:
    - Binary classification (2 classes)
    - Multiclass classification (>2 classes) using One-vs-Rest (OvR) macro average
    
    Args:
        y_true: True labels
        scores: Classifier scores (probabilities or decision values)
        classes: Unique class labels
        
    Returns:
        float: AUC score, or NaN if computation fails
    """
    try:
        if len(classes) == 2:
            # Binary classification
            if scores is None:
                return np.nan
            if scores.ndim > 1:
                return roc_auc_score(y_true, scores[:, 1])
            return roc_auc_score(y_true, scores)
        else:
            # Multiclass: use One-vs-Rest (OvR) with macro averaging
            if scores is None:
                return np.nan
            Y = label_binarize(y_true, classes=classes)
            return roc_auc_score(Y, scores, average="macro", multi_class="ovr")
    except Exception:
        return np.nan

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

def embed_graph2vec(graphs, dim=128, seed=0):
    """
    Generate Graph2Vec embeddings.
    
    Graph2Vec learns graph-level embeddings by treating graphs as "documents"
    and their substructures (from Weisfeiler-Lehman) as "words", then applies
    doc2vec-style training.
    
    TRANSFERABILITY:
    Graph2Vec is unsupervised and captures structural patterns.
    Good transferability if source/target share similar motifs.
    
    Args:
        graphs: List of PyG graphs
        dim: Embedding dimension
        seed: Random seed for reproducibility
        
    Returns:
        X: Embedding matrix [num_graphs, dim]
    """
    Gs = to_nx_with_labels(graphs)
    model = Graph2Vec(
        dimensions=dim,
        wl_iterations=2,  # Depth of WL kernel
        epochs=20,        # Training epochs
        seed=seed,
        workers=1,
        min_count=5       # Minimum frequency for substructures
    )
    model.fit(Gs)
    return model.get_embedding()

def _netlsd_signature_dense(G, times):
    """
    Compute NetLSD heat trace signature using dense eigendecomposition.
    
    NetLSD (Network Laplacian Spectral Descriptor) characterizes a graph
    through the heat trace of its normalized Laplacian at different time scales.
    
    Heat trace at time t: h(t) = Σᵢ exp(-t · λᵢ)
    where λᵢ are the eigenvalues of the normalized Laplacian.
    
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

def embed_netlsd(graphs, dim=128, seed=0):
    """
    Generate NetLSD embeddings with PCA compression.
    
    NetLSD is a spectral method based on graph Laplacian properties.
    
    TRANSFERABILITY:
    NetLSD captures spectral properties (connectivity, community structure).
    Good transferability if source/target have similar spectral characteristics.
    
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
    
    # Compute signatures for all graphs
    sigs = []
    for g in graphs:
        G = to_networkx(g, to_undirected=True)
        sigs.append(_netlsd_signature_dense(G, times))
    X = np.vstack(sigs)
    
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
    
    TRANSFERABILITY:
    GIN is trained supervised on source labels.
    Transfer depends on similarity between source/target label distributions.
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
        # First layer: input → hidden
        self.mlps.append(nn.Sequential(
            nn.Linear(in_dim, h),
            nn.ReLU(),
            nn.Linear(h, h)
        ))
        self.convs.append(GINConv(self.mlps[0]))
        
        # Remaining layers: hidden → hidden
        for _ in range(layers - 1):
            mlp = nn.Sequential(
                nn.Linear(h, h),
                nn.ReLU(),
                nn.Linear(h, h)
            )
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
            g: Graph embeddings [batch_size, hidden]
        """
        h = x
        # Message passing layers
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Aggregate to graph-level embedding
        g = global_mean_pool(h, batch)
        
        # Classification
        out = self.lin(g)
        return out, g

def train_gin_embed(graphs, dim=64, seed=0, epochs=30, batch_size=64, lr=1e-3):
    """
    Train GIN encoder and extract graph embeddings.
    
    Training process:
    1. Train GIN for graph classification (supervised)
    2. Extract embeddings from penultimate layer
    3. Use these embeddings for transfer experiments
    
    Args:
        graphs: List of PyG graphs
        dim: Hidden dimension (= embedding dimension)
        seed: Random seed
        epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        
    Returns:
        X: Embedding matrix [num_graphs, dim]
    """
    set_seed(seed)
    graphs = ensure_node_features(graphs)
    in_dim = graphs[0].x.size(1)
    n_classes = int(torch.stack([g.y for g in graphs]).max()) + 1
    
    # Initialize model
    model = GINSmall(in_dim, hidden=dim, n_classes=n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)

    # Training loop
    model.train()
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
            _, emb = model(g.x, g.edge_index, g.batch)
            X.append(emb)
    return torch.cat(X, dim=0).cpu().numpy()

def get_embeddings(method, graphs, dim, seed):
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
    if method == "graph2vec":
        return embed_graph2vec(graphs, dim, seed)
    elif method == "netlsd":
        return embed_netlsd(graphs, dim, seed)
    elif method == "gin":
        return train_gin_embed(graphs, dim, seed)
    else:
        raise ValueError(f"Unknown method: {method}")

# ============================================================================
# CLASSIFIER & EVALUATION
# ============================================================================

def fit_clf(X, y, seed):
    """
    Train MLP classifier on embeddings.
    
    Uses a simple one-hidden-layer MLP with:
    - 128 hidden units
    - ReLU activation
    - Adam optimizer
    - Standard scaling of inputs
    
    Args:
        X: Feature matrix [num_samples, dim]
        y: Labels
        seed: Random seed
        
    Returns:
        clf: Trained classifier (sklearn pipeline)
    """
    clf = make_pipeline(
        StandardScaler(with_mean=True),  # Standardize features
        MLPClassifier(
            hidden_layer_sizes=(128,),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=seed
        )
    )
    clf.fit(X, y)
    return clf

def eval_within(X, y, seed):
    """
    Train and evaluate classifier on the same dataset (within-dataset baseline).
    
    This establishes the baseline performance when training and testing
    on the same distribution.
    
    Args:
        X: Feature matrix
        y: Labels
        seed: Random seed
        
    Returns:
        clf: Trained classifier
        acc: Accuracy
        f1: F1 score (macro)
        auc: AUC score
    """
    clf = fit_clf(X, y, seed)
    y_pred = clf.predict(X)
    y_score = clf.predict_proba(X) if hasattr(clf, "predict_proba") else None

    acc = accuracy_score(y, y_pred)
    f1  = f1_score(y, y_pred, average="macro")
    auc = auc_any(y, y_score, classes=np.unique(y))

    return clf, acc, f1, auc

def eval_transfer(clf, X_tgt, y_tgt):
    """
    Evaluate pre-trained classifier on target dataset (cross-dataset transfer).
    
    This measures how well the source-trained classifier performs on
    target data, revealing transferability of the embedding method.
    
    TRANSFER PROCESS:
    1. Classifier is already trained on source embeddings
    2. Apply to target embeddings (no retraining)
    3. Measure performance on target labels
    
    Args:
        clf: Pre-trained classifier (from source dataset)
        X_tgt: Target embeddings
        y_tgt: Target labels
        
    Returns:
        acc: Accuracy on target
        f1: F1 score (macro) on target
        auc: AUC score on target
    """
    y_pred = clf.predict(X_tgt)
    y_score = clf.predict_proba(X_tgt) if hasattr(clf, "predict_proba") else None

    acc = accuracy_score(y_tgt, y_pred)
    f1  = f1_score(y_tgt, y_pred, average="macro")
    auc = auc_any(y_tgt, y_score, classes=np.unique(y_tgt))

    return acc, f1, auc

# ============================================================================
# MAIN TRANSFER EXPERIMENT
# ============================================================================

def run_transfer(datasets, methods, dims, seeds):
    """
    Run complete transfer learning experiment.
    
    EXPERIMENTAL DESIGN:
    For each combination of (seed, dim, method):
        For each source dataset:
            1. Generate embeddings on source
            2. Train classifier on source
            3. Evaluate on source (within-dataset baseline)
            
            For each target dataset:
                4. Generate embeddings on target
                5. Apply source classifier to target embeddings
                6. Evaluate on target (cross-dataset transfer)
                7. Compute Δ metrics (performance drop)
    
    METRICS COMPUTED:
    - acc, f1, auc: Performance on target
    - acc_src, f1_src, auc_src: Within-source baseline
    - delta_*: Performance degradation (src - tgt)
    
    Lower Δ = Better transferability
    
    Args:
        datasets: List of dataset names
        methods: List of embedding methods
        dims: List of embedding dimensions
        seeds: List of random seeds
        
    Returns:
        df: DataFrame with all transfer results
    """
    results = []
    data_cache = {}

    # Load all datasets once for efficiency
    print("Loading datasets...")
    for ds_name in datasets:
        ds = TUDataset(root="data", name=ds_name)
        data_cache[ds_name] = [ds[i] for i in range(len(ds))]
        print(f"  {ds_name}: {len(data_cache[ds_name])} graphs")

    # Main experiment loop
    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        
        for dim in dims:
            print(f"  Dimension: {dim}")
            
            for method in methods:
                print(f"    Method: {method.upper()}")
                
                # For each source dataset
                for src in datasets:
                    # Generate source embeddings
                    graphs_src = ensure_node_features(data_cache[src])
                    X_src = get_embeddings(method, graphs_src, dim, seed)
                    y_src = ds_labels(data_cache[src])

                    # Train classifier on source (within-dataset)
                    clf, acc_src, f1_src, auc_src = eval_within(X_src, y_src, seed)

                    # Test on all target datasets
                    for tgt in datasets:
                        # Generate target embeddings
                        graphs_tgt = ensure_node_features(data_cache[tgt])
                        X_tgt = get_embeddings(method, graphs_tgt, dim, seed)
                        y_tgt = ds_labels(data_cache[tgt])

                        # Evaluate transfer (apply source classifier to target)
                        acc, f1, auc = eval_transfer(clf, X_tgt, y_tgt)

                        # Store all results
                        res = dict(
                            src=src,
                            tgt=tgt,
                            method=method,
                            dim=dim,
                            seed=seed,
                            
                            # Target performance
                            acc=acc,
                            f1=f1,
                            auc=auc,
                            
                            # Source baseline
                            acc_src=acc_src,
                            f1_src=f1_src,
                            auc_src=auc_src,
                            
                            # Performance degradation
                            delta_acc=acc_src - acc,
                            delta_f1=f1_src - f1,
                            delta_auc=auc_src - auc,
                        )
                        results.append(res)

                        # Log results
                        print(
                            f"      {src}->{tgt} | "
                            f"ACC={acc:.3f} (Δ={res['delta_acc']:+.3f})  "
                            f"F1={f1:.3f} (Δ={res['delta_f1']:+.3f})  "
                            f"AUC={auc:.3f} (Δ={res['delta_auc']:+.3f})"
                        )

    # Save results to CSV
    df = pd.DataFrame(results)
    out_csv = os.path.join(OUT_DIR_TABLES, "transfer_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Saved results to {out_csv}")
    return df

# ============================================================================
# VISUALIZATION
# ============================================================================

def _pivot(df, metric, method, dim):
    """
    Create pivot table for heatmap visualization.
    
    Args:
        df: Results DataFrame
        metric: Metric name (acc, f1, auc)
        method: Embedding method
        dim: Embedding dimension
        
    Returns:
        pivot: Pivot table with sources as rows, targets as columns
    """
    sub = df[(df["method"] == method) & (df["dim"] == dim)]
    return sub.pivot_table(values=metric, index="src", columns="tgt", aggfunc="mean")

def plot_heatmaps(df):
    """
    Generate transfer heatmaps showing source → target performance.
    
    HEATMAP INTERPRETATION:
    - Rows: Source datasets
    - Columns: Target datasets
    - Diagonal: Within-dataset performance (baseline)
    - Off-diagonal: Cross-dataset transfer performance
    - Brighter = Better performance
    
    Creates one heatmap per (method, dimension, metric) combination.
    
    Args:
        df: Results DataFrame
    """
    metrics = ["auc", "acc", "f1"]
    
    for method in df["method"].unique():
        for dim in sorted(df["dim"].unique()):
            for metric in metrics:
                # Create pivot table
                pivot = _pivot(df, metric, method, dim)
                
                # Plot heatmap
                plt.figure(figsize=(6, 5))
                sns.heatmap(
                    pivot,
                    annot=True,      # Show values
                    cmap="YlGnBu",   # Yellow-Green-Blue colormap
                    fmt=".2f",       # 2 decimal places
                    vmin=0,          # Scale from 0
                    vmax=1           # to 1
                )
                plt.title(
                    f"{method.upper()} — Transfer {metric.upper()} (dim={dim})\n"
                    f"Rows=Source, Cols=Target"
                )
                plt.tight_layout()
                
                # Save figure
                outpath = f"{OUT_DIR_FIGS}/{method}_transfer_heatmap_{metric}_d{dim}.png"
                plt.savefig(outpath, dpi=150)
                plt.close()
                print(f"  Saved: {outpath}")

def plot_barplots_cross(df):
    """
    Generate barplots for cross-dataset transfer only (src ≠ tgt).
    
    BARPLOT INTERPRETATION:
    - X-axis: Embedding method
    - Y-axis: Performance metric
    - Bars: Grouped by transfer direction (src→tgt)
    - Excludes diagonal (within-dataset)
    
    This focuses on true transfer scenarios, ignoring baseline performance.
    
    Creates one barplot per metric (acc, f1, auc).
    
    Args:
        df: Results DataFrame
    """
    # Filter to cross-dataset only
    df_cross = df[df["src"] != df["tgt"]].copy()
    df_cross["pair"] = df_cross["src"] + "→" + df_cross["tgt"]

    metrics = ["auc", "acc", "f1"]
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=df_cross,
            x="method",
            y=metric,
            hue="pair",
            estimator=np.mean,
            errorbar=None
        )
        plt.ylabel(metric.upper())
        plt.xlabel("Embedding Method")
        plt.title(
            f"Cross-Dataset {metric.upper()} by Transfer Direction\n"
            f"(src→tgt pairs, excluding within-dataset)"
        )
        plt.legend(title="Transfer Direction", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR_FIGS}/transfer_barplot_{metric}_cross_only.png", dpi=150)
        plt.close()
        print(f"  Saved: {outpath}")

def plot_scatter_cross(df):
    """
    Generate scatter plots comparing within vs transfer performance.
    
    SCATTER INTERPRETATION:
    - X-axis: Within-source performance (src→src)
    - Y-axis: Transfer performance (src→tgt)
    - Diagonal line: y = x (no performance drop)
    - Points below line: Performance degradation
    - Distance from line: Magnitude of transfer gap
    
    Colors: Different embedding methods
    Shapes: Different transfer pairs
    
    Creates separate scatter plots for each metric (acc, f1, auc).
    Only includes cross-dataset transfers (src ≠ tgt).
    
    Args:
        df: Results DataFrame
    """
    # Filter to cross-dataset only
    df_cross = df[df["src"] != df["tgt"]].copy()
    df_cross["pair"] = df_cross["src"] + "→" + df_cross["tgt"]

    # Define metric pairs (source_col, target_col, name)
    metric_pairs = [
        ("auc_src", "auc", "auc"),
        ("acc_src", "acc", "acc"),
        ("f1_src",  "f1",  "f1"),
    ]

    for xcol, ycol, tag in metric_pairs:
        plt.figure(figsize=(6, 6))
        sns.scatterplot(
            data=df_cross,
            x=xcol,
            y=ycol,
            hue="method",
            style="pair",
            s=80
        )
        # y = x reference line, just to see degradation
        plt.plot([0,1],[0,1],'k--',alpha=0.5)

        plt.xlabel(f"Within-Source {tag.upper()} (src→src)")
        plt.ylabel(f"Transfer {tag.upper()} (src→tgt)")
        plt.title(f"Cross Transfer vs. Within Performance ({tag.upper()})\n(src != tgt only)")
        plt.tight_layout()

        outpath = f"{OUT_DIR_FIGS}/transfer_scatter_{tag}_cross_only.png"
        plt.savefig(outpath, dpi=150)
        plt.close()
        print(f"  Saved: {outpath}")

# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Task (d): Cross-Dataset Transferability Analysis"
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["MUTAG", "ENZYMES", "IMDB-MULTI"],
        help="Datasets to use for transfer experiments"
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
        default=[0],
        help="Random seeds for reproducibility"
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
    2. Run transfer learning experiments
    3. Generate visualizations:
       - Heatmaps (all pairs including diagonal)
       - Barplots (cross-dataset only)
       - Scatter plots (cross-dataset only)
    4. Save results and figures
    
    OUTPUT FILES:
    - report/tables/transfer_results.csv: All results
    - report/figures/*_heatmap_*.png: Transfer heatmaps
    - report/figures/transfer_barplot_*.png: Cross-dataset bars
    - report/figures/transfer_scatter_*.png: Within vs transfer
    """
    args = parse_args()
    
    print("="*70)
    print("CROSS-DATASET TRANSFERABILITY ANALYSIS")
    print("="*70)
    print(f"Datasets: {args.datasets}")
    print(f"Methods: {args.methods}")
    print(f"Dimensions: {args.dims}")
    print(f"Seeds: {args.seeds}")
    print("="*70)
    
    # Run transfer experiments
    df = run_transfer(args.datasets, args.methods, args.dims, args.seeds)

    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    print("\n1. Heatmaps (includes within-dataset baseline):")
    plot_heatmaps(df)

    print("\n2. Barplots (cross-dataset only):")
    plot_barplots_cross(df)

    print("\n3. Scatter plots (within vs transfer):")
    plot_scatter_cross(df)

    print("\n" + "="*70)
    print("TRANSFER ANALYSIS COMPLETE!")
    print("="*70)
    print("\nOutputs:")
    print(f"  - Results: {OUT_DIR_TABLES}/transfer_results.csv")
    print(f"  - Figures: {OUT_DIR_FIGS}/*.png")
    print("\n✅ All results saved!")


if __name__ == "__main__":
    main()


# ============================================================================
# INTERPRETATION GUIDE
# ============================================================================

"""
HOW TO INTERPRET RESULTS:

1. HEATMAP ANALYSIS:
   - Diagonal values: Within-dataset performance (baseline)
   - Off-diagonal values: Cross-dataset transfer
   - Compare diagonal vs off-diagonal: Transfer gap
   - Symmetric patterns: Similar transfer in both directions
   - Asymmetric: One-way transfer works better

2. BARPLOT ANALYSIS:
   - Compare bars across methods for same transfer pair
   - Higher bars = Better transfer performance
   - Look for methods with consistently high bars

3. SCATTER PLOT ANALYSIS:
   - Points on diagonal: Perfect transfer (no degradation)
   - Points below diagonal: Performance drop during transfer
   - Vertical distance from diagonal: Transfer gap
   - Points close to diagonal: Good transferability
   - Clustered points: Consistent behavior across pairs

4. DELTA METRICS:
   - Δ = within_performance - transfer_performance
   - Positive Δ: Performance drop (expected)
   - Small Δ: Good transferability
   - Large Δ: Poor transfer, method is dataset-specific
   - Negative Δ: Transfer better than baseline (rare, check for overfitting)

5. METHOD COMPARISON:
   - Graph2Vec: Often transfers well for similar graph types
   - NetLSD: Good for spectral similarity
   - GIN: May be more dataset-specific (supervised)

6. DATASET PAIRS TO EXAMINE:
   - MUTAG ↔ ENZYMES: Both molecular graphs
   - IMDB-MULTI: Social network, may not transfer well to molecular
"""