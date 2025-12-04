#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task (d): Cross-Dataset Transferability

PURPOSE:
- Test how well embeddings learned on one dataset (SOURCE) generalize to another (TARGET)
- Evaluate transfer learning capability of different embedding methods
- Identify which methods produce more universal/transferable representations

KEY QUESTION:
"If I train a classifier on dataset A's embeddings, how well does it work on dataset B?"

WHY THIS MATTERS:
- Real-world: Often have labeled data from one domain, need to apply to another
- Example: Drug discovery on one protein family, apply to another
- Example: Social network analysis trained on Twitter, applied to Facebook
- Good transferability → Embeddings capture fundamental graph properties
- Poor transferability → Embeddings are dataset-specific

WORKFLOW:
1. For each SOURCE dataset:
   a. Generate embeddings for all SOURCE graphs
   b. Train classifier on SOURCE embeddings + labels
   c. Record SOURCE performance (within-dataset baseline)
   
2. For each TARGET dataset:
   a. Generate embeddings for all TARGET graphs
   b. Apply SOURCE classifier to TARGET embeddings (NO retraining!)
   c. Evaluate on TARGET labels
   d. Compare: within-dataset vs cross-dataset performance

3. Compute transfer gap: Δ AUC = AUC_within - AUC_cross
   - Small gap → Good transferability
   - Large gap → Poor transferability

4. Visualize:
   - Heatmaps: All source→target combinations
   - Scatter: Within-dataset vs cross-dataset AUC
   - Bar plots: Method comparison
"""

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

import matplotlib
matplotlib.use("Agg")  # Headless plotting

import os, time, argparse, warnings, random
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns  # For heatmaps

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
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from karateclub import Graph2Vec

# ============================================================================
# COMPATIBILITY PATCH
# ============================================================================

# Fix for SciPy errstate (needed for NetLSD stability)
import scipy as sp
if not hasattr(sp, "errstate"):
    sp.errstate = np.errstate

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================

OUT_DIR_TABLES = "report/tables"
OUT_DIR_FIGS   = "report/figures"
os.makedirs(OUT_DIR_TABLES, exist_ok=True)
os.makedirs(OUT_DIR_FIGS, exist_ok=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def ds_labels(ds):
    """Extract integer labels from PyTorch Geometric dataset."""
    return np.array([int(g.y) for g in ds])

def ensure_node_features(graphs):
    """
    Ensure all graphs have node features.
    
    If a graph lacks features, use node degree as a simple 1D feature.
    This is essential for GIN which requires node attributes.
    
    Args:
        graphs: List of PyG Data objects
        
    Returns:
        graphs: Same list with x attribute added if missing
    """
    out = []
    for g in graphs:
        if getattr(g, "x", None) is None:
            # Compute node degrees
            deg = torch.bincount(
                g.edge_index[0],
                minlength=g.num_nodes
            ).float().view(-1, 1)
            # Create new graph with degree features
            g = Data(x=deg, edge_index=g.edge_index, y=g.y, num_nodes=g.num_nodes)
        out.append(g)
    return out

def auc_any(y_true, scores, classes):
    """
    Compute AUC for binary or multiclass problems.
    
    Handles different classifier output formats gracefully.
    Returns NaN if computation fails (e.g., single class in predictions).
    
    Args:
        y_true: True labels
        scores: Predicted probabilities or decision scores
        classes: Unique class labels
        
    Returns:
        float: AUC score or NaN
    """
    try:
        if len(classes) == 2:
            # Binary classification
            # Use positive class probability (column 1) if 2D
            if scores.ndim > 1:
                return roc_auc_score(y_true, scores[:, 1])
            else:
                return roc_auc_score(y_true, scores)
        else:
            # Multiclass: One-vs-Rest (OvR) with macro averaging
            Y = label_binarize(y_true, classes=classes)
            return roc_auc_score(Y, scores, average="macro", multi_class="ovr")
    except Exception:
        # Return NaN if AUC cannot be computed
        # (e.g., only one class present in predictions)
        return np.nan

# ============================================================================
# EMBEDDING METHODS
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
        for n in G.nodes:
            G.nodes[n]["label"] = int(degs[n])
        Gs.append(G)
    return Gs

def embed_graph2vec(graphs, dim=128, seed=0):
    """
    Generate Graph2Vec embeddings (unsupervised).
    
    Graph2Vec treats graphs as documents and substructures as words,
    then applies doc2vec-style learning.
    
    Args:
        graphs: List of PyG graphs
        dim: Embedding dimension
        seed: Random seed
        
    Returns:
        X: Embedding matrix [num_graphs, dim]
    """
    Gs = to_nx_with_labels(graphs)
    model = Graph2Vec(
        dimensions=dim,
        wl_iterations=2,  # Weisfeiler-Lehman depth
        epochs=20,
        seed=seed,
        workers=1,
        min_count=5
    )
    model.fit(Gs)
    return model.get_embedding()

def _netlsd_signature_dense(G, times):
    """
    Compute NetLSD heat trace signature using dense eigendecomposition.
    
    Heat trace: h(t) = sum_i exp(-t * lambda_i)
    where lambda_i are eigenvalues of normalized Laplacian.
    
    Args:
        G: NetworkX graph
        times: Array of diffusion times
        
    Returns:
        Array of heat trace values
    """
    n = G.number_of_nodes()
    if n == 0:
        return np.zeros_like(times)
    
    # Normalized Laplacian
    L = nx.normalized_laplacian_matrix(G).astype(float).toarray()
    
    # Eigenvalues
    lam = np.linalg.eigvalsh(L)
    
    # Heat trace at each time
    return np.exp(-np.outer(times, lam)).sum(axis=1)

def embed_netlsd(graphs, dim=128, seed=0):
    """
    Generate NetLSD embeddings (unsupervised, spectral).
    
    NetLSD captures graph structure through heat diffusion signatures
    at logarithmically-spaced time scales.
    
    Args:
        graphs: List of PyG graphs
        dim: Target embedding dimension (after PCA)
        seed: Random seed for PCA
        
    Returns:
        X: Embedding matrix [num_graphs, dim]
    """
    times = np.logspace(-2, 2, 256)  # 256 time points from 0.01 to 100
    
    sigs = []
    for g in graphs:
        G = to_networkx(g, to_undirected=True)
        sigs.append(_netlsd_signature_dense(G, times))
    
    X = np.vstack(sigs)  # [num_graphs, 256]
    
    # Apply PCA to compress to target dimension
    if dim != X.shape[1]:
        X = PCA(n_components=dim, random_state=seed).fit_transform(X)
    
    return X

# ===== GIN ENCODER (Supervised) =====

class GINSmall(nn.Module):
    """
    Lightweight GIN encoder for graph embeddings.
    
    Architecture:
    - Stack of GIN convolution layers (message passing)
    - Global mean pooling (node features → graph feature)
    - Classification head (for supervised training)
    """
    
    def __init__(self, in_dim, hidden=64, layers=3, n_classes=2, dropout=0.2):
        """
        Args:
            in_dim: Input feature dimension
            hidden: Hidden layer size (= embedding dimension)
            layers: Number of GIN layers
            n_classes: Number of classes (for classification head)
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = dropout
        self.mlps = nn.ModuleList()
        self.convs = nn.ModuleList()
        
        h = hidden
        
        # First GIN layer
        self.mlps.append(nn.Sequential(
            nn.Linear(in_dim, h),
            nn.ReLU(),
            nn.Linear(h, h)
        ))
        self.convs.append(GINConv(self.mlps[0]))
        
        # Additional GIN layers
        for _ in range(layers - 1):
            mlp = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
            self.mlps.append(mlp)
            self.convs.append(GINConv(mlp))
        
        # Classification head
        self.lin = nn.Linear(h, n_classes)

    def forward(self, x, edge_index, batch):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment [num_nodes]
            
        Returns:
            out: Classification logits [num_graphs, n_classes]
            g: Graph embeddings [num_graphs, hidden]
        """
        h = x
        
        # Message passing through GIN layers
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Graph-level pooling
        g = global_mean_pool(h, batch)  # [num_graphs, hidden]
        
        # Classification
        out = self.lin(g)
        
        return out, g

def train_gin_embed(graphs, dim=64, seed=0, epochs=30, batch_size=64, lr=1e-3):
    """
    Train GIN with supervision, then extract embeddings.
    
    Even though this is a transfer learning task, we train GIN
    with labels on the source dataset to get good embeddings.
    
    Args:
        graphs: List of PyG graphs
        dim: Embedding dimension (= hidden dimension)
        seed: Random seed
        epochs: Training epochs
        batch_size: Mini-batch size
        lr: Learning rate
        
    Returns:
        X: Graph embeddings [num_graphs, dim]
    """
    set_seed(seed)
    
    # Ensure node features exist
    graphs = ensure_node_features(graphs)
    in_dim = graphs[0].x.size(1)
    n_classes = int(torch.stack([g.y for g in graphs]).max()) + 1
    
    # Initialize model
    model = GINSmall(in_dim, hidden=dim, n_classes=n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Data loader
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)

    # ===== TRAINING PHASE =====
    model.train()
    for _ in range(epochs):
        for batch in loader:
            opt.zero_grad()
            logits, _ = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            opt.step()

    # ===== EMBEDDING EXTRACTION PHASE =====
    model.eval()
    X = []
    with torch.no_grad():
        for g in DataLoader(graphs, batch_size=batch_size, shuffle=False):
            _, emb = model(g.x, g.edge_index, g.batch)
            X.append(emb)
    
    return torch.cat(X, dim=0).cpu().numpy()

def get_embeddings(method, graphs, dim, seed):
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
    if method == "graph2vec":
        return embed_graph2vec(graphs, dim, seed)
    elif method == "netlsd":
        return embed_netlsd(graphs, dim, seed)
    elif method == "gin":
        return train_gin_embed(graphs, dim, seed)
    else:
        raise ValueError(method)


# ---------------- Classifier & evaluation ----------------
def eval_classifier(X_train, y_train, X_test, y_test, seed):
    classes = np.unique(y_train)
    clf = make_pipeline(StandardScaler(with_mean=True), MLPClassifier(hidden_layer_sizes=(128,),
                                                                      activation="relu",
                                                                      solver="adam",
                                                                      max_iter=500,
                                                                      random_state=seed))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro")
    auc = auc_any(y_test, y_score, classes=np.unique(y_test))
    return acc, f1, auc


def run_transfer(datasets, methods, dims, seeds):
    """
    Run complete transfer learning experiment.
    
    EXPERIMENT DESIGN:
    For each (seed, dimension, method):
        For each SOURCE dataset:
            1. Generate SOURCE embeddings
            2. Train classifier on SOURCE
            3. Record SOURCE performance (baseline)
            
            For each TARGET dataset:
                4. Generate TARGET embeddings
                5. Apply SOURCE classifier to TARGET (no retraining!)
                6. Evaluate on TARGET labels
                7. Compute transfer gap: Δ AUC = AUC_source - AUC_target
    
    KEY INSIGHT:
    - When source == target: Measures within-dataset performance
    - When source != target: Measures cross-dataset transferability
    - Compare diagonal (within) vs off-diagonal (cross) in heatmap
    
    Args:
        datasets: List of dataset names (e.g., ["MUTAG", "ENZYMES"])
        methods: List of embedding methods
        dims: List of embedding dimensions
        seeds: List of random seeds
        
    Returns:
        DataFrame with all transfer results
    """
    results = []
    
    # Pre-load all datasets (cache to avoid reloading)
    data_cache = {}
    for ds_name in datasets:
        print(f"Loading {ds_name}...")
        ds = TUDataset(root="data", name=ds_name)
        data_cache[ds_name] = [ds[i] for i in range(len(ds))]

    # ===== MAIN LOOP =====
    for seed in seeds:
        for dim in dims:
            for method in methods:
                print(f"\n=== {method.upper()} | dim={dim} | seed={seed} ===")
                
                # Loop over SOURCE datasets
                for src in datasets:
                    print(f"\nSource: {src}")
                    
                    # ===== STEP 1: GENERATE SOURCE EMBEDDINGS =====
                    graphs_src = ensure_node_features(data_cache[src])
                    X_src = get_embeddings(method, graphs_src, dim, seed)
                    y_src = ds_labels(data_cache[src])

                    # ===== STEP 2: TRAIN CLASSIFIER ON SOURCE =====
                    # Use MLP classifier (could also use SVM)
                    clf = make_pipeline(
                        StandardScaler(with_mean=True),
                        MLPClassifier(
                            hidden_layer_sizes=(128,),
                            activation="relu",
                            solver="adam",
                            max_iter=500,
                            random_state=seed
                        )
                    )
                    clf.fit(X_src, y_src)
                    
                    # ===== STEP 3: EVALUATE ON SOURCE (BASELINE) =====
                    y_pred_src = clf.predict(X_src)
                    y_score_src = clf.predict_proba(X_src) if hasattr(clf, "predict_proba") else None
                    auc_src = auc_any(y_src, y_score_src, np.unique(y_src))

                    # ===== STEP 4: TEST ON ALL TARGET DATASETS =====
                    for tgt in datasets:
                        # Generate TARGET embeddings
                        graphs_tgt = ensure_node_features(data_cache[tgt])
                        X_tgt = get_embeddings(method, graphs_tgt, dim, seed)
                        y_tgt = ds_labels(data_cache[tgt])

                        # ===== APPLY SOURCE CLASSIFIER TO TARGET =====
                        # NO RETRAINING! Just predict using source classifier
                        y_pred_tgt = clf.predict(X_tgt)
                        y_score_tgt = clf.predict_proba(X_tgt) if hasattr(clf, "predict_proba") else None

                        # ===== EVALUATE ON TARGET =====
                        acc = accuracy_score(y_tgt, y_pred_tgt)
                        f1  = f1_score(y_tgt, y_pred_tgt, average="macro")
                        auc = auc_any(y_tgt, y_score_tgt, np.unique(y_tgt))
                        
                        # ===== COMPUTE TRANSFER GAP =====
                        # Δ AUC = performance drop when transferring
                        # Small Δ → Good transferability
                        # Large Δ → Poor transferability
                        delta_auc = auc_src - auc

                        # Save results
                        results.append(dict(
                            src=src,           # Source dataset
                            tgt=tgt,           # Target dataset
                            method=method,     # Embedding method
                            dim=dim,           # Embedding dimension
                            seed=seed,         # Random seed
                            acc=acc,           # Target accuracy
                            f1=f1,             # Target F1
                            auc=auc,           # Target AUC
                            auc_src=auc_src,   # Source AUC (baseline)
                            delta_auc=delta_auc # Transfer gap
                        ))

                        print(f"{method} {src}->{tgt} dim={dim} seed={seed} AUC={auc:.3f}")

    # ===== SAVE RESULTS =====
    df = pd.DataFrame(results)
    out_csv = os.path.join(OUT_DIR_TABLES, "transfer_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Saved results to {out_csv}")
    
    return df

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_heatmaps(df):
    """
    Create transfer heatmaps for each (method, dimension).
    
    Heatmap visualization:
    - Rows: Source datasets
    - Columns: Target datasets
    - Values: AUC scores
    - Diagonal: Within-dataset performance (best case)
    - Off-diagonal: Cross-dataset transfer (what we're testing)
    
    INTERPRETATION:
    - Bright diagonal, dark off-diagonal → Poor transferability
    - Uniformly bright → Excellent transferability
    - Specific patterns reveal dataset relationships
    
    Args:
        df: Results DataFrame
    """
    for method in df["method"].unique():
        sub = df[df["method"] == method]
        
        for dim in sorted(sub["dim"].unique()):
            pivot = sub[sub["dim"] == dim].pivot_table(values="auc", index="src", columns="tgt", aggfunc="mean")
            plt.figure(figsize=(6,5))
            sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".2f")
            plt.title(f"{method.upper()} — Transfer AUC (dim={dim})")
            plt.tight_layout()
            plt.savefig(f"{OUT_DIR_FIGS}/{method}_transfer_heatmap_d{dim}.png", dpi=150)
            plt.close()

def plot_barplots(df):
    """
    Create bar plots comparing methods across target datasets.
    
    Bar plot visualization:
    - X-axis: Embedding method
    - Y-axis: AUC score
    - Colors: Different target datasets
    - Grouped bars for comparison
    
    INTERPRETATION:
    - Taller bars → Better performance
    - Similar height across targets → Good transferability
    - Variable height → Dataset-specific performance
    
    Args:
        df: Results DataFrame
    """
    plt.figure(figsize=(8, 6))
    
    # Create grouped bar plot
    sns.barplot(
        data=df,
        x="method",
        y="auc",
        hue="tgt",     # Color by target dataset
    )
    
    plt.title("Cross-Dataset AUC (All Methods)")
    plt.tight_layout()
    
    # Save figure
    outpath = f"{OUT_DIR_FIGS}/transfer_barplot_all.png"
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved: {outpath}")

def plot_scatter(df):
    """
    Create scatter plot: within-dataset vs cross-dataset AUC.
    
    Scatter plot visualization:
    - X-axis: Within-dataset AUC (source performance)
    - Y-axis: Cross-dataset AUC (target performance)
    - Diagonal line: Perfect transfer (x = y)
    - Points below diagonal: Performance drops during transfer
    - Distance from diagonal: Transfer gap
    
    INTERPRETATION:
    - Points near diagonal → Good transferability
    - Points far below diagonal → Large transfer gap
    - Clustering by method → Method-specific transfer patterns
    
    Args:
        df: Results DataFrame
    """
    plt.figure(figsize=(6, 6))
    
    # Create scatter plot
    sns.scatterplot(
        data=df,
        x="auc_src",   # Within-dataset AUC (x-axis)
        y="auc",       # Cross-dataset AUC (y-axis)
        hue="method",  # Color by embedding method
        style="tgt",   # Marker shape by target dataset
        s=80,          # Marker size
    )
    
    # Add diagonal line (perfect transfer)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Transfer')
    
    plt.xlabel("Within-Dataset AUC (Source)")
    plt.ylabel("Cross-Dataset AUC (Target)")
    plt.title("Transferability vs. Within-Dataset Performance")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    outpath = f"{OUT_DIR_FIGS}/transfer_scatter.png"
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved: {outpath}")

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
        help="Datasets to evaluate"
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=["graph2vec", "netlsd", "gin"],
        help="Embedding methods to test"
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
    3. Generate visualizations
    4. Report results
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
    plot_heatmaps(df)
    plot_barplots(df)
    plot_scatter(df)
    
    # Final summary
    print("\n" + "="*70)
    print("TRANSFER ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  - Results: {OUT_DIR_TABLES}/transfer_results.csv")
    print(f"  - Figures: {OUT_DIR_FIGS}/")
    print(f"\nKey files:")
    print(f"  - *_transfer_heatmap_*.png  : Source → Target AUC matrices")
    print(f"  - transfer_barplot_all.png  : Method comparison")
    print(f"  - transfer_scatter.png      : Within vs Cross AUC")
    
    # Quick statistics
    print(f"\n" + "="*70)
    print("QUICK STATISTICS")
    print("="*70)
    
    # Average performance within vs cross dataset
    within = df[df["src"] == df["tgt"]]["auc"].mean()
    cross = df[df["src"] != df["tgt"]]["auc"].mean()
    gap = within - cross
    
    print(f"Average Within-Dataset AUC:  {within:.3f}")
    print(f"Average Cross-Dataset AUC:   {cross:.3f}")
    print(f"Average Transfer Gap:        {gap:.3f}")
    
    # Best transferring method
    avg_by_method = df.groupby("method")["auc"].mean().sort_values(ascending=False)
    print(f"\nBest Method (by avg AUC): {avg_by_method.index[0]} ({avg_by_method.values[0]:.3f})")
    
    print("\n✅ All analysis complete!")

if __name__ == "__main__":
    main()


# ============================================================================
# USAGE EXAMPLES & INTERPRETATION GUIDE
# ============================================================================

"""
BASIC USAGE:
-----------
# Quick test with defaults
python final_task4.py

# Custom configuration
python final_task4.py \
  --datasets MUTAG ENZYMES \
  --methods graph2vec netlsd \
  --dims 64 128

# More seeds for statistical confidence
python final_task4.py \
  --seeds 0 1 2


COMPLETE EVALUATION:
-------------------
python final_task4.py \
  --datasets MUTAG ENZYMES IMDB-MULTI \
  --methods graph2vec netlsd gin \
  --dims 32 64 128 \
  --seeds 0 1 2 3 4

"""