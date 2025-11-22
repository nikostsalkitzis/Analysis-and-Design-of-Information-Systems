#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task (a): Classification for unsupervised graph embeddings.

- Methods: Graph2Vec, NetLSD (robust dense-eigs implementation)
- Datasets: MUTAG, ENZYMES, IMDB-MULTI (configurable via CLI)
- Classifiers: Linear SVM, MLP
- Metrics: Accuracy, Macro-F1, ROC-AUC (binary & multiclass, macro OvR)
- Logs: Embedding generation time & memory; Classifier training time & memory
- Plots:
    * Accuracy vs. dimension (mean ± std across seeds)  <-- updated to show ribbon + error bars
    * Generation time vs. dimension
    * t-SNE & UMAP scatter plots of embeddings per (dataset, method, dim)

Outputs:
  - CSV:  report/tables/classification_eval.csv          (per run / per seed)
  - CSV:  report/tables/classification_eval_agg.csv      (mean/std over seeds)
  - PNGs: report/figures/
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

import os, time, argparse, json, tracemalloc, warnings
from contextlib import contextmanager
warnings.filterwarnings("ignore")  # Suppress scientific library warnings

import numpy as np
import pandas as pd
import psutil  # For memory profiling
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from sklearn.manifold import TSNE
try:
    from umap import UMAP
except Exception:
    from umap.umap_ import UMAP

import networkx as nx
from torch_geometric.datasets import TUDataset  # Graph datasets
from torch_geometric.utils import to_networkx

from karateclub import Graph2Vec  # Unsupervised graph embedding

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================

OUT_DIR_TABLES = "report/tables"
OUT_DIR_FIGS   = "report/figures"
os.makedirs(OUT_DIR_TABLES, exist_ok=True)
os.makedirs(OUT_DIR_FIGS, exist_ok=True)
os.makedirs("report/logs", exist_ok=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ds_labels(ds):
    """Extract integer labels from PyTorch Geometric dataset."""
    return np.array([int(g.y) for g in ds])

def to_nx_with_labels(ds_slice):
    """
    Convert PyG graphs to NetworkX format and add node labels.
    
    Graph2Vec requires discrete node labels. Here we use node degree.
    
    Args:
        ds_slice: List of PyG graph objects
        
    Returns:
        List of NetworkX graphs with 'label' attribute on each node
    """
    nx_graphs = []
    for g in ds_slice:
        G = to_networkx(g, to_undirected=True)  # Convert to NetworkX
        degs = dict(G.degree())  # Get degree of each node
        for n in G.nodes():
            G.nodes[n]['label'] = int(degs[n])  # Assign degree as discrete label
        nx_graphs.append(G)
    return nx_graphs

@contextmanager
def timed_mem(name="block"):
    """
    Context manager to measure execution time and memory usage.
    
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
    rss_before = proc.memory_info().rss  # Memory before execution
    tracemalloc.start()  # Start Python memory tracing
    t0 = time.perf_counter()
    meta = {}
    try:
        yield meta
    finally:
        elapsed = time.perf_counter() - t0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_after = proc.memory_info().rss  # Memory after execution
        
        # Store metrics in the meta dict
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
            # Binary classification
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
            # Multiclass: use One-vs-Rest (OvR) with macro averaging
            if scores is None:
                return np.nan
            scores = np.asarray(scores)
            Y = label_binarize(y_true, classes=classes)  # Binarize labels
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
    # Train with timing & memory tracking
    with timed_mem(f"train_{tag}") as meta:
        clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)

    # Get continuous scores for AUC calculation
    scores = None
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(X_test)
    elif hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(X_test)

    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")  # Macro: average per class
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
# EMBEDDING METHODS
# ============================================================================

def embed_graph2vec(ds_slice, dim=128, seed=0, epochs=20, wl_iterations=2, min_count=5):
    """
    Generate Graph2Vec embeddings.
    
    Graph2Vec learns graph-level embeddings by treating graphs as "documents"
    and their substructures (from Weisfeiler-Lehman) as "words", then applies
    doc2vec-style training.
    
    Args:
        ds_slice: List of PyG graphs
        dim: Embedding dimension
        seed: Random seed
        epochs: Training epochs for doc2vec
        wl_iterations: Depth of WL kernel
        min_count: Minimum frequency for substructures
        
    Returns:
        X: Embedding matrix [num_graphs, dim]
        info: dict with generation time & memory
    """
    Gs = to_nx_with_labels(ds_slice)
    with timed_mem("embed_graph2vec") as meta:
        model = Graph2Vec(
            dimensions=dim,
            wl_iterations=wl_iterations,  # WL kernel depth
            epochs=epochs,
            seed=seed,
            workers=1,
            min_count=min_count
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
    """
    Compute NetLSD heat trace signature using dense eigendecomposition.
    
    NetLSD (Network Laplacian Spectral Descriptor) characterizes a graph
    through the heat trace of its normalized Laplacian at different time scales.
    
    Heat trace at time t: h(t) = sum_i exp(-t * λ_i)
    where λ_i are the eigenvalues of the normalized Laplacian.
    
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
    lam = np.linalg.eigvalsh(L)  # Eigenvalues (always real for symmetric L)

    # Heat trace: h(t) = sum_i exp(-t * lambda_i)
    H = np.exp(-np.outer(times, lam)).sum(axis=1)
    return H

def embed_netlsd(ds_slice, dim=128, pca_seed=0, n_times=256, t_min=1e-2, t_max=1e+2):
    """
    Generate NetLSD embeddings with PCA compression.
    
    Process:
    1. Compute heat trace signatures at logarithmically-spaced times
    2. Stack into matrix [num_graphs, n_times]
    3. Apply PCA to reduce to desired dimension
    
    Args:
        ds_slice: List of PyG graphs
        dim: Target embedding dimension
        pca_seed: Random seed for PCA
        n_times: Number of time points for heat trace
        t_min, t_max: Min/max diffusion times
        
    Returns:
        X: Embedding matrix [num_graphs, dim]
        info: dict with generation time & memory
    """
    times = np.logspace(np.log10(t_min), np.log10(t_max), num=n_times, base=10.0)
    Gs = [to_networkx(g, to_undirected=True) for g in ds_slice]

    with timed_mem("embed_netlsd") as meta:
        # Compute heat trace signatures for all graphs
        sigs = []
        for G in Gs:
            s = _netlsd_signature_dense(G, times)
            sigs.append(s)
        X = np.vstack(sigs)  # [num_graphs, n_times]

        # PCA compression if needed
        if dim != X.shape[1]:
            X = PCA(n_components=dim, random_state=pca_seed).fit_transform(X)

    info = dict(
        gen_time_sec=meta["time_sec"],
        gen_rss_before_mb=meta["rss_before_mb"],
        gen_rss_after_mb=meta["rss_after_mb"],
        gen_py_peak_mb=meta["py_peak_mb"],
    )
    return X, info

# ============================================================================
# VISUALIZATION: t-SNE & UMAP
# ============================================================================

def scatter_2d(X2, y, title, outpath):
    """Create a 2D scatter plot colored by labels."""
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.scatter(X2[:,0], X2[:,1], c=y, s=18)
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_embeddings_2d(emb, labels, dataset, method, dim, seed, which=("tsne","umap")):
    """
    Generate t-SNE and UMAP 2D visualizations of embeddings.
    
    These plots help visualize cluster structure and class separation.
    
    Args:
        emb: Embedding matrix [num_graphs, dim]
        labels: Ground truth labels
        dataset, method, dim, seed: Metadata for naming
        which: Tuple of visualization methods to use
    """
    if "tsne" in which:
        # t-SNE: nonlinear dimensionality reduction preserving local structure
        tsne = TSNE(
            n_components=2,
            random_state=seed,
            init="pca",  # Initialize with PCA
            perplexity=min(30, max(5, len(emb)//10))  # Adaptive perplexity
        )
        Xt = tsne.fit_transform(emb)
        scatter_2d(
            Xt, labels,
            f"{dataset} — {method} d={dim} — t-SNE",
            os.path.join(OUT_DIR_FIGS, f"{dataset}_{method}_d{dim}_seed{seed}_tsne.png")
        )
    
    if "umap" in which:
        # UMAP: preserves both local and global structure better than t-SNE
        nn = min(15, max(2, len(emb)-1))  # Adaptive n_neighbors
        um = UMAP(n_components=2, random_state=seed, n_neighbors=nn, min_dist=0.1)
        Xu = um.fit_transform(emb)
        scatter_2d(
            Xu, labels,
            f"{dataset} — {method} d={dim} — UMAP",
            os.path.join(OUT_DIR_FIGS, f"{dataset}_{method}_d{dim}_seed{seed}_umap.png")
        )

# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def run_experiment(datasets, methods, dims, seeds, test_size=0.2,
                   out_csv="report/tables/classification_eval.csv",
                   embed_plots="first_seed"):
    """
    Run the full classification experiment.
    
    For each combination of (dataset, method, dimension, seed):
    1. Generate embeddings for all graphs
    2. Split into train/test
    3. Train SVM and MLP classifiers
    4. Compute metrics and resource usage
    5. Optionally generate t-SNE/UMAP plots
    6. Save results progressively to CSV
    
    Args:
        datasets: List of dataset names (e.g., ["MUTAG", "ENZYMES"])
        methods: List of embedding methods (e.g., ["graph2vec", "netlsd"])
        dims: List of embedding dimensions (e.g., [32, 64, 128])
        seeds: List of random seeds for reproducibility
        test_size: Fraction for test split
        out_csv: Path to output CSV
        embed_plots: 'none', 'first_seed', or 'all' - controls visualization frequency
        
    Returns:
        DataFrame with all results
    """
    rows = []
    
    for ds_name in datasets:
        print(f"\n=== Dataset: {ds_name} ===")
        
        # Load dataset
        ds = TUDataset(root="data", name=ds_name)
        y_all = ds_labels(ds)
        classes = np.unique(y_all)
        idx_all = np.arange(len(ds))
        ds_all = [ds[i] for i in idx_all]

        first_seed = min(seeds) if len(seeds) else 0

        for seed in seeds:
            # Train/test split with stratification
            tr_idx, te_idx = train_test_split(
                idx_all,
                test_size=test_size,
                random_state=seed,
                stratify=y_all  # Maintain class proportions
            )
            
            for method in methods:
                for dim in dims:
                    print(f"\n--- {method} | dim={dim} | seed={seed} ---")

                    # Generate embeddings for all graphs
                    if method.lower() == "graph2vec":
                        X_all, gen_info = embed_graph2vec(ds_all, dim=dim, seed=seed, epochs=20)
                    elif method.lower() == "netlsd":
                        X_all, gen_info = embed_netlsd(ds_all, dim=dim, pca_seed=seed)
                    else:
                        raise ValueError(f"Unknown method: {method}")

                    # Generate embedding visualizations (controlled by embed_plots)
                    if embed_plots == "all" or (embed_plots == "first_seed" and seed == first_seed):
                        try:
                            plot_embeddings_2d(X_all, y_all, ds_name, method, dim, seed)
                        except Exception as e:
                            print(f"[warn] embedding plots failed: {e}")

                    # Split embeddings into train/test
                    X_train, X_test = X_all[tr_idx], X_all[te_idx]
                    y_train, y_test = y_all[tr_idx], y_all[te_idx]

                    # CLASSIFIER 1: Linear SVM with scaling
                    svm = make_pipeline(
                        StandardScaler(with_mean=True),  # Standardize features
                        LinearSVC(dual=False, random_state=seed)  # Linear SVM
                    )
                    svm_metrics, svm_time, _ = eval_classifier(
                        svm, X_train, X_test, y_train, y_test, classes, tag="svm"
                    )

                    # CLASSIFIER 2: MLP with one hidden layer
                    mlp = make_pipeline(
                        StandardScaler(with_mean=True),
                        MLPClassifier(
                            hidden_layer_sizes=(128,),  # One hidden layer with 128 units
                            activation="relu",
                            solver="adam",
                            alpha=1e-4,  # L2 regularization
                            max_iter=800,
                            tol=1e-4,
                            random_state=seed
                        )
                    )
                    mlp_metrics, mlp_time, _ = eval_classifier(
                        mlp, X_train, X_test, y_train, y_test, classes, tag="mlp"
                    )

                    # Collect all results into one row
                    row = dict(
                        dataset=ds_name,
                        method=method,
                        dim=dim,
                        seed=seed,
                        n_graphs=len(ds),
                        n_classes=len(classes),
                        # Embedding generation stats
                        gen_time_sec=gen_info["gen_time_sec"],
                        gen_rss_before_mb=gen_info["gen_rss_before_mb"],
                        gen_rss_after_mb=gen_info["gen_rss_after_mb"],
                        gen_py_peak_mb=gen_info["gen_py_peak_mb"],
                        # SVM results
                        acc_svm=svm_metrics["acc"],
                        f1_svm=svm_metrics["f1"],
                        auc_svm=svm_metrics["auc"],
                        train_time_svm_sec=svm_time["train_time_sec"],
                        train_rss_before_svm_mb=svm_time["train_rss_before_mb"],
                        train_rss_after_svm_mb=svm_time["train_rss_after_mb"],
                        train_py_peak_svm_mb=svm_time["train_py_peak_mb"],
                        # MLP results
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

                    # Progressive save to CSV (so we don't lose results if crash)
                    pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    print(f"\nSaved per-run results to {out_csv}")
    return df

# ============================================================================
# AGGREGATION & PLOTTING
# ============================================================================

def aggregate_and_plot(df, out_csv_agg="report/tables/classification_eval_agg.csv"):
    """
    Aggregate results across random seeds and generate summary plots.
    
    Creates:
    1. Aggregated CSV with mean ± std per (dataset, method, dim, classifier)
    2. Accuracy vs dimension plots with error bars and shaded regions
    3. Generation time vs dimension plots
    
    Args:
        df: DataFrame from run_experiment()
        out_csv_agg: Path for aggregated results CSV
    """
    if df.empty:
        print("No rows to aggregate.")
        return

    # Convert wide format (svm/mlp columns) to long format (one row per classifier)
    recs = []
    for _, r in df.iterrows():
        for clf in ["svm", "mlp"]:
            recs.append(dict(
                dataset=r["dataset"],
                method=r["method"],
                dim=int(r["dim"]),
                seed=int(r["seed"]),
                clf=clf,
                acc=r[f"acc_{clf}"],
                f1=r[f"f1_{clf}"],
                auc=r[f"auc_{clf}"],
                gen_time_sec=r["gen_time_sec"],
            ))
    long = pd.DataFrame(recs)

    # Aggregate: compute mean and std across seeds
    agg = long.groupby(["dataset", "method", "dim", "clf"]).agg(
        acc_mean=("acc", "mean"),
        acc_std=("acc", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
        gen_time_mean=("gen_time_sec", "mean"),
        gen_time_std=("gen_time_sec", "std"),
        n_runs=("acc", "count"),
    ).reset_index()

    agg.to_csv(out_csv_agg, index=False)
    print(f"Saved aggregated results to {out_csv_agg}")

    # Generate plots for each dataset and method
    for dataset in agg["dataset"].unique():
        subD = agg[agg["dataset"] == dataset]
        for method in subD["method"].unique():
            subDM = subD[subD["method"] == method].sort_values("dim")

            # PLOT 1: Accuracy vs Dimension (with shaded std regions + error bars)
            fig = plt.figure(figsize=(6,4))
            ax = fig.add_subplot(111)
            
            for clf in ["svm", "mlp"]:
                sdf = subDM[subDM["clf"] == clf]
                if sdf.empty:
                    continue
                x = sdf["dim"].values
                y = sdf["acc_mean"].values
                s = sdf["acc_std"].fillna(0).values  # std (0 if single seed)

                # Shaded region: mean ± std
                ax.fill_between(x, y - s, y + s, alpha=0.20, linewidth=0)

                # Line with error bars
                ax.errorbar(x, y, yerr=s, fmt='-o', capsize=4, label=f"{clf.upper()} acc")

            n_runs = int(subDM["n_runs"].max()) if "n_runs" in subDM else None
            subtitle = f"{dataset} — Accuracy vs. Embedding Dim ({method})"
            if n_runs:
                subtitle += f"  (n_seeds={n_runs})"
            ax.set_title(subtitle)
            ax.set_xlabel("Embedding dimension")
            ax.set_ylabel("Accuracy (mean ± std)")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend()
            fig.tight_layout()
            fig.savefig(f"{OUT_DIR_FIGS}/{dataset}_{method}_acc_vs_dim.png", dpi=150)
            plt.close(fig)

            # PLOT 2: Generation Time vs Dimension
            sdf = subDM[subDM["clf"] == "svm"].sort_values("dim")  # Time is same for both clfs
            fig2 = plt.figure(figsize=(6,4))
            ax2 = fig2.add_subplot(111)
            ax2.plot(sdf["dim"], sdf["gen_time_mean"], marker="o")
            ax2.set_title(f"{dataset} — Generation Time vs. Dim ({method})")
            ax2.set_xlabel("Embedding dimension")
            ax2.set_ylabel("Generation time (s)")
            ax2.grid(True, linestyle="--", alpha=0.4)
            fig2.tight_layout()
            fig2.savefig(f"{OUT_DIR_FIGS}/{dataset}_{method}_gentime_vs_dim.png", dpi=150)
            plt.close(fig2)

# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Task (a): Unsupervised embeddings → classification eval, with plots"
    )
    p.add_argument("--datasets", nargs="+", default=["MUTAG", "ENZYMES", "IMDB-MULTI"])
    p.add_argument("--methods",  nargs="+", default=["graph2vec", "netlsd"])
    p.add_argument("--dims",     nargs="+", type=int, default=[32, 64, 128])
    p.add_argument("--seeds",    nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--out", type=str, default="report/tables/classification_eval.csv")
    p.add_argument(
        "--embed_plots",
        choices=["none","first_seed","all"],
        default="first_seed",
        help="Save t-SNE/UMAP plots for embeddings."
    )
    return p.parse_args()

def main():
    """Main entry point."""
    os.environ.setdefault("PYTHONNOUSERSITE", "1")  # Avoid user site-packages
    args = parse_args()

    # Run experiments
    df = run_experiment(
        datasets=args.datasets,
        methods=args.methods,
        dims=args.dims,
        seeds=args.seeds,
        test_size=args.test_size,
        out_csv=args.out,
        embed_plots=args.embed_plots,
    )
    
    # Aggregate and plot
    aggregate_and_plot(df)

if __name__ == "__main__":
    main()