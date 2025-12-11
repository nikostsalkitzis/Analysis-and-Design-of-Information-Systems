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

import os, time, argparse, json, warnings, inspect
warnings.filterwarnings("ignore")

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

# After patches, we can import the rest
import networkx as nx
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool

from karateclub import Graph2Vec  # NetLSD implemented below


# ---------------- Paths ----------------
OUT_DIR_TABLES = "report/tables"
OUT_DIR_FIGS   = "report/figures"
OUT_DIR_LOGS   = "report/logs"
os.makedirs(OUT_DIR_TABLES, exist_ok=True)
os.makedirs(OUT_DIR_FIGS,   exist_ok=True)
os.makedirs(OUT_DIR_LOGS,   exist_ok=True)


# ---------------- Utilities ----------------
def ds_labels(ds):
    return np.array([int(g.y) for g in ds])

def attach_degree_as_feature(graph):
    """
    If graph.x is missing or None, create a single feature column = node degree.
    Needed for GIN on datasets without node attributes (e.g. IMDB-MULTI).
    """
    if getattr(graph, "x", None) is None:
        G = to_networkx(graph, to_undirected=True)
        deg = np.array([d for _, d in G.degree()], dtype=np.float32)
        graph.x = torch.from_numpy(deg).view(-1, 1)
    return graph

def to_nx_with_labels(ds_slice):
    """
    Convert PyG graphs to undirected NetworkX graphs and assign each node
    a discrete 'label' feature (here: its degree) for Graph2Vec.
    """
    Gs = []
    for g in ds_slice:
        G = to_networkx(g, to_undirected=True)
        degs = dict(G.degree())
        for n in G.nodes():
            G.nodes[n]["label"] = int(degs[n])
        Gs.append(G)
    return Gs

@contextmanager
def timed(name="block"):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(f"[{name}] time={time.perf_counter()-t0:.2f}s")


# ---------------- Embeddings: Graph2Vec & NetLSD ----------------
def embed_graph2vec(ds_slice, dim=128, seed=0, epochs=20,
                    wl_iterations=2, min_count=5):
    Gs = to_nx_with_labels(ds_slice)
    with timed("Graph2Vec"):
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
    return X

def _netlsd_signature_dense(G, times):
    """
    Compute NetLSD-style heat trace signature using the full eigendecomposition
    of the normalized Laplacian. Robust for small graphs.
    """
    n = G.number_of_nodes()
    if n == 0:
        return np.zeros_like(times)
    L = nx.normalized_laplacian_matrix(G).astype(float).toarray()
    lam = np.linalg.eigvalsh(L)  # Laplacian symmetric => stable eigenvalues
    # heat trace at each diffusion time t: sum(exp(-t * lambda_i))
    return np.exp(-np.outer(times, lam)).sum(axis=1)

def embed_netlsd(ds_slice, dim=128, pca_seed=0,
                 n_times=256, t_min=1e-2, t_max=1e2):
    times = np.logspace(np.log10(t_min), np.log10(t_max), num=n_times)
    Gs = [to_networkx(g, to_undirected=True) for g in ds_slice]
    with timed("NetLSD"):
        sigs = [_netlsd_signature_dense(G, times) for G in Gs]
        X = np.vstack(sigs)  # [num_graphs, n_times]
        # optional PCA compression to match requested embedding dim
        if dim != X.shape[1]:
            X = PCA(n_components=dim, random_state=pca_seed).fit_transform(X)
    return X


# ---------------- Embeddings: GIN ----------------
class GINEncoder(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=3, dropout=0.2, n_classes=2):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        h = in_dim
        for _ in range(layers):
            mlp = nn.Sequential(
                nn.Linear(h, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.layers.append(GINConv(mlp))
            h = hidden
        # supervised head for training
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.layers:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, batch)  # graph-level embedding
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
    Train GIN on the known labels (supervised), then export the pooled
    graph embedding from the penultimate layer. If hidden != dim, apply PCA.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    graphs = [attach_degree_as_feature(g.clone()) for g in ds_slice]
    num_classes = len(np.unique([int(g.y) for g in graphs]))
    in_dim = graphs[0].x.size(-1)

    model = GINEncoder(
        in_dim,
        hidden=hidden,
        layers=layers,
        dropout=dropout,
        n_classes=num_classes,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)

    with timed("GIN-train"):
        model.train()
        for _ in range(epochs):
            for batch in loader:
                batch = batch.to(device)
                logits, _ = model(batch)
                loss = F.cross_entropy(logits, batch.y.view(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()

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

    if dim != X.shape[1]:
        X = PCA(n_components=dim, random_state=seed).fit_transform(X)

    return X


# ---------------- Clustering & metrics ----------------
def preprocess_for_clustering(X):
    """
    Standardize features before distance-based clustering/metrics.
    This makes different embeddings and dimensions more comparable.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def cluster_and_score(X, y, n_clusters, seed, algo="kmeans"):
    """
    Run clustering on standardized X.
    Returns ARI, silhouette, cluster assignments, and the standardized X.
    """
    X_proc = preprocess_for_clustering(X)

    if algo == "kmeans":
        model = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
        pred = model.fit_predict(X_proc)
    elif algo == "spectral":
        model = SpectralClustering(
            n_clusters=n_clusters,
            assign_labels="kmeans",
            affinity="rbf",
            random_state=seed,
        )
        pred = model.fit_predict(X_proc)
    else:
        raise ValueError(algo)

    ari = adjusted_rand_score(y, pred)

    sil = np.nan
    try:
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


# ---------------- Visualization (t-SNE + UMAP) ----------------
def scatter_2d(X2, labels, title, outpath):
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
        ds = TUDataset(root="data", name=ds_name)
        y = ds_labels(ds)
        graphs = [ds[i] for i in range(len(ds))]
        n_clusters = len(np.unique(y))
        first_seed = min(seeds) if len(seeds) else 0

        for seed in seeds:
            for method in methods:
                for dim in dims:
                    print(f"\n--- {method} | dim={dim} | seed={seed} ---")

                    # 1) Compute embeddings
                    if method.lower() == "graph2vec":
                        X = embed_graph2vec(
                            graphs,
                            dim=dim,
                            seed=seed,
                            epochs=20,
                            wl_iterations=2,
                            min_count=5,
                        )
                    elif method.lower() == "netlsd":
                        X = embed_netlsd(
                            graphs,
                            dim=dim,
                            pca_seed=seed,
                            n_times=256,
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
                        raise ValueError(method)

                    # 2) Clustering & metrics
                    res_km = cluster_and_score(X, y, n_clusters, seed, algo="kmeans")
                    res_sp = cluster_and_score(X, y, n_clusters, seed, algo="spectral")

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

                    print("KMeans:", json.dumps(row_km))
                    print("Spectral:", json.dumps(row_sp))

                    rows.extend([row_km, row_sp])

                    # 3) Plots (t-SNE + UMAP) for first_seed or for all
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

                    # Progressive write so partial results are not lost
                    pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    print(f"\nSaved per-run clustering results to {out_csv}")
    return df


def aggregate_and_rank(
    df,
    out_csv_agg=f"{OUT_DIR_TABLES}/clustering_eval_agg.csv",
    out_csv_top=f"{OUT_DIR_TABLES}/clustering_eval_top.csv",
):
    """
    Aggregate across seeds. If only one seed, std is NaN; we fill with 0.0 so it
    doesn't look scary in tables.
    """
    if df.empty:
        print("No rows to aggregate.")
        return df, None

    agg = (
        df.groupby(["dataset", "method", "dim", "algo"])
        .agg(
            ari_mean=("ari", "mean"),
            ari_std=("ari", "std"),
            sil_mean=("silhouette", "mean"),
            sil_std=("silhouette", "std"),
            n_runs=("ari", "count"),
        )
        .reset_index()
    )

    agg["ari_std"] = agg["ari_std"].fillna(0.0)
    agg["sil_std"] = agg["sil_std"].fillna(0.0)

    agg.to_csv(out_csv_agg, index=False)
    print(f"Saved aggregated results to {out_csv_agg}")

    # Best per dataset by ARI, break ties using silhouette
    tops = []
    for ds in agg["dataset"].unique():
        sub = agg[agg["dataset"] == ds]
        best = sub.sort_values(["ari_mean", "sil_mean"], ascending=False).head(1)
        tops.append(best)
        print(f"\nBest separation for {ds}:")
        print(best.to_string(index=False))

    top_df = pd.concat(tops, ignore_index=True)
    top_df.to_csv(out_csv_top, index=False)
    print(f"Saved top-by-ARI table to {out_csv_top}")
    return agg, top_df


# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Task (b): Clustering of graph embeddings (Graph2Vec, NetLSD, GIN)"
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["MUTAG", "ENZYMES", "IMDB-MULTI"],
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=["graph2vec", "netlsd", "gin"],
    )
    p.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=[32, 64, 128],
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
    )
    p.add_argument(
        "--plot_policy",
        choices=["none", "first_seed", "all"],
        default="first_seed",
        help="Save t-SNE/UMAP plots per (dataset, method, dim).",
    )
    # GIN hyperparams
    p.add_argument("--gin_hidden", type=int, default=64)
    p.add_argument("--gin_layers", type=int, default=3)
    p.add_argument("--gin_dropout", type=float, default=0.2)
    p.add_argument("--gin_epochs", type=int, default=30)
    p.add_argument("--gin_batch", type=int, default=64)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main():
    # keep environment predictable
    os.environ.setdefault("PYTHONNOUSERSITE", "1")

    args = parse_args()

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

    aggregate_and_rank(df)


if __name__ == "__main__":
    main()