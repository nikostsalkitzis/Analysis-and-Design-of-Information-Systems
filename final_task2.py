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

# ---------------- Headless plotting + compat patches ----------------
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

import os, time, argparse, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from contextlib import contextmanager

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.manifold import TSNE
try:
    from umap import UMAP
except Exception:
    from umap.umap_ import UMAP

import networkx as nx
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool

from karateclub import Graph2Vec  # NetLSD replaced by robust local impl below


# ---------------- Paths ----------------
OUT_DIR_TABLES = "report/tables"
OUT_DIR_FIGS   = "report/figures"
os.makedirs(OUT_DIR_TABLES, exist_ok=True)
os.makedirs(OUT_DIR_FIGS,   exist_ok=True)
os.makedirs("report/logs",  exist_ok=True)


# ---------------- Utilities ----------------
def ds_labels(ds):
    return np.array([int(g.y) for g in ds])

def infer_num_node_features(ds):
    """Return node feature size; if missing, fall back to degree one-hot up to min(10, max degree)."""
    x0 = getattr(ds[0], "x", None)
    if x0 is not None and x0 is not False:
        return ds.num_node_features
    # Create synthetic degree feature later in collate; here just signal 1 (degree scalar)
    return 1

def attach_degree_as_feature(graph):
    """If a graph has no x, create a single degree feature column."""
    if getattr(graph, "x", None) is None:
        G = to_networkx(graph, to_undirected=True)
        deg = np.array([d for _, d in G.degree()], dtype=np.float32)
        graph.x = torch.from_numpy(deg).view(-1, 1)
    return graph

def to_nx_with_labels(ds_slice):
    """Convert PyG graphs to undirected NetworkX + categorical node labels (degree) for Graph2Vec."""
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
def embed_graph2vec(ds_slice, dim=128, seed=0, epochs=20, wl_iterations=2, min_count=5):
    Gs = to_nx_with_labels(ds_slice)
    with timed("Graph2Vec"):
        model = Graph2Vec(dimensions=dim, wl_iterations=wl_iterations, epochs=epochs,
                          seed=seed, workers=1, min_count=min_count)
        model.fit(Gs)
        X = model.get_embedding()
    return X

# Robust NetLSD (dense eigendecomposition; no ARPACK k<=0 issue)
def _netlsd_signature_dense(G, times):
    n = G.number_of_nodes()
    if n == 0:
        return np.zeros_like(times)
    L = nx.normalized_laplacian_matrix(G).astype(float).toarray()
    lam = np.linalg.eigvalsh(L)                      # symmetric -> stable
    return np.exp(-np.outer(times, lam)).sum(axis=1) # heat trace

def embed_netlsd(ds_slice, dim=128, pca_seed=0, n_times=256, t_min=1e-2, t_max=1e2):
    times = np.logspace(np.log10(t_min), np.log10(t_max), num=n_times)
    Gs = [to_networkx(g, to_undirected=True) for g in ds_slice]
    with timed("NetLSD"):
        sigs = [_netlsd_signature_dense(G, times) for G in Gs]
        X = np.vstack(sigs)  # [n_graphs, n_times]
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
                nn.Linear(h, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
            )
            self.layers.append(GINConv(mlp))
            h = hidden
        self.head = nn.Linear(hidden, n_classes)  # used only during training

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, batch)  # graph-level embedding
        logits = self.head(g)
        return logits, g

def train_gin_get_embeddings(ds_slice, dim=128, seed=0, hidden=64, layers=3, dropout=0.2,
                             epochs=30, batch_size=64, lr=1e-3, device="cpu"):
    """
    Train a small supervised GIN to learn an encoder, then return graph embeddings (penultimate layer).
    Embedding dimension returned is `hidden`. If `dim` != hidden, we apply PCA to `dim`.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Ensure each graph has features
    graphs = [attach_degree_as_feature(g.clone()) for g in ds_slice]
    num_classes = len(np.unique([int(g.y) for g in graphs]))
    in_dim = graphs[0].x.size(-1)

    model = GINEncoder(in_dim, hidden=hidden, layers=layers, dropout=dropout, n_classes=num_classes).to(device)
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

    if dim != X.shape[1]:
        X = PCA(n_components=dim, random_state=seed).fit_transform(X)
    return X


# ---------------- Clustering & metrics ----------------
def cluster_and_score(X, y, n_clusters, seed, algo="kmeans"):
    if algo == "kmeans":
        model = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
        pred = model.fit_predict(X)
    elif algo == "spectral":
        model = SpectralClustering(
            n_clusters=n_clusters,
            assign_labels="kmeans",
            affinity="rbf",       # robust on compact embeddings
            random_state=seed
        )
        pred = model.fit_predict(X)
    else:
        raise ValueError(algo)

    ari = adjusted_rand_score(y, pred)
    try:
        # silhouette requires at least 2 clusters present
        if len(np.unique(pred)) > 1:
            sil = silhouette_score(X, pred)
        else:
            sil = np.nan
    except Exception:
        sil = np.nan

    return {
        "ari": float(ari),
        "silhouette": float(np.nan if np.isnan(sil) else sil),
        "labels": pred
    }


# ---------------- Visualization ----------------
def scatter_2d(X2, y, title, outpath):
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.scatter(X2[:,0], X2[:,1], c=y, s=16)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

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


# ---------------- Runner ----------------
def run(datasets, methods, dims, seeds, plot_policy="first_seed",
        out_csv=f"{OUT_DIR_TABLES}/clustering_eval.csv",
        gin_hidden=64, gin_layers=3, gin_dropout=0.2, gin_epochs=30, gin_batch=64, device="cpu"):
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
                    # Embeddings
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
                        raise ValueError(method)

                    # Clustering
                    res_km = cluster_and_score(X, y, n_clusters, seed, algo="kmeans")
                    res_sp = cluster_and_score(X, y, n_clusters, seed, algo="spectral")

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
                    print("KMeans:", json.dumps(row_km))
                    print("Spectral:", json.dumps(row_sp))
                    rows.extend([row_km, row_sp])

                    # Plots
                    do_plots = (
                        plot_policy == "all" or
                        (plot_policy == "first_seed" and seed == first_seed)
                    )
                    if do_plots:
                        prefix = f"{OUT_DIR_FIGS}/{ds_name}_{method}_d{dim}"
                        title  = f"{ds_name} | {method} (d={dim})"
                        try:
                            plot_tsne_umap(X, y, title, prefix)
                        except Exception as e:
                            print(f"[warn] plot failed: {e}")

                    # Progressive write
                    pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    print(f"\nSaved per-run clustering results to {out_csv}")
    return df

def aggregate_and_rank(df,
                       out_csv_agg=f"{OUT_DIR_TABLES}/clustering_eval_agg.csv",
                       out_csv_top=f"{OUT_DIR_TABLES}/clustering_eval_top.csv"):
    if df.empty:
        print("No rows to aggregate.")
        return df, None

    agg = df.groupby(["dataset", "method", "dim", "algo"]).agg(
        ari_mean=("ari", "mean"),   ari_std=("ari", "std"),
        sil_mean=("silhouette", "mean"), sil_std=("silhouette", "std"),
        n_runs=("ari", "count"),
    ).reset_index()

    agg.to_csv(out_csv_agg, index=False)
    print(f"Saved aggregated results to {out_csv_agg}")

    # Best by ARI per dataset (tie-break by Silhouette)
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

