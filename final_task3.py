#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task (c): Stability Analysis

- Methods: graph2vec, netlsd (robust dense eigs), gin (hidden = dim)
- Datasets: MUTAG, ENZYMES, IMDB-MULTI (configurable)
- Perturbations:
    * edges: add/remove a % of edges
    * attrs: shuffle a % of node features within each graph
- Metrics:
    * Embedding stability: mean cosine similarity (↑=better), mean L2 drift (↓=better)
    * Δ metrics (perturbed − clean) for Acc / F1 / AUC using SVM and MLP
- Plots:
    * ΔAUC vs. perturbation level (edges & attrs) — shaded across seeds
    * Embedding drift vs. level — shaded across seeds
  Titles are placed via `suptitle` with extra top margin so they don’t get clipped.

Outputs:
  - report/tables/stability_results.csv
  - report/figures/*_delta_auc_edges.png, *_delta_auc_attrs.png
  - report/figures/*_embed_drift_edges.png, *_embed_drift_attrs.png
"""

# ---------- Headless plotting + compat patches ----------
import matplotlib
matplotlib.use("Agg")

# SciPy errstate patch (seen in some mixes)
import numpy as _np
import scipy as _sp
if not hasattr(_sp, "errstate"):
    _sp.errstate = _np.errstate

# UMAP↔sklearn check_array kw mismatch (older sklearn)
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
# --------------------------------------------------------

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

from karateclub import Graph2Vec  # NetLSD implemented locally below

# --------- Paths ---------
OUT_DIR_TABLES = "report/tables"
OUT_DIR_FIGS   = "report/figures"
os.makedirs(OUT_DIR_TABLES, exist_ok=True)
os.makedirs(OUT_DIR_FIGS,   exist_ok=True)

# --------- Small helpers ---------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def ds_labels(ds):
    return np.array([int(g.y) for g in ds])

def ensure_node_features(graphs: List[Data]) -> List[Data]:
    """If x is missing, use degree as one-dimensional feature."""
    out = []
    for g in graphs:
        if getattr(g, "x", None) is None:
            deg = torch.bincount(g.edge_index[0], minlength=g.num_nodes).float().view(-1, 1)
            g = Data(x=deg, edge_index=g.edge_index, y=g.y, num_nodes=g.num_nodes)
        out.append(g)
    return out

@contextmanager
def timed(name="block"):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"[{name}] {dt:.2f}s")

# --------- Perturbations ---------
def perturb_edges(g: Data, level: float, seed: int) -> Data:
    """Add/remove ~level fraction of edges uniformly at random (undirected approximation)."""
    set_seed(seed)
    n = g.num_nodes
    E = g.edge_index.t().tolist()
    # Make undirected simple set
    und = set(tuple(sorted(e)) for e in E if e[0] != e[1])
    m = len(und)
    if m == 0:
        return g.clone()

    k = max(1, int(level * m))

    # Remove
    rem = random.sample(list(und), min(k, m))
    for e in rem:
        if e in und: und.remove(e)

    # Add
    possible = set()
    # sample until we get k new edges or reach a cap attempt
    attempts = 0
    while len(possible) < k and attempts < 20 * k:
        u = random.randrange(n); v = random.randrange(n)
        if u == v: 
            attempts += 1; continue
        e = tuple(sorted((u, v)))
        if e not in und:
            possible.add(e)
        attempts += 1
    und.update(possible)

    # Build new edge_index (both directions)
    u, v = zip(*und) if und else ([], [])
    ei = torch.tensor([list(u)+list(v), list(v)+list(u)], dtype=torch.long)
    return Data(x=g.x.clone(), edge_index=ei, y=g.y, num_nodes=n)

def perturb_attrs(g: Data, level: float, seed: int) -> Data:
    """Shuffle a fraction of node features within-graph."""
    set_seed(seed)
    x = g.x.clone()
    n = x.size(0)
    k = max(1, int(level * n))
    idx = np.arange(n); np.random.shuffle(idx)
    take = idx[:k]
    # simple permutation of rows among selected nodes
    perm = take.copy(); np.random.shuffle(perm)
    x[take] = x[perm]
    return Data(x=x, edge_index=g.edge_index.clone(), y=g.y, num_nodes=g.num_nodes)

# --------- Embeddings ---------
def to_nx_with_labels(ds_slice):
    Gs = []
    for g in ds_slice:
        G = to_networkx(g, to_undirected=True)
        degs = dict(G.degree())
        for n in G.nodes:
            G.nodes[n]["label"] = int(degs[n])
        Gs.append(G)
    return Gs

def embed_graph2vec(graphs: List[Data], dim: int, seed: int):
    Gs = to_nx_with_labels(graphs)
    with timed("graph2vec"):
        model = Graph2Vec(dimensions=dim, wl_iterations=2, epochs=20,
                          seed=seed, workers=1, min_count=5)
        model.fit(Gs)
        X = model.get_embedding()
    return X

def _netlsd_signature_dense(G, times):
    n = G.number_of_nodes()
    if n == 0:
        return np.zeros_like(times)
    L = nx.normalized_laplacian_matrix(G).astype(float).toarray()
    lam = np.linalg.eigvalsh(L)
    return np.exp(-np.outer(times, lam)).sum(axis=1)

def embed_netlsd(graphs: List[Data], dim: int, seed: int):
    times = np.logspace(-2, 2, 256)
    Gs = [to_networkx(g, to_undirected=True) for g in graphs]
    with timed("netlsd"):
        sigs = [_netlsd_signature_dense(G, times) for G in Gs]
        X = np.vstack(sigs)
        if dim != X.shape[1]:
            X = PCA(n_components=dim, random_state=seed).fit_transform(X)
    return X

# --- GIN encoder (hidden = dim), supervised training then take penultimate embeddings ---
class GINSmall(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=3, n_classes=2, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.mlps = nn.ModuleList()
        self.convs = nn.ModuleList()

        h = hidden
        # first MLP/conv
        self.mlps.append(nn.Sequential(nn.Linear(in_dim, h), nn.ReLU(), nn.Linear(h, h)))
        self.convs.append(GINConv(self.mlps[0]))
        # rest
        for _ in range(layers - 1):
            mlp = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
            self.mlps.append(mlp)
            self.convs.append(GINConv(mlp))

        self.lin = nn.Linear(h, n_classes)

    def forward(self, x, edge_index, batch):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        g = global_mean_pool(h, batch)   # graph embedding (hidden dim)
        out = self.lin(g)
        return out, g

def train_gin_embed(graphs: List[Data], dim: int, seed: int, epochs=30, batch_size=64, lr=1e-3, layers=3, dropout=0.2):
    set_seed(seed)
    graphs = ensure_node_features(graphs)
    in_dim = graphs[0].x.size(1)
    n_classes = int(torch.stack([g.y for g in graphs]).max()) + 1

    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
    model = GINSmall(in_dim, hidden=dim, layers=layers, n_classes=n_classes, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    with timed(f"gin(h={dim})_train"):
        for _ in range(epochs):
            for batch in loader:
                opt.zero_grad()
                logits, _ = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(logits, batch.y)
                loss.backward()
                opt.step()

    # embeddings
    model.eval()
    X = []
    with torch.no_grad():
        for g in DataLoader(graphs, batch_size=batch_size, shuffle=False):
            logits, emb = model(g.x, g.edge_index, g.batch)
            X.append(emb)
    X = torch.cat(X, dim=0).cpu().numpy()
    return X

def get_embeddings(method: str, graphs: List[Data], dim: int, seed: int) -> np.ndarray:
    method = method.lower()
    if method == "graph2vec":
        return embed_graph2vec(graphs, dim, seed)
    if method == "netlsd":
        return embed_netlsd(graphs, dim, seed)
    if method == "gin":
        return train_gin_embed(graphs, dim, seed, epochs=30, layers=3)
    raise ValueError(method)

# --------- Metrics ---------
def auc_any(y_true, scores, classes):
    y_true = np.asarray(y_true); classes = np.asarray(classes)
    try:
        if len(classes) == 2:
            if scores.ndim == 1:
                return roc_auc_score(y_true, scores)
            else:
                pos = 1 if scores.shape[1] > 1 else 0
                return roc_auc_score(y_true, scores[:, pos])
        else:
            Y = label_binarize(y_true, classes=classes)
            return roc_auc_score(Y, scores, average="macro", multi_class="ovr")
    except Exception:
        return np.nan

def eval_clfs(X, y, seed):
    classes = np.unique(y)
    # SVM
    svm = make_pipeline(StandardScaler(with_mean=True), LinearSVC(dual=False, random_state=seed))
    svm.fit(X, y)
    yhat = svm.predict(X)
    s_score = svm.decision_function(X) if hasattr(svm, "decision_function") else None
    acc_svm = accuracy_score(y, yhat)
    f1_svm  = f1_score(y, yhat, average="macro")
    auc_svm = auc_any(y, s_score, classes)

    # MLP
    mlp = make_pipeline(
        StandardScaler(with_mean=True),
        MLPClassifier(hidden_layer_sizes=(128,), activation="relu", solver="adam",
                      alpha=1e-4, max_iter=800, tol=1e-4, random_state=seed)
    )
    mlp.fit(X, y)
    yhat2 = mlp.predict(X)
    s_score2 = mlp.predict_proba(X) if hasattr(mlp, "predict_proba") else None
    acc_mlp = accuracy_score(y, yhat2)
    f1_mlp  = f1_score(y, yhat2, average="macro")
    auc_mlp = auc_any(y, s_score2, classes)

    return dict(acc_svm=acc_svm, f1_svm=f1_svm, auc_svm=auc_svm,
                acc_mlp=acc_mlp, f1_mlp=f1_mlp, auc_mlp=auc_mlp)

def emb_stability(X_clean, X_pert):
    # row-wise cosine similarity
    cs = np.diag(cosine_similarity(X_clean, X_pert))
    l2 = np.linalg.norm(X_clean - X_pert, axis=1)
    return float(np.mean(cs)), float(np.mean(l2))

# --------- Plotting (titles fixed) ---------
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

# --------- Runner ---------
def run_one_dataset(ds_name, methods, dims, seeds, levels_edges, levels_attrs):
    print(f"\n=== Dataset: {ds_name} ===")
    ds = TUDataset(root="data", name=ds_name)
    graphs = ensure_node_features([ds[i] for i in range(len(ds))])
    y = ds_labels(ds)
    rows = []

    for seed in seeds:
        set_seed(seed)
        # clean embeddings/metrics per (method,dim)
        cache_clean = {}

        for method in methods:
            for dim in dims:
                X_clean = get_embeddings(method, graphs, dim, seed)
                metrics_clean = eval_clfs(X_clean, y, seed)
                cache_clean[(method, dim)] = (X_clean, metrics_clean)

                # Edge perturbations
                for lv in levels_edges:
                    pert = [perturb_edges(g, lv, seed+123) for g in graphs]
                    Xp = get_embeddings(method, pert, dim, seed)
                    metrics_p = eval_clfs(Xp, y, seed)
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

def aggregate_and_plot(df_all):
    if df_all.empty:
        print("No rows to plot.")
        return
    # Save raw
    out_csv = os.path.join(OUT_DIR_TABLES, "stability_results.csv")
    df_all.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")

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
    p = argparse.ArgumentParser(description="Task (c): Stability Analysis with plots")
    p.add_argument("--datasets", nargs="+", default=["MUTAG", "ENZYMES", "IMDB-MULTI"])
    p.add_argument("--methods",  nargs="+", default=["graph2vec", "netlsd", "gin"])
    p.add_argument("--dims",     nargs="+", type=int, default=[32, 64])
    p.add_argument("--seeds",    nargs="+", type=int, default=[0, 1])

    p.add_argument("--levels_edges", nargs="+", type=float, default=[0.5, 1.0],
                   help="Relative fraction of edges to add/remove.")
    p.add_argument("--levels_attrs", nargs="+", type=float, default=[0.5, 1.0],
                   help="Relative fraction of node features to shuffle.")

    return p.parse_args()

def main():
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
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

    df_all = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    aggregate_and_plot(df_all)

if __name__ == "__main__":
    main()

