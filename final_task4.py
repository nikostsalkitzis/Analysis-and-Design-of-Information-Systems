#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task (d): Cross-Dataset Transferability

Goal:
  Evaluate how well embeddings trained on one dataset (source)
  generalize to another (target).

Methods: Graph2Vec, NetLSD, GIN
Datasets: MUTAG, ENZYMES, IMDB-MULTI (or whatever you pass)

Metrics:
  - Accuracy, F1, AUC on target dataset
  - Δ metrics = (within on source) − (transfer to target)

Visualizations (updated per request):
  - Heatmap: source → target scores (keeps within + transfer)
  - Barplots: ONLY cross-dataset (src != tgt), grouped by src→tgt
  - Scatter: ONLY cross-dataset (src != tgt), showing within(src) vs transfer(src→tgt)

Outputs:
  - report/tables/transfer_results.csv
  - report/figures/...
"""

# ---------------- Headless plotting ----------------
import matplotlib
matplotlib.use("Agg")

import os, argparse, warnings, random
warnings.filterwarnings("ignore")

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

from karateclub import Graph2Vec

# --- Fix for SciPy errstate (NetLSD stability) ---
import scipy as sp
if not hasattr(sp, "errstate"):
    sp.errstate = np.errstate
# ---------------------------------------------------

# ---------------- Paths ----------------
OUT_DIR_TABLES = "report_transfer/tables"
OUT_DIR_FIGS   = "report_transfer/figures"
os.makedirs(OUT_DIR_TABLES, exist_ok=True)
os.makedirs(OUT_DIR_FIGS, exist_ok=True)


# ---------------- Helpers ----------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def ds_labels(ds):
    return np.array([int(g.y) for g in ds])

def ensure_node_features(graphs):
    out = []
    for g in graphs:
        if getattr(g, "x", None) is None:
            deg = torch.bincount(g.edge_index[0], minlength=g.num_nodes).float().view(-1, 1)
            g = Data(x=deg, edge_index=g.edge_index, y=g.y, num_nodes=g.num_nodes)
        out.append(g)
    return out

def auc_any(y_true, scores, classes):
    """
    Handles binary and multi-class AUC, returns np.nan if it can't compute.
    """
    try:
        if len(classes) == 2:
            # binary
            if scores is None:
                return np.nan
            if scores.ndim > 1:
                return roc_auc_score(y_true, scores[:, 1])
            return roc_auc_score(y_true, scores)
        else:
            # multiclass macro-ovr
            if scores is None:
                return np.nan
            Y = label_binarize(y_true, classes=classes)
            return roc_auc_score(Y, scores, average="macro", multi_class="ovr")
    except Exception:
        return np.nan


# ---------------- Embedding methods ----------------
def to_nx_with_labels(ds_slice):
    Gs = []
    for g in ds_slice:
        G = to_networkx(g, to_undirected=True)
        degs = dict(G.degree())
        for n in G.nodes:
            G.nodes[n]["label"] = int(degs[n])
        Gs.append(G)
    return Gs

def embed_graph2vec(graphs, dim=128, seed=0):
    Gs = to_nx_with_labels(graphs)
    model = Graph2Vec(
        dimensions=dim,
        wl_iterations=2,
        epochs=20,
        seed=seed,
        workers=1,
        min_count=5
    )
    model.fit(Gs)
    return model.get_embedding()

def _netlsd_signature_dense(G, times):
    n = G.number_of_nodes()
    if n == 0:
        return np.zeros_like(times)
    L = nx.normalized_laplacian_matrix(G).astype(float).toarray()
    lam = np.linalg.eigvalsh(L)
    return np.exp(-np.outer(times, lam)).sum(axis=1)

def embed_netlsd(graphs, dim=128, seed=0):
    times = np.logspace(-2, 2, 256)
    sigs = []
    for g in graphs:
        G = to_networkx(g, to_undirected=True)
        sigs.append(_netlsd_signature_dense(G, times))
    X = np.vstack(sigs)
    if dim != X.shape[1]:
        X = PCA(n_components=dim, random_state=seed).fit_transform(X)
    return X

class GINSmall(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=3, n_classes=2, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.mlps = nn.ModuleList()
        self.convs = nn.ModuleList()
        h = hidden
        self.mlps.append(nn.Sequential(nn.Linear(in_dim, h), nn.ReLU(), nn.Linear(h, h)))
        self.convs.append(GINConv(self.mlps[0]))
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
        g = global_mean_pool(h, batch)
        out = self.lin(g)
        return out, g

def train_gin_embed(graphs, dim=64, seed=0, epochs=30, batch_size=64, lr=1e-3):
    set_seed(seed)
    graphs = ensure_node_features(graphs)
    in_dim = graphs[0].x.size(1)
    n_classes = int(torch.stack([g.y for g in graphs]).max()) + 1
    model = GINSmall(in_dim, hidden=dim, n_classes=n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for batch in loader:
            opt.zero_grad()
            logits, _ = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            opt.step()

    model.eval()
    X = []
    with torch.no_grad():
        for g in DataLoader(graphs, batch_size=batch_size, shuffle=False):
            _, emb = model(g.x, g.edge_index, g.batch)
            X.append(emb)
    return torch.cat(X, dim=0).cpu().numpy()

def get_embeddings(method, graphs, dim, seed):
    if method == "graph2vec":
        return embed_graph2vec(graphs, dim, seed)
    elif method == "netlsd":
        return embed_netlsd(graphs, dim, seed)
    elif method == "gin":
        return train_gin_embed(graphs, dim, seed)
    else:
        raise ValueError(method)


# ---------------- Classifier & evaluation ----------------
def fit_clf(X, y, seed):
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
    clf.fit(X, y)
    return clf

def eval_within(X, y, seed):
    clf = fit_clf(X, y, seed)
    y_pred = clf.predict(X)
    y_score = clf.predict_proba(X) if hasattr(clf, "predict_proba") else None

    acc = accuracy_score(y, y_pred)
    f1  = f1_score(y, y_pred, average="macro")
    auc = auc_any(y, y_score, classes=np.unique(y))

    return clf, acc, f1, auc

def eval_transfer(clf, X_tgt, y_tgt):
    y_pred = clf.predict(X_tgt)
    y_score = clf.predict_proba(X_tgt) if hasattr(clf, "predict_proba") else None

    acc = accuracy_score(y_tgt, y_pred)
    f1  = f1_score(y_tgt, y_pred, average="macro")
    auc = auc_any(y_tgt, y_score, classes=np.unique(y_tgt))

    return acc, f1, auc


# ---------------- Runner ----------------
def run_transfer(datasets, methods, dims, seeds):
    results = []
    data_cache = {}

    # load datasets once
    for ds_name in datasets:
        ds = TUDataset(root="data", name=ds_name)
        data_cache[ds_name] = [ds[i] for i in range(len(ds))]

    # iterate
    for seed in seeds:
        for dim in dims:
            for method in methods:
                for src in datasets:
                    graphs_src = ensure_node_features(data_cache[src])
                    X_src = get_embeddings(method, graphs_src, dim, seed)
                    y_src = ds_labels(data_cache[src])

                    # train classifier on source embeddings (within-source training)
                    clf, acc_src, f1_src, auc_src = eval_within(X_src, y_src, seed)

                    for tgt in datasets:
                        graphs_tgt = ensure_node_features(data_cache[tgt])
                        X_tgt = get_embeddings(method, graphs_tgt, dim, seed)
                        y_tgt = ds_labels(data_cache[tgt])

                        acc, f1, auc = eval_transfer(clf, X_tgt, y_tgt)

                        res = dict(
                            src=src,
                            tgt=tgt,
                            method=method,
                            dim=dim,
                            seed=seed,

                            acc=acc,
                            f1=f1,
                            auc=auc,

                            acc_src=acc_src,
                            f1_src=f1_src,
                            auc_src=auc_src,

                            delta_acc=acc_src - acc,
                            delta_f1=f1_src - f1,
                            delta_auc=auc_src - auc,
                        )
                        results.append(res)

                        print(
                            f"{method.upper()} {src}->{tgt} dim={dim} seed={seed} | "
                            f"ACC={acc:.3f} (Δ={res['delta_acc']:+.3f})  "
                            f"F1={f1:.3f} (Δ={res['delta_f1']:+.3f})  "
                            f"AUC={auc:.3f} (Δ={res['delta_auc']:+.3f})"
                        )

    df = pd.DataFrame(results)
    out_csv = os.path.join(OUT_DIR_TABLES, "transfer_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved results to {out_csv}")
    return df


# ---------------- Visualization ----------------
def _pivot(df, metric, method, dim):
    sub = df[(df["method"] == method) & (df["dim"] == dim)]
    return sub.pivot_table(values=metric, index="src", columns="tgt", aggfunc="mean")

def plot_heatmaps(df):
    """
    Keeps all pairs, including src==tgt.
    """
    metrics = ["auc", "acc", "f1"]
    for method in df["method"].unique():
        for dim in sorted(df["dim"].unique()):
            for metric in metrics:
                pivot = _pivot(df, metric, method, dim)
                plt.figure(figsize=(6,5))
                sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".2f")
                plt.title(f"{method.upper()} — Transfer {metric.upper()} (dim={dim})")
                plt.tight_layout()
                plt.savefig(f"{OUT_DIR_FIGS}/{method}_transfer_heatmap_{metric}_d{dim}.png", dpi=150)
                plt.close()

def plot_barplots_cross(df):
    """
    Barplots for ONLY cross-dataset transfer (src != tgt).
    Hue is the explicit direction 'src→tgt'.
    One figure per metric.
    """
    df_cross = df[df["src"] != df["tgt"]].copy()
    df_cross["pair"] = df_cross["src"] + "→" + df_cross["tgt"]

    metrics = ["auc", "acc", "f1"]
    for metric in metrics:
        plt.figure(figsize=(10,6))
        sns.barplot(
            data=df_cross,
            x="method",
            y=metric,
            hue="pair",
            estimator=np.mean,
            errorbar=None
        )
        plt.ylabel(metric.upper())
        plt.xlabel("Embedding method")
        plt.title(f"Cross-Dataset {metric.upper()} by Transfer Direction (src→tgt, no within)")
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR_FIGS}/transfer_barplot_{metric}_cross_only.png", dpi=150)
        plt.close()

def plot_scatter_cross(df):
    """
    Scatter comparing:
      x = within-dataset metric on SRC (src→src),
      y = cross-dataset metric on TGT (src→tgt),
    but ONLY for src != tgt.

    We show separate figures for AUC / ACC / F1.
    Hue = method, style = transfer pair.
    """
    df_cross = df[df["src"] != df["tgt"]].copy()
    df_cross["pair"] = df_cross["src"] + "→" + df_cross["tgt"]

    # (within metric col, transfer metric col, tag/metric name)
    metric_pairs = [
        ("auc_src", "auc", "auc"),
        ("acc_src", "acc", "acc"),
        ("f1_src",  "f1",  "f1"),
    ]

    for xcol, ycol, tag in metric_pairs:
        plt.figure(figsize=(6,6))
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
        plt.savefig(f"{OUT_DIR_FIGS}/transfer_scatter_{tag}_cross_only.png", dpi=150)
        plt.close()


# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Task (d): Cross-Dataset Transferability")
    p.add_argument("--datasets", nargs="+", default=["MUTAG", "ENZYMES", "IMDB-MULTI"])
    p.add_argument("--methods",  nargs="+", default=["graph2vec", "netlsd", "gin"])
    p.add_argument("--dims",     nargs="+", type=int, default=[32, 64])
    p.add_argument("--seeds",    nargs="+", type=int, default=[0])
    return p.parse_args()


def main():
    args = parse_args()
    df = run_transfer(args.datasets, args.methods, args.dims, args.seeds)

    # Heatmaps (includes diagonal)
    plot_heatmaps(df)

    # Your requested views: ONLY true cross, src != tgt
    plot_barplots_cross(df)
    plot_scatter_cross(df)

    print("\n✅ All plots saved in report/figures/")
    print("✅ Transfer results saved in report/tables/transfer_results.csv")


if __name__ == "__main__":
    main()

