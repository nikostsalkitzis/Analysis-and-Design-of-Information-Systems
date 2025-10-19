#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task (d): Cross-Dataset Transferability

Goal:
  Evaluate how well embeddings trained on one dataset (source)
  generalize to another (target).

Methods: Graph2Vec, NetLSD, GIN
Datasets: MUTAG, ENZYMES, IMDB-MULTI

Metrics:
  - Accuracy, F1, AUC on target dataset
  - ΔAUC_transfer = AUC_within − AUC_cross

Visualizations:
  - Heatmap of transfer AUCs (source → target)
  - Barplots per method
  - Scatter plots of within vs. transfer AUC

Outputs:
  - report/tables/transfer_results.csv
  - report/figures/transfer_heatmap_*.png
  - report/figures/transfer_barplot_*.png
  - report/figures/transfer_scatter.png
"""

# ---------------- Headless plotting ----------------
import matplotlib
matplotlib.use("Agg")

import os, time, argparse, warnings, random
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
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from karateclub import Graph2Vec

# --- Fix for SciPy errstate (NetLSD stability) ---
import scipy as sp
if not hasattr(sp, "errstate"):
    sp.errstate = np.errstate
# ---------------------------------------------------

# ---------------- Paths ----------------
OUT_DIR_TABLES = "report/tables"
OUT_DIR_FIGS   = "report/figures"
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
    try:
        if len(classes) == 2:
            return roc_auc_score(y_true, scores[:, 1] if scores.ndim > 1 else scores)
        else:
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
    model = Graph2Vec(dimensions=dim, wl_iterations=2, epochs=20, seed=seed, workers=1, min_count=5)
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


# ---------------- Runner ----------------
def run_transfer(datasets, methods, dims, seeds):
    results = []
    data_cache = {}

    for ds_name in datasets:
        ds = TUDataset(root="data", name=ds_name)
        data_cache[ds_name] = [ds[i] for i in range(len(ds))]

    for seed in seeds:
        for dim in dims:
            for method in methods:
                for src in datasets:
                    graphs_src = ensure_node_features(data_cache[src])
                    X_src = get_embeddings(method, graphs_src, dim, seed)
                    y_src = ds_labels(data_cache[src])

                    # Train classifier on source
                    clf = make_pipeline(StandardScaler(with_mean=True),
                                        MLPClassifier(hidden_layer_sizes=(128,),
                                                      activation="relu", solver="adam",
                                                      max_iter=500, random_state=seed))
                    clf.fit(X_src, y_src)
                    y_pred_src = clf.predict(X_src)
                    y_score_src = clf.predict_proba(X_src) if hasattr(clf, "predict_proba") else None
                    auc_src = auc_any(y_src, y_score_src, np.unique(y_src))

                    for tgt in datasets:
                        graphs_tgt = ensure_node_features(data_cache[tgt])
                        X_tgt = get_embeddings(method, graphs_tgt, dim, seed)
                        y_tgt = ds_labels(data_cache[tgt])

                        y_pred_tgt = clf.predict(X_tgt)
                        y_score_tgt = clf.predict_proba(X_tgt) if hasattr(clf, "predict_proba") else None

                        acc = accuracy_score(y_tgt, y_pred_tgt)
                        f1  = f1_score(y_tgt, y_pred_tgt, average="macro")
                        auc = auc_any(y_tgt, y_score_tgt, np.unique(y_tgt))
                        delta_auc = auc_src - auc

                        results.append(dict(
                            src=src, tgt=tgt, method=method, dim=dim, seed=seed,
                            acc=acc, f1=f1, auc=auc, auc_src=auc_src, delta_auc=delta_auc
                        ))

                        print(f"{method} {src}->{tgt} dim={dim} seed={seed} AUC={auc:.3f}")

    df = pd.DataFrame(results)
    out_csv = os.path.join(OUT_DIR_TABLES, "transfer_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved results to {out_csv}")
    return df


# ---------------- Visualization ----------------
def plot_heatmaps(df):
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
    plt.figure(figsize=(8,6))
    sns.barplot(data=df, x="method", y="auc", hue="tgt")
    plt.title("Cross-Dataset AUC (All methods)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR_FIGS}/transfer_barplot_all.png", dpi=150)
    plt.close()

def plot_scatter(df):
    plt.figure(figsize=(6,6))
    sns.scatterplot(data=df, x="auc_src", y="auc", hue="method", style="tgt", s=80)
    plt.plot([0,1],[0,1],'k--',alpha=0.5)
    plt.xlabel("Within-Dataset AUC")
    plt.ylabel("Cross-Dataset AUC")
    plt.title("Transferability vs. Within-Dataset AUC")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR_FIGS}/transfer_scatter.png", dpi=150)
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
    plot_heatmaps(df)
    plot_barplots(df)
    plot_scatter(df)
    print("\n✅ All plots saved in report/figures/")
    print("✅ Transfer results saved in report/tables/transfer_results.csv")


if __name__ == "__main__":
    main()

