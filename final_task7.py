#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task: Concatenated Graph Embeddings (Are Complementary Embeddings Better?)

Idea:
  - Instead of treating Graph2Vec / NetLSD / GIN separately,
    we combine their embeddings into a single feature vector.

Example combinations:
  - graph2vec + netlsd
  - netlsd + gin
  - graph2vec + netlsd + gin

Procedure:
  1) For each dataset, seed, and embedding dimension:
       - Compute base embeddings for each method in the combo.
       - Normalize each embedding block independently (StandardScaler).
       - Concatenate normalized blocks into one feature matrix.
  2) Train a classifier (SVM, MLP) on concatenated embeddings (train split).
  3) Evaluate on test split: Accuracy, Macro-F1, AUC.
  4) Record:
       - Sum of embedding generation times of the constituent methods.
       - Classifier training time & memory.
  5) Aggregate across seeds and plot:
       - Accuracy vs. dimension (mean ± std) for SVM & MLP, per combo.
       - F1-score vs. dimension (mean ± std) for SVM & MLP, per combo.
       - AUC vs. dimension (mean ± std) for SVM & MLP, per combo.
       - Generation time vs. dimension, per combo.

Outputs:
  - CSV (per-run):      report/tables/concat_classification_eval.csv
  - CSV (aggregated):   report/tables/concat_classification_eval_agg.csv
  - PNG (figures):      report/figures/concat_{dataset}_{combo}_acc_vs_dim.png
                        report/figures/concat_{dataset}_{combo}_f1_vs_dim.png
                        report/figures/concat_{dataset}_{combo}_auc_vs_dim.png
                        report/figures/concat_{dataset}_{combo}_gentime_vs_dim.png
"""

# ------------------ Env + compat patches ------------------
import matplotlib
matplotlib.use("Agg")  # headless plotting

# Patch SciPy's errstate if missing (seen in some NumPy/SciPy mixes)
import numpy as _np
import scipy as _sp
if not hasattr(_sp, "errstate"):
    _sp.errstate = _np.errstate

# ----------------------------------------------------------

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

from karateclub import Graph2Vec  # NetLSD implemented below


# ------------------ Paths ------------------
OUT_DIR_TABLES = "report/tables"
OUT_DIR_FIGS   = "report/figures"
OUT_DIR_LOGS   = "report/logs"
os.makedirs(OUT_DIR_TABLES, exist_ok=True)
os.makedirs(OUT_DIR_FIGS,   exist_ok=True)
os.makedirs(OUT_DIR_LOGS,   exist_ok=True)


# ------------------ Utils ------------------
def ds_labels(ds):
    return np.array([int(g.y) for g in ds])

def ensure_node_features(graph: Data) -> Data:
    """
    If graph.x is missing or None, create a single feature column = node degree.
    Needed for GIN on datasets without node attributes (e.g. IMDB-MULTI).
    """
    if getattr(graph, "x", None) is None:
        G = to_networkx(graph, to_undirected=True)
        deg = np.array([d for _, d in G.degree()], dtype=np.float32)
        graph.x = torch.from_numpy(deg).view(-1, 1)
    return graph

@contextmanager
def timed_mem(name="block"):
    """
    Measure wallclock and memory (process RSS via psutil + Python peak via tracemalloc).
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
    ROC-AUC for binary & multiclass (macro OvR).
    Accepts decision_function (1D/2D) or predict_proba (2D). Returns NaN if not computable.
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
    """Fit classifier + compute metrics + timing & memory."""
    with timed_mem(f"train_{tag}") as meta:
        clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # continuous scores
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


# ------------------ Embeddings: Graph2Vec & NetLSD ------------------
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


def embed_graph2vec(ds_slice, dim=128, seed=0, epochs=20,
                    wl_iterations=2, min_count=5):
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
    """
    Compute NetLSD-style heat trace signature using the full eigendecomposition
    of the normalized Laplacian. Robust for small graphs.
    """
    n = G.number_of_nodes()
    if n == 0:
        return np.zeros_like(times)
    L = nx.normalized_laplacian_matrix(G).astype(float).toarray()
    lam = np.linalg.eigvalsh(L)  # Laplacian symmetric => stable eigenvalues
    return np.exp(-np.outer(times, lam)).sum(axis=1)


def embed_netlsd(ds_slice, dim=128, pca_seed=0,
                 n_times=256, t_min=1e-2, t_max=1e2):
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


# ------------------ Embeddings: GIN ------------------
class GINEncoder(nn.Module):
    """
    Simple GIN-based graph encoder with global mean pooling and linear head.
    We use the pooled representation as graph embedding of dimension `hidden`,
    and set `hidden = dim` so GIN embeddings match requested size.
    """
    def __init__(self, in_dim, hidden=64, layers=3, n_classes=2, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.bns    = nn.ModuleList()

        # first layer
        mlp0 = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.layers.append(GINConv(mlp0))
        self.bns.append(nn.BatchNorm1d(hidden))

        # subsequent layers
        for _ in range(layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.layers.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden))

        self.lin = nn.Linear(hidden, n_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.layers, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, batch)
        out = self.lin(g)
        return out, g  # logits, embedding


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
    Train GIN supervised on the full dataset (for simplicity) and
    export graph-level embeddings of dimension = dim.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    graphs = [ensure_node_features(g.clone()) for g in ds_slice]
    num_classes = len(np.unique([int(g.y) for g in graphs]))
    in_dim = graphs[0].x.size(-1)

    model = GINEncoder(
        in_dim=in_dim,
        hidden=dim,
        layers=layers,
        n_classes=num_classes,
        dropout=dropout,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()

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

    # collect embeddings for all graphs
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
    method = method.lower()
    if method == "graph2vec":
        return embed_graph2vec(ds_slice, dim=dim, seed=seed)
    elif method == "netlsd":
        return embed_netlsd(ds_slice, dim=dim, pca_seed=seed)
    elif method == "gin":
        return embed_gin(ds_slice, dim=dim, seed=seed, device=device)
    else:
        raise ValueError(f"Unknown method: {method}")


# ------------------ Concatenation helper ------------------
def concat_normalized_blocks(blocks):
    """
    blocks: list of (name, X_block) where X_block is [n_graphs, d_i].
    We standardize each block separately, then concatenate.
    """
    norm_blocks = []
    for name, X in blocks:
        scaler = StandardScaler(with_mean=True)
        Xn = scaler.fit_transform(X)
        norm_blocks.append(Xn)
    return np.concatenate(norm_blocks, axis=1)


# ------------------ Runner ------------------
def parse_combo_strings(combo_strs):
    """
    Parse a list of strings like ["graph2vec+netlsd", "netlsd+gin"]
    into list-of-lists: [["graph2vec","netlsd"], ["netlsd","gin"]]
    """
    combos = []
    for s in combo_strs:
        parts = [p.strip().lower() for p in s.split("+") if p.strip()]
        if len(parts) < 2:
            raise ValueError(f"Combination '{s}' must have at least two methods separated by '+'.")
        combos.append(parts)
    return combos


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
    For each dataset, combination of methods, dimension, and seed:
      - Compute embeddings for each method in combo.
      - Normalize each block separately.
      - Concatenate into one feature vector per graph.
      - Train SVM & MLP on concatenated embeddings.
      - Log metrics & times.
    """
    rows = []

    # which base methods do we need overall?
    all_base_methods = sorted({m for combo in combos for m in combo})
    print(f"Base methods used in combos: {all_base_methods}")

    for ds_name in datasets:
        print(f"\n=== Dataset: {ds_name} ===")
        ds = TUDataset(root="data", name=ds_name)
        y_all = ds_labels(ds)
        classes = np.unique(y_all)
        idx_all = np.arange(len(ds))
        ds_all  = [ds[i] for i in idx_all]

        for seed in seeds:
            print(f"\n-- Seed: {seed} --")
            tr_idx, te_idx = train_test_split(
                idx_all,
                test_size=test_size,
                random_state=seed,
                stratify=y_all,
            )

            # cache single-method embeddings: (method, dim) -> (X_all, info)
            embedding_cache = {}

            for dim in dims:
                print(f"\n   >> dim={dim}")
                # compute embeddings for all base methods required
                for method in all_base_methods:
                    key = (method, dim)
                    if key in embedding_cache:
                        continue
                    print(f"      [emb] {method} (dim={dim})")
                    X_all, gen_info = get_single_method_embedding(
                        method, ds_all, dim=dim, seed=seed, device=device
                    )
                    embedding_cache[key] = (X_all, gen_info)

                # Now evaluate each combination for this dim
                for combo_methods in combos:
                    combo_name = "+".join(combo_methods)
                    print(f"      [combo] {combo_name} | dim={dim} | seed={seed}")

                    # collect blocks & generation info
                    blocks = []
                    gen_time = 0.0
                    gen_rss_before = None
                    gen_rss_after = None
                    gen_py_peak = 0.0

                    for m in combo_methods:
                        X_m, info_m = embedding_cache[(m, dim)]
                        blocks.append((m, X_m))
                        gen_time += info_m["gen_time_sec"]
                        gen_py_peak = max(gen_py_peak, info_m["gen_py_peak_mb"])
                        if gen_rss_before is None:
                            gen_rss_before = info_m["gen_rss_before_mb"]
                        gen_rss_after = info_m["gen_rss_after_mb"]

                    # normalized concatenation
                    X_all_concat = concat_normalized_blocks(blocks)

                    # train/test split
                    X_train, X_test = X_all_concat[tr_idx], X_all_concat[te_idx]
                    y_train, y_test = y_all[tr_idx], y_all[te_idx]

                    # classifiers
                    svm = make_pipeline(
                        StandardScaler(with_mean=True),
                        LinearSVC(dual=False, random_state=seed),
                    )
                    svm_metrics, svm_time, _ = eval_classifier(
                        svm, X_train, X_test, y_train, y_test, classes, tag="svm"
                    )

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

                    row = dict(
                        dataset=ds_name,
                        combo=combo_name,
                        base_methods=",".join(combo_methods),
                        dim=int(dim),
                        seed=int(seed),
                        n_graphs=len(ds),
                        n_classes=len(classes),
                        # generation stats (sum over methods)
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
                        # MLP
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

                    # progressive write
                    pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    print(f"\nSaved per-run concatenation results to {out_csv}")
    return df


# ------------------ Aggregation + summary plots ------------------
def aggregate_and_plot(df, out_csv_agg=f"{OUT_DIR_TABLES}/concat_classification_eval_agg.csv"):
    """
    Aggregate across seeds: mean/std per (dataset, combo, dim, classifier).
    Produce plots:
      - Accuracy vs dimension (mean ± std) for SVM & MLP
      - F1 vs dimension (mean ± std) for SVM & MLP
      - AUC vs dimension (mean ± std) for SVM & MLP
      - Generation time vs dimension
    """
    if df.empty:
        print("No rows to aggregate.")
        return

    # Long format per classifier
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

    # --- Plots per dataset/combo ---
    for dataset in agg["dataset"].unique():
        subD = agg[agg["dataset"] == dataset]
        for combo_name in subD["combo"].unique():
            subDM = subD[subD["combo"] == combo_name].sort_values("dim")
            safe_combo = combo_name.replace("+", "-")

            # 1) Accuracy vs dim (SVM & MLP)
            fig = plt.figure(figsize=(6,4))
            ax = fig.add_subplot(111)
            for clf in ["svm", "mlp"]:
                sdf = subDM[subDM["clf"] == clf]
                if sdf.empty:
                    continue
                x = sdf["dim"].values
                y = sdf["acc_mean"].values
                s = sdf["acc_std"].fillna(0).values

                ax.fill_between(x, y - s, y + s, alpha=0.20, linewidth=0)
                ax.errorbar(x, y, yerr=s, fmt='-o', capsize=4, label=f"{clf.upper()} acc")

            n_runs = int(subDM["n_runs"].max()) if "n_runs" in subDM else None
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

            # 2) F1 vs dim (SVM & MLP)
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

            # 3) AUC vs dim (SVM & MLP)
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

            # 4) Generation time vs dim
            sdf = subDM[subDM["clf"] == "svm"].sort_values("dim")
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


# ------------------ CLI ------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Concatenated graph embeddings (Graph2Vec, NetLSD, GIN combinations)"
    )
    p.add_argument("--datasets", nargs="+", default=["MUTAG", "ENZYMES", "IMDB-MULTI"])
    # combos like: graph2vec+netlsd, netlsd+gin, graph2vec+netlsd+gin
    p.add_argument(
        "--combos",
        nargs="+",
        default=["graph2vec+netlsd", "netlsd+gin", "graph2vec+netlsd+gin"],
        help="Embedding method combinations, e.g. 'graph2vec+netlsd', 'netlsd+gin'.",
    )
    p.add_argument("--dims",  nargs="+", type=int, default=[32, 64, 128])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default=f"{OUT_DIR_TABLES}/concat_classification_eval.csv")
    return p.parse_args()


def main():
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    args = parse_args()

    combos = parse_combo_strings(args.combos)

    df = run_experiment(
        datasets=args.datasets,
        combos=combos,
        dims=args.dims,
        seeds=args.seeds,
        device=args.device,
        test_size=args.test_size,
        out_csv=args.out,
    )
    aggregate_and_plot(df)


if __name__ == "__main__":
    main()

