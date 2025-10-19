#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task (N): Graph Embedding Explainability & Attention (All datasets & embeddings)
— Normalized & comparable across methods —

Embeddings:
  - GIN: true gradient saliency wrt node features (Grad-CAM-like)
  - Graph2Vec: fast pseudo-saliency via WL document + Doc2Vec infer_vector (no retraining)
  - NetLSD: pseudo-saliency via node-deletion signature change

Datasets:
  - MUTAG, ENZYMES, IMDB-MULTI (configurable via CLI)

Outputs:
  - report/figures/saliency/<DATASET>_<METHOD>_graph<ID>.png
  - report/figures/saliency/<DATASET>_<METHOD>_graph<ID>_bar.png
  - report/tables/saliency_summary.csv  (raw + normalized saliency per graph)
  - report/figures/saliency/saliency_mean_heatmap.png (mean normalized saliency per dataset × method)
"""

# ---------------- Headless plotting ----------------
import matplotlib
matplotlib.use("Agg")

import os, random, warnings, argparse
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.utils import to_networkx

from karateclub import Graph2Vec
from sklearn.metrics.pairwise import cosine_similarity

# ---- Patch SciPy errstate -> numpy.errstate (for NetLSD safety) ----
import numpy as _np
import scipy as _sp
if not hasattr(_sp, "errstate"):
    _sp.errstate = _np.errstate
# -------------------------------------------------------------------

OUT_DIR_FIG = "report/figures/saliency"
OUT_DIR_TAB = "report/tables"
os.makedirs(OUT_DIR_FIG, exist_ok=True)
os.makedirs(OUT_DIR_TAB, exist_ok=True)

# ---------------- Utilities ----------------
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def ensure_node_features(graphs):
    """If graph.x is missing, use degree as a 1D feature."""
    out = []
    for g in graphs:
        if getattr(g, "x", None) is None:
            deg = torch.bincount(g.edge_index[0], minlength=g.num_nodes).float().view(-1, 1)
            g.x = deg
        out.append(g)
    return out

def nx_with_degree_labels(G: nx.Graph) -> nx.Graph:
    """Stamp categorical 'label' = degree on nodes (Graph2Vec requires node labels)."""
    degs = dict(G.degree())
    for n in G.nodes:
        G.nodes[n]['label'] = int(degs[n])
    return G

def reindex_contiguously(G: nx.Graph) -> nx.Graph:
    """Relabel nodes to 0..n-1 (KarateClub Graph2Vec expects contiguous ids)."""
    return nx.convert_node_labels_to_integers(G, ordering='default')

def pyg_to_nx_labeled(graph) -> nx.Graph:
    """PyG Data -> undirected NetworkX with degree 'label' + contiguous ids."""
    G = to_networkx(graph, to_undirected=True)
    G = nx_with_degree_labels(G)
    G = reindex_contiguously(G)
    return G

# ---------------- WL document generator (minimal) ----------------
def wl_document(G: nx.Graph, wl_iterations: int = 2):
    """Build a Graph2Vec-like 'document' (list of subtree labels) via WL relabeling."""
    labels = {n: str(G.nodes[n].get('label', G.degree[n])) for n in G.nodes}
    doc_words = []
    for it in range(wl_iterations):
        new_labels = {}
        for n in G.nodes:
            neigh = sorted(labels[nb] for nb in G.neighbors(n))
            new_label = labels[n] + "|" + "|".join(neigh)
            new_labels[n] = new_label
            doc_words.append(f"it{it}:{new_label}")
        labels = new_labels
    return doc_words

# ---------------- Plotting ----------------
def plot_graph_saliency(graph, saliency, dataset, method, idx, topk_frac=0.0, add_colorbar=True):
    """Plot a single graph colored by node saliency (expects normalized [0,1] for viz)."""
    G = to_networkx(graph, to_undirected=True)
    pos = nx.spring_layout(G, seed=0)

    s = np.asarray(saliency, dtype=float)
    s = (s - s.min()) / (s.ptp() + 1e-9)
    colors = cm.viridis(s)

    fig, ax = plt.subplots(figsize=(5, 5))
    nx.draw_networkx(G, pos=pos, node_color=colors, node_size=80, edge_color="gray",
                     with_labels=False, alpha=0.9, ax=ax)
    title = f"{dataset} | {method} | Graph #{idx}"
    ax.set_title(title)
    ax.set_axis_off()

    # optional top-k outline
    if topk_frac and topk_frac > 0:
        k = max(1, int(topk_frac * len(s)))
        topk = np.argsort(s)[-k:]
        nx.draw_networkx_nodes(G, pos, nodelist=topk, node_size=180,
                               node_color='none', edgecolors='red', linewidths=1.5, ax=ax)

    if add_colorbar:
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="node saliency (normalized)")

    out_path = os.path.join(OUT_DIR_FIG, f"{dataset}_{method}_graph{idx}.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=220)
    plt.close(fig)

def plot_mean_saliency_bar(mean_value, dataset, method, idx):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.bar([0], [mean_value])
    ax.set_title(f"{dataset} | {method} | mean saliency (graph {idx})")
    ax.set_ylabel("Mean node saliency (normalized)")
    ax.set_xticks([])
    out_path = os.path.join(OUT_DIR_FIG, f"{dataset}_{method}_graph{idx}_bar.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

# ---------------- GIN model & true saliency ----------------
class GINSmall(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=3, n_classes=2, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        h = hidden
        for i in range(layers):
            inp = in_dim if i == 0 else h
            mlp = nn.Sequential(nn.Linear(inp, h), nn.ReLU(), nn.Linear(h, h))
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

@torch.no_grad()
def make_single_batch(n: int, device):
    return torch.zeros(n, dtype=torch.long, device=device)

def gin_saliency(model: nn.Module, graph, device: torch.device, mode="grad"):
    """
    mode: 'grad' (|∂||z||/∂x|) or 'gradxinput' (|∂||z||/∂x * x|)
    """
    model.eval()
    g = graph.to(device)
    if getattr(g, "batch", None) is None:
        g.batch = make_single_batch(g.num_nodes, device)
    x = g.x.clone().detach().requires_grad_(True)
    _, emb = model(x, g.edge_index, g.batch)
    score = emb.norm(p=2)
    grads = torch.autograd.grad(score, x, retain_graph=False, create_graph=False)[0]
    if mode.lower() == "gradxinput":
        sal = (grads * x).abs().sum(dim=1).detach().cpu().numpy()
    else:
        sal = grads.abs().sum(dim=1).detach().cpu().numpy()
    return sal

# ---------------- NetLSD saliency ----------------
def netlsd_signature(G: nx.Graph, times=None):
    if times is None:
        times = np.logspace(-2, 2, 128)
    n = G.number_of_nodes()
    if n == 0:
        return np.zeros_like(times)
    L = nx.normalized_laplacian_matrix(G).astype(float).toarray()
    lam = np.linalg.eigvalsh(L)
    return np.exp(-np.outer(times, lam)).sum(axis=1)

def _dist(a, b, metric="cosine"):
    if metric == "cosine":
        return 1.0 - float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0])
    elif metric == "relative_l2":
        return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-9))
    else:  # 'l2'
        return float(np.linalg.norm(a - b))

def netlsd_saliency(graph, metric="cosine"):
    G_base = pyg_to_nx_labeled(graph)
    base = netlsd_signature(G_base)
    sal = []
    for n in range(G_base.number_of_nodes()):
        G_del = G_base.copy()
        G_del.remove_node(n)            # <-- simple removal
        G_del = reindex_contiguously(G_del)
        sig = netlsd_signature(G_del)
        sal.append(_dist(base, sig, metric=metric))
    return np.array(sal, dtype=float)

# ---------------- Graph2Vec fast saliency: infer_vector ----------------
def graph2vec_train_one(graph, dim=32, wl_iterations=2, epochs=8, seed=0):
    """Train Graph2Vec on a single base graph; reuse Doc2Vec for infer_vector."""
    G = pyg_to_nx_labeled(graph)
    model = Graph2Vec(dimensions=dim, wl_iterations=wl_iterations, epochs=epochs,
                      seed=seed, workers=1, min_count=1)
    model.fit([G])
    return model

def wl_infer_embedding(model: Graph2Vec, G: nx.Graph, wl_iterations=2):
    G = nx_with_degree_labels(reindex_contiguously(G.copy()))
    words = wl_document(G, wl_iterations=wl_iterations)
    vec = model.model.infer_vector(words, epochs=20)
    return np.asarray(vec, dtype=float)

def graph2vec_saliency_fast(graph, base_model: Graph2Vec, wl_iterations=2, metric="cosine"):
    G_base = pyg_to_nx_labeled(graph)
    base = base_model.get_embedding()[0]
    n = G_base.number_of_nodes()
    sal = np.zeros(n, dtype=float)
    for node in range(n):
        G_del = G_base.copy()
        G_del.remove_node(node)
        emb_new = wl_infer_embedding(base_model, G_del, wl_iterations=wl_iterations)
        sal[node] = _dist(base, emb_new, metric=metric)
    return sal

# ---------------- Normalization helpers ----------------
def normalize_saliency(sal, mode="minmax"):
    if mode == "minmax":
        s = sal.astype(float)
        return (s - s.min()) / (s.ptp() + 1e-9)
    elif mode == "mean":
        m = sal.mean() + 1e-9
        return sal / m
    else:  # 'none'
        return sal

# ---------------- Runner (all datasets & methods) ----------------
def run_saliency(datasets, methods, sample_k=6, seed=0,
                 # GIN
                 gin_hidden=32, gin_epochs=15, gin_mode="grad",
                 # Graph2Vec
                 g2v_dim=32, g2v_wl=2, g2v_epochs=8, g2v_metric="cosine",
                 # NetLSD
                 netlsd_metric="cosine",
                 # Visualization / aggregation
                 topk_frac=0.0, norm_mode="minmax"):
    """
    norm_mode: 'minmax' (default), 'mean', or 'none' — used for cross-method comparability.
    g2v_metric/netlsd_metric: 'cosine' (default), 'relative_l2', or 'l2'
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    summary_rows = []

    for ds_name in datasets:
        print(f"\n=== Dataset: {ds_name} ===")
        ds = TUDataset(root="data", name=ds_name)
        graphs = ensure_node_features([ds[i] for i in range(len(ds))])
        y_all = np.array([int(g.y) for g in graphs])
        num_classes = int(y_all.max()) + 1

        # Train GIN lightly once per dataset
        gin_model = None
        if "GIN" in methods:
            gin_model = GINSmall(graphs[0].x.size(1), hidden=gin_hidden, layers=3, n_classes=num_classes).to(device)
            opt = torch.optim.Adam(gin_model.parameters(), lr=1e-3, weight_decay=1e-4)
            loader = DataLoader(graphs, batch_size=64, shuffle=True)
            gin_model.train()
            for _ in range(gin_epochs):
                for batch in loader:
                    batch = batch.to(device)
                    opt.zero_grad()
                    logits, _ = gin_model(batch.x, batch.edge_index, batch.batch)
                    loss = F.cross_entropy(logits, batch.y)
                    loss.backward()
                    opt.step()
            gin_model.eval()

        # sample graphs
        k = min(sample_k, len(graphs))
        sample_ids = np.random.choice(len(graphs), k, replace=False)

        for i in sample_ids:
            g = graphs[i]
            label = int(g.y)
            print(f"  -> graph {i} (class {label})")

            # Train Graph2Vec model for THIS base graph (fast)
            if "Graph2Vec" in methods:
                g2v_model = graph2vec_train_one(g, dim=g2v_dim, wl_iterations=g2v_wl,
                                                epochs=g2v_epochs, seed=seed)

            for method in methods:
                print(f"     method: {method}")
                try:
                    if method == "GIN":
                        sal_raw = gin_saliency(gin_model, g, device, mode=gin_mode)
                    elif method == "Graph2Vec":
                        sal_raw = graph2vec_saliency_fast(g, g2v_model, wl_iterations=g2v_wl, metric=g2v_metric)
                    elif method == "NetLSD":
                        sal_raw = netlsd_saliency(g, metric=netlsd_metric)
                        if len(sal_raw) != g.num_nodes:  # guard
                            sal_raw = np.pad(sal_raw, (0, max(0, g.num_nodes - len(sal_raw))), constant_values=0.0)[:g.num_nodes]
                    else:
                        continue

                    # Normalize for visualization & cross-method aggregation
                    sal_norm = normalize_saliency(sal_raw, mode=norm_mode)

                    # Plots
                    plot_graph_saliency(g, sal_norm, ds_name, method, i, topk_frac=topk_frac, add_colorbar=True)
                    plot_mean_saliency_bar(float(np.mean(sal_norm)), ds_name, method, i)

                    summary_rows.append({
                        "dataset": ds_name,
                        "method": method,
                        "graph_id": int(i),
                        "label": int(label),
                        "num_nodes": int(g.num_nodes),
                        # store BOTH raw and normalized summaries
                        "mean_saliency_raw": float(np.mean(sal_raw)),
                        "max_saliency_raw": float(np.max(sal_raw)),
                        "std_saliency_raw": float(np.std(sal_raw)),
                        "mean_saliency_norm": float(np.mean(sal_norm)),
                        "max_saliency_norm": float(np.max(sal_norm)),
                        "std_saliency_norm": float(np.std(sal_norm)),
                    })

                except Exception as e:
                    print(f"     [warn] saliency failed ({method}, graph {i}): {e}")
                    # still record a row to keep table consistent
                    summary_rows.append({
                        "dataset": ds_name,
                        "method": method,
                        "graph_id": int(i),
                        "label": int(label),
                        "num_nodes": int(getattr(g, "num_nodes", 0)),
                        "mean_saliency_raw": float("nan"),
                        "max_saliency_raw": float("nan"),
                        "std_saliency_raw": float("nan"),
                        "mean_saliency_norm": float("nan"),
                        "max_saliency_norm": float("nan"),
                        "std_saliency_norm": float("nan"),
                    })

    # Save summary CSV
    import pandas as pd
    df = pd.DataFrame(summary_rows)
    out_csv = os.path.join(OUT_DIR_TAB, "saliency_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Saved saliency summary to: {out_csv}")

    # Heatmap: dataset × method → mean of normalized mean-saliency (comparable)
    try:
        pivot = df.pivot_table(values="mean_saliency_norm", index="dataset", columns="method", aggfunc="mean")
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(7, 4.2))
        sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".3f", vmin=0.0, vmax=1.0, ax=ax)
        ax.set_title("Average Mean-Node Saliency (normalized, dataset × method)")
        fig.tight_layout()
        heat_path = os.path.join(OUT_DIR_FIG, "saliency_mean_heatmap.png")
        fig.savefig(heat_path, dpi=180)
        plt.close(fig)
        print(f"✅ Saved heatmap to: {heat_path}")
    except Exception as e:
        print(f"[warn] heatmap failed: {e}")

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Task (N): Graph saliency for GIN, Graph2Vec, NetLSD across datasets (normalized).")
    p.add_argument("--datasets", nargs="+", default=["MUTAG", "ENZYMES", "IMDB-MULTI"])
    p.add_argument("--methods",  nargs="+", default=["GIN", "Graph2Vec", "NetLSD"])
    p.add_argument("--sample_k", type=int, default=6, help="Graphs per dataset to visualize.")
    p.add_argument("--seed", type=int, default=0)

    # GIN params
    p.add_argument("--gin_hidden", type=int, default=32)
    p.add_argument("--gin_epochs", type=int, default=15)
    p.add_argument("--gin_mode", choices=["grad","gradxinput"], default="grad")

    # Graph2Vec params
    p.add_argument("--g2v_dim", type=int, default=32)
    p.add_argument("--g2v_wl", type=int, default=2)
    p.add_argument("--g2v_epochs", type=int, default=8)
    p.add_argument("--g2v_metric", choices=["cosine","relative_l2","l2"], default="cosine")

    # NetLSD params
    p.add_argument("--netlsd_metric", choices=["cosine","relative_l2","l2"], default="cosine")

    # Visualization / aggregation
    p.add_argument("--topk_frac", type=float, default=0.0, help="Outline top-k fraction of salient nodes (e.g., 0.1).")
    p.add_argument("--norm_mode", choices=["minmax","mean","none"], default="minmax",
                   help="Per-graph normalization used before aggregation/plots.")
    return p.parse_args()

def main():
    args = parse_args()
    run_saliency(
        datasets=args.datasets,
        methods=args.methods,
        sample_k=args.sample_k,
        seed=args.seed,
        gin_hidden=args.gin_hidden,
        gin_epochs=args.gin_epochs,
        gin_mode=args.gin_mode,
        g2v_dim=args.g2v_dim,
        g2v_wl=args.g2v_wl,
        g2v_epochs=args.g2v_epochs,
        g2v_metric=args.g2v_metric,
        netlsd_metric=args.netlsd_metric,
        topk_frac=args.topk_frac,
        norm_mode=args.norm_mode,
    )
    print("\nAll saliency figures saved under:", OUT_DIR_FIG)

if __name__ == "__main__":
    main()

