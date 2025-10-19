#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task (O): Counterfactual Graph Explanations (Causal Importance)
==============================================================

Goal:
  For each graph, find the smallest set of nodes whose removal flips the model’s predicted class.

Embeddings / Methods:
  - GIN (supervised)
  - Graph2Vec (unsupervised embedding + SVM classifier)
  - NetLSD (unsupervised embedding + SVM classifier)

Datasets:
  - MUTAG, ENZYMES, IMDB-MULTI

Outputs:
  - report/tables/counterfactual_results.csv
  - report/figures/counterfactual_bar_<dataset>.png
"""

# ---------------- Headless plotting ----------------
import matplotlib
matplotlib.use("Agg")

import os, random, warnings, argparse
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool

from karateclub import Graph2Vec
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity

# ---- Patch SciPy errstate -> numpy.errstate ----
import numpy as _np
import scipy as _sp
if not hasattr(_sp, "errstate"):
    _sp.errstate = _np.errstate
# -------------------------------------------------

OUT_DIR_TAB = "report/tables"
OUT_DIR_FIG = "report/figures"
os.makedirs(OUT_DIR_TAB, exist_ok=True)
os.makedirs(OUT_DIR_FIG, exist_ok=True)

# ---------------- Utility Functions ----------------
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def ensure_node_features(graphs):
    out = []
    for g in graphs:
        if getattr(g, "x", None) is None:
            deg = torch.bincount(g.edge_index[0], minlength=g.num_nodes).float().view(-1, 1)
            g.x = deg
        out.append(g)
    return out

def reindex_contiguously(G: nx.Graph):
    return nx.convert_node_labels_to_integers(G, ordering='default')

def pyg_to_nx_labeled(graph):
    G = to_networkx(graph, to_undirected=True)
    for n in G.nodes:
        G.nodes[n]["label"] = G.degree[n]
    return reindex_contiguously(G)

# ---------------- GIN Model ----------------
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
def make_single_batch(n, device):
    return torch.zeros(n, dtype=torch.long, device=device)

def gin_saliency(model, graph, device):
    """|∂||z||/∂x| as saliency."""
    model.eval()
    g = graph.to(device)
    if getattr(g, "batch", None) is None:
        g.batch = make_single_batch(g.num_nodes, device)
    x = g.x.clone().detach().requires_grad_(True)
    _, emb = model(x, g.edge_index, g.batch)
    score = emb.norm(p=2)
    grads = torch.autograd.grad(score, x, retain_graph=False, create_graph=False)[0]
    return grads.abs().sum(dim=1).detach().cpu().numpy()

# ---------------- NetLSD ----------------
def netlsd_signature(G, times=None):
    if times is None:
        times = np.logspace(-2, 2, 128)
    n = G.number_of_nodes()
    if n == 0:
        return np.zeros_like(times)
    L = nx.normalized_laplacian_matrix(G).astype(float).toarray()
    lam = np.linalg.eigvalsh(L)
    return np.exp(-np.outer(times, lam)).sum(axis=1)

def netlsd_saliency(graph):
    G_base = pyg_to_nx_labeled(graph)
    base = netlsd_signature(G_base)
    sal = []
    for n in range(G_base.number_of_nodes()):
        G_del = G_base.copy()
        G_del.remove_node(n)
        G_del = reindex_contiguously(G_del)
        sig = netlsd_signature(G_del)
        sal.append(float(np.linalg.norm(base - sig)))
    return np.array(sal)

# ---------------- Graph2Vec ----------------
def wl_document(G, wl_iterations=2):
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

def graph2vec_train(graphs, dim=32, wl_iterations=2, epochs=10, seed=0):
    model = Graph2Vec(dimensions=dim, wl_iterations=wl_iterations, epochs=epochs, seed=seed, workers=1)
    Gs = [pyg_to_nx_labeled(g) for g in graphs]
    model.fit(Gs)
    return model, np.array(model.get_embedding())

def graph2vec_saliency(graph, base_model):
    G_base = pyg_to_nx_labeled(graph)
    base = base_model.model.infer_vector(wl_document(G_base))
    sal = []
    for n in range(G_base.number_of_nodes()):
        G_del = G_base.copy()
        G_del.remove_node(n)
        G_del = reindex_contiguously(G_del)
        vec = base_model.model.infer_vector(wl_document(G_del))
        sal.append(float(np.linalg.norm(base - vec)))
    return np.array(sal)

# ---------------- Counterfactual Logic ----------------
def predict_graph(model, graph, device):
    g = graph.to(device)
    if getattr(g, "batch", None) is None:
        g.batch = make_single_batch(g.num_nodes, device)
    with torch.no_grad():
        logits, _ = model(g.x, g.edge_index, g.batch)
        pred = logits.argmax(dim=1).item()
    return pred

def remove_nodes(graph, nodes_to_remove):
    G = pyg_to_nx_labeled(graph)
    G.remove_nodes_from(nodes_to_remove)
    G = reindex_contiguously(G)
    edges = np.array(list(G.edges)).T
    edge_index = torch.tensor(edges, dtype=torch.long) if edges.size > 0 else torch.empty((2,0), dtype=torch.long)
    x = torch.ones((G.number_of_nodes(), graph.x.size(1)))
    return type(graph)(x=x, edge_index=edge_index, y=graph.y, num_nodes=G.number_of_nodes())

def counterfactual_search(graph, saliency, model, device, orig_pred):
    order = np.argsort(-saliency)
    removed = []
    for k in range(1, len(order)+1):
        removed.append(order[k-1])
        g_new = remove_nodes(graph, removed)
        if g_new.num_nodes < 2:
            break
        new_pred = predict_graph(model, g_new, device)
        if new_pred != orig_pred:
            return k
    return len(order)

# ---------------- Main Runner ----------------
def run_counterfactual(datasets, methods, gin_hidden=32, gin_epochs=15, seed=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    rows = []

    for ds_name in datasets:
        print(f"\n=== Dataset: {ds_name} ===")
        ds = TUDataset(root="data", name=ds_name)
        graphs = ensure_node_features([ds[i] for i in range(len(ds))])
        y = np.array([int(g.y) for g in graphs])
        n_classes = int(y.max()) + 1

        # --- Train GIN ---
        gin_model = GINSmall(graphs[0].x.size(1), hidden=gin_hidden, layers=3, n_classes=n_classes).to(device)
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

        # --- Train Graph2Vec & SVM ---
        g2v_model, g2v_embed = graph2vec_train(graphs, dim=32, seed=seed)
        svm = make_pipeline(StandardScaler(), SVC(kernel="linear", probability=False, random_state=seed))
        svm.fit(g2v_embed, y)

        # --- NetLSD Embeddings ---
        netlsd_embeds = []
        for g in graphs:
            sig = netlsd_signature(pyg_to_nx_labeled(g))
            netlsd_embeds.append(sig)
        netlsd_embeds = np.vstack(netlsd_embeds)
        svm_netlsd = make_pipeline(StandardScaler(), SVC(kernel="linear", probability=False, random_state=seed))
        svm_netlsd.fit(netlsd_embeds, y)

        # --- Run per-graph ---
        for i, g in enumerate(graphs[:10]):  # sample first 10 for speed
            label = int(g.y)
            print(f"  -> Graph {i}, class {label}")

            # Original predictions
            orig_gin_pred = predict_graph(gin_model, g, device)
            orig_g2v_pred = svm.predict(g2v_embed[i].reshape(1,-1))[0]
            orig_netlsd_pred = svm_netlsd.predict(netlsd_embeds[i].reshape(1,-1))[0]

            # Saliencies
            sal_gin = gin_saliency(gin_model, g, device)
            sal_g2v = graph2vec_saliency(g, g2v_model)
            sal_net = netlsd_saliency(g)

            # Counterfactual search (how many nodes to flip)
            k_gin = counterfactual_search(g, sal_gin, gin_model, device, orig_gin_pred)
            k_g2v = counterfactual_search(g, sal_g2v, gin_model, device, orig_gin_pred)
            k_net = counterfactual_search(g, sal_net, gin_model, device, orig_gin_pred)

            rows += [
                dict(dataset=ds_name, graph=i, method="GIN", nodes_to_flip=k_gin, num_nodes=g.num_nodes),
                dict(dataset=ds_name, graph=i, method="Graph2Vec", nodes_to_flip=k_g2v, num_nodes=g.num_nodes),
                dict(dataset=ds_name, graph=i, method="NetLSD", nodes_to_flip=k_net, num_nodes=g.num_nodes)
            ]

    # Save table
    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR_TAB, "counterfactual_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Saved counterfactual results to: {out_csv}")

    # Plot: mean #nodes to flip per method per dataset
    summary = df.groupby(["dataset","method"])["nodes_to_flip"].mean().unstack()
    summary.plot(kind="bar", figsize=(8,4))
    plt.ylabel("Mean # nodes to flip class")
    plt.title("Counterfactual Graph Explanations (avg nodes removed to flip class)")
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR_FIG, "counterfactual_bar.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"✅ Saved bar chart to: {fig_path}")


# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Task (O): Counterfactual Graph Explanations (GIN, Graph2Vec, NetLSD)")
    p.add_argument("--datasets", nargs="+", default=["MUTAG", "ENZYMES", "IMDB-MULTI"])
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    run_counterfactual(
        datasets=args.datasets,
        methods=["GIN","Graph2Vec","NetLSD"],
        gin_hidden=32,
        gin_epochs=15,
        seed=args.seed
    )
    print("\nAll results saved under 'report/tables' and 'report/figures'.")

if __name__ == "__main__":
    main()

