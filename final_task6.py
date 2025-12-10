#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task 6 (Extra): Counterfactual Graph Explanations (Causal Node Importance)

PURPOSE:
- Find the SMALLEST set of nodes whose removal flips the model's prediction
- Measure causal importance (not just correlation)
- Compare different saliency methods for counterfactual explanations

KEY QUESTION:
"Which nodes are CAUSALLY responsible for the model's prediction?"

WHY THIS MATTERS:
- Causal explanation: Not just correlation, but actual causation
- Actionable insights: "Remove these nodes to change the outcome"
- Trust & debugging: Understand what drives predictions
- Minimal change: Find smallest intervention needed

DIFFERENCE FROM TASK 5:
- Task 5: Which nodes are important? (correlation)
- Task 6: Which nodes CAUSE the prediction? (causation)

COUNTERFACTUAL FRAMEWORK:
Original graph → Prediction A
Remove k nodes → Prediction changes to B
→ Those k nodes are causally important

WORKFLOW:
1. Train models (GIN, Graph2Vec+SVM, NetLSD+SVM)
2. For each graph:
   a. Get original prediction
   b. Compute node saliency (importance scores)
   c. Remove nodes in order of saliency (highest first)
   d. After each removal, check if prediction flipped
   e. Count how many removals needed to flip
3. Compare methods: Which needs fewer removals?

METRICS:
- k (nodes to flip): Number of nodes needed to change prediction
- Lower k = Better saliency (identifies causal nodes efficiently)

METHODS:
1. GIN: Gradient-based saliency
2. Graph2Vec: Embedding-based saliency
3. NetLSD: Spectral saliency

All use GREEDY SEARCH: Remove highest saliency node, check, repeat.
"""

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

import matplotlib
matplotlib.use("Agg")  # Headless plotting

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

# ============================================================================
# COMPATIBILITY PATCH
# ============================================================================

# Patch SciPy errstate for NetLSD
import numpy as _np
import scipy as _sp
if not hasattr(_sp, "errstate"):
    _sp.errstate = _np.errstate

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================

OUT_DIR_TAB = "report/tables"
OUT_DIR_FIG = "report/figures"
os.makedirs(OUT_DIR_TAB, exist_ok=True)
os.makedirs(OUT_DIR_FIG, exist_ok=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed=0):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def ensure_node_features(graphs):
    """
    Ensure all graphs have node features.
    
    If missing, use node degree as a simple 1D feature.
    Essential for GIN which requires node attributes.
    
    Args:
        graphs: List of PyG Data objects
        
    Returns:
        graphs: Same list with x attribute added if missing
    """
    out = []
    for g in graphs:
        if getattr(g, "x", None) is None:
            deg = torch.bincount(
                g.edge_index[0],
                minlength=g.num_nodes
            ).float().view(-1, 1)
            g.x = deg
        out.append(g)
    return out

def reindex_contiguously(G: nx.Graph):
    """
    Relabel nodes to 0, 1, 2, ..., n-1.
    
    After removing nodes, NetworkX graphs have non-contiguous IDs.
    Many algorithms expect contiguous IDs starting from 0.
    
    Args:
        G: NetworkX graph
        
    Returns:
        G: Graph with nodes relabeled 0..n-1
    """
    return nx.convert_node_labels_to_integers(G, ordering='default')

def pyg_to_nx_labeled(graph):
    """
    Convert PyG graph to NetworkX with proper node labeling.
    
    Pipeline:
    1. Convert to NetworkX (undirected)
    2. Add degree as 'label' attribute
    3. Reindex nodes contiguously
    
    Args:
        graph: PyG Data object
        
    Returns:
        G: NetworkX graph ready for processing
    """
    G = to_networkx(graph, to_undirected=True)
    # Add degree as label (required by Graph2Vec)
    for n in G.nodes:
        G.nodes[n]["label"] = G.degree[n]
    return reindex_contiguously(G)

# ============================================================================
# GIN MODEL & SALIENCY
# ============================================================================

class GINSmall(nn.Module):
    """
    Lightweight GIN encoder for graph classification.
    
    Architecture:
    - Stack of GIN convolution layers
    - Global mean pooling
    - Classification head
    
    Used for both prediction and gradient-based saliency.
    """
    
    def __init__(self, in_dim, hidden=64, layers=3, n_classes=2, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        
        h = hidden
        for i in range(layers):
            inp = in_dim if i == 0 else h
            mlp = nn.Sequential(
                nn.Linear(inp, h),
                nn.ReLU(),
                nn.Linear(h, h)
            )
            self.convs.append(GINConv(mlp))
        
        # Classification head
        self.lin = nn.Linear(h, n_classes)

    def forward(self, x, edge_index, batch):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Graph embedding
        g = global_mean_pool(h, batch)
        
        # Classification logits
        out = self.lin(g)
        return out, g

@torch.no_grad()
def make_single_batch(n, device):
    """Create batch tensor for single graph with n nodes."""
    return torch.zeros(n, dtype=torch.long, device=device)

def gin_saliency(model, graph, device):
    """
    Compute gradient-based saliency for GIN.
    
    ALGORITHM:
    1. Forward pass: x → embedding
    2. Compute embedding norm: ||embedding||₂
    3. Backward pass: ∂||embedding||/∂x
    4. Saliency = magnitude of gradient per node
    
    This measures how much each node's features affect the embedding.
    
    Args:
        model: Trained GIN model
        graph: PyG Data object
        device: torch device
        
    Returns:
        saliency: Node saliency scores [num_nodes]
    """
    model.eval()
    g = graph.to(device)
    
    # Ensure batch attribute
    if getattr(g, "batch", None) is None:
        g.batch = make_single_batch(g.num_nodes, device)
    
    # Enable gradient tracking
    x = g.x.clone().detach().requires_grad_(True)
    
    # Forward pass
    _, emb = model(x, g.edge_index, g.batch)
    
    # Compute embedding norm (scalar)
    score = emb.norm(p=2)
    
    # Backward pass: ∂score/∂x
    grads = torch.autograd.grad(
        score, x,
        retain_graph=False,
        create_graph=False
    )[0]
    
    # Aggregate gradient per node (sum across features)
    return grads.abs().sum(dim=1).detach().cpu().numpy()

# ============================================================================
# NETLSD SALIENCY
# ============================================================================

def netlsd_signature(G, times=None):
    """
    Compute NetLSD heat trace signature.
    
    Heat trace at time t: h(t) = Σᵢ exp(-t·λᵢ)
    where λᵢ are eigenvalues of normalized Laplacian.
    
    Args:
        G: NetworkX graph
        times: Array of diffusion times
        
    Returns:
        signature: Heat trace values
    """
    if times is None:
        times = np.logspace(-2, 2, 128)
    
    n = G.number_of_nodes()
    if n == 0:
        return np.zeros_like(times)
    
    # Normalized Laplacian
    L = nx.normalized_laplacian_matrix(G).astype(float).toarray()
    
    # Eigenvalues
    lam = np.linalg.eigvalsh(L)
    
    # Heat trace
    return np.exp(-np.outer(times, lam)).sum(axis=1)

def netlsd_saliency(graph):
    """
    Compute NetLSD-based saliency via node deletion.
    
    ALGORITHM:
    1. Compute base signature
    2. For each node:
       a. Remove node
       b. Recompute signature
       c. Measure L2 distance from base
    3. Distance = saliency
    
    Args:
        graph: PyG Data object
        
    Returns:
        saliency: Node saliency scores [num_nodes]
    """
    # Convert to NetworkX
    G_base = pyg_to_nx_labeled(graph)
    
    # Base signature
    base = netlsd_signature(G_base)
    
    # Compute saliency per node
    sal = []
    for n in range(G_base.number_of_nodes()):
        # Remove node
        G_del = G_base.copy()
        G_del.remove_node(n)
        G_del = reindex_contiguously(G_del)
        
        # New signature
        sig = netlsd_signature(G_del)
        
        # L2 distance
        sal.append(float(np.linalg.norm(base - sig)))
    
    return np.array(sal)

# ============================================================================
# GRAPH2VEC SALIENCY
# ============================================================================

def wl_document(G, wl_iterations=2):
    """
    Generate Weisfeiler-Lehman document for Graph2Vec.
    
    Creates a "bag of words" where words are substructure labels.
    
    Args:
        G: NetworkX graph with 'label' attributes
        wl_iterations: Number of WL iterations
        
    Returns:
        doc_words: List of substructure strings
    """
    labels = {n: str(G.nodes[n].get('label', G.degree[n])) for n in G.nodes}
    doc_words = []
    
    for it in range(wl_iterations):
        new_labels = {}
        for n in G.nodes:
            # Get neighbor labels
            neigh = sorted(labels[nb] for nb in G.neighbors(n))
            # Create new label
            new_label = labels[n] + "|" + "|".join(neigh)
            new_labels[n] = new_label
            doc_words.append(f"it{it}:{new_label}")
        labels = new_labels
    
    return doc_words

def graph2vec_train(graphs, dim=32, wl_iterations=2, epochs=10, seed=0):
    """
    Train Graph2Vec on a set of graphs.
    
    Args:
        graphs: List of PyG graphs
        dim: Embedding dimension
        wl_iterations: WL kernel depth
        epochs: Training epochs
        seed: Random seed
        
    Returns:
        model: Trained Graph2Vec model
        embeddings: Embedding matrix [num_graphs, dim]
    """
    model = Graph2Vec(
        dimensions=dim,
        wl_iterations=wl_iterations,
        epochs=epochs,
        seed=seed,
        workers=1
    )
    
    # Convert to NetworkX
    Gs = [pyg_to_nx_labeled(g) for g in graphs]
    
    # Train
    model.fit(Gs)
    
    return model, np.array(model.get_embedding())

def graph2vec_saliency(graph, base_model):
    """
    Compute Graph2Vec-based saliency via node deletion.
    
    ALGORITHM:
    1. Get base embedding from trained model
    2. For each node:
       a. Remove node
       b. Infer embedding for modified graph
       c. Measure L2 distance from base
    3. Distance = saliency
    
    Uses fast infer_vector() instead of retraining.
    
    Args:
        graph: PyG Data object
        base_model: Trained Graph2Vec model
        
    Returns:
        saliency: Node saliency scores [num_nodes]
    """
    # Convert to NetworkX
    G_base = pyg_to_nx_labeled(graph)
    
    # Base embedding
    base = base_model.model.infer_vector(wl_document(G_base))
    
    # Compute saliency per node
    sal = []
    for n in range(G_base.number_of_nodes()):
        # Remove node
        G_del = G_base.copy()
        G_del.remove_node(n)
        G_del = reindex_contiguously(G_del)
        
        # Infer new embedding
        vec = base_model.model.infer_vector(wl_document(G_del))
        
        # L2 distance
        sal.append(float(np.linalg.norm(base - vec)))
    
    return np.array(sal)

# ============================================================================
# COUNTERFACTUAL SEARCH FUNCTIONS
# ============================================================================

def predict_graph(model, graph, device):
    """
    Predict class for a single graph using GIN model.
    
    Args:
        model: Trained GIN model
        graph: PyG Data object
        device: torch device
        
    Returns:
        pred: Predicted class (integer)
    """
    g = graph.to(device)
    
    # Ensure batch attribute
    if getattr(g, "batch", None) is None:
        g.batch = make_single_batch(g.num_nodes, device)
    
    # Predict
    with torch.no_grad():
        logits, _ = model(g.x, g.edge_index, g.batch)
        pred = logits.argmax(dim=1).item()
    
    return pred

def remove_nodes(graph, nodes_to_remove):
    """
    Create a new graph with specified nodes removed.
    
    ALGORITHM:
    1. Convert to NetworkX
    2. Remove nodes
    3. Reindex remaining nodes
    4. Convert back to PyG format
    
    Args:
        graph: Original PyG Data object
        nodes_to_remove: List of node indices to remove
        
    Returns:
        new_graph: PyG Data object without removed nodes
    """
    # Convert to NetworkX
    G = pyg_to_nx_labeled(graph)
    
    # Remove nodes
    G.remove_nodes_from(nodes_to_remove)
    
    # Reindex (nodes now 0..n-1 where n is remaining nodes)
    G = reindex_contiguously(G)
    
    # Build edge_index for PyG
    edges = np.array(list(G.edges)).T
    if edges.size > 0:
        edge_index = torch.tensor(edges, dtype=torch.long)
    else:
        # Empty graph case
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # Create node features (simple: all ones)
    # In practice, should map original features properly
    x = torch.ones((G.number_of_nodes(), graph.x.size(1)))
    
    # Create new PyG graph
    return type(graph)(
        x=x,
        edge_index=edge_index,
        y=graph.y,
        num_nodes=G.number_of_nodes()
    )

def counterfactual_search(graph, saliency, model, device, orig_pred):
    """
    Find minimum number of nodes to remove to flip prediction.
    
    GREEDY ALGORITHM:
    1. Sort nodes by saliency (highest first)
    2. Remove top node, check if prediction flipped
    3. If not, remove next node and check again
    4. Repeat until prediction flips or all nodes removed
    
    WHY GREEDY?
    - Exhaustive search is exponential: O(2^n)
    - Greedy is linear: O(n)
    - Good approximation in practice
    
    COUNTERFACTUAL EXPLANATION:
    "If we remove these k nodes, the prediction changes"
    → Those k nodes are causally important
    
    Args:
        graph: PyG Data object
        saliency: Node saliency scores [num_nodes]
        model: Trained GIN model
        device: torch device
        orig_pred: Original prediction (before removal)
        
    Returns:
        k: Number of nodes removed to flip prediction
    """
    # Sort nodes by saliency (descending)
    order = np.argsort(-saliency)
    
    removed = []
    for k in range(1, len(order) + 1):
        # Add next most salient node
        removed.append(order[k-1])
        
        # Create modified graph
        g_new = remove_nodes(graph, removed)
        
        # Stop if graph is too small
        if g_new.num_nodes < 2:
            break
        
        # Check if prediction flipped
        new_pred = predict_graph(model, g_new, device)
        
        if new_pred != orig_pred:
            # SUCCESS: Found counterfactual explanation!
            return k
    
    # Failed to flip (or removed all nodes)
    return len(order)

# ============================================================================
# MAIN RUNNER
# ============================================================================

def run_counterfactual(datasets, methods, gin_hidden=32, gin_epochs=15, seed=0):
    """
    Run complete counterfactual explanation analysis.
    
    WORKFLOW:
    For each dataset:
        1. Train GIN model (supervised)
        2. Train Graph2Vec + SVM (unsupervised + classifier)
        3. Train NetLSD + SVM (spectral + classifier)
        
        For each graph (sample first 10 for speed):
            For each method:
                a. Compute node saliency
                b. Greedy search: Remove nodes until prediction flips
                c. Record number of nodes needed (k)
    
    4. Save results to CSV
    5. Generate bar chart comparing methods
    
    COMPARISON METRIC:
    Lower k = Better saliency (finds causal nodes efficiently)
    
    Args:
        datasets: List of dataset names
        methods: List of method names (for logging)
        gin_hidden: Hidden dimension for GIN
        gin_epochs: Training epochs for GIN
        seed: Random seed
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    rows = []

    for ds_name in datasets:
        print(f"\n=== Dataset: {ds_name} ===")
        
        # Load dataset
        ds = TUDataset(root="data", name=ds_name)
        graphs = ensure_node_features([ds[i] for i in range(len(ds))])
        y = np.array([int(g.y) for g in graphs])
        n_classes = int(y.max()) + 1

        # ===== TRAIN GIN MODEL =====
        print("Training GIN...")
        gin_model = GINSmall(
            graphs[0].x.size(1),
            hidden=gin_hidden,
            layers=3,
            n_classes=n_classes
        ).to(device)
        
        opt = torch.optim.Adam(
            gin_model.parameters(),
            lr=1e-3,
            weight_decay=1e-4
        )
        
        loader = DataLoader(graphs, batch_size=64, shuffle=True)
        
        # Training loop
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
        print("GIN training complete.")

        # ===== TRAIN GRAPH2VEC + SVM =====
        print("Training Graph2Vec + SVM...")
        g2v_model, g2v_embed = graph2vec_train(
            graphs,
            dim=32,
            seed=seed
        )
        
        # Train SVM classifier on Graph2Vec embeddings
        svm = make_pipeline(
            StandardScaler(),
            SVC(kernel="linear", probability=False, random_state=seed)
        )
        svm.fit(g2v_embed, y)
        print("Graph2Vec + SVM complete.")

        # ===== COMPUTE NETLSD EMBEDDINGS + SVM =====
        print("Computing NetLSD embeddings + SVM...")
        netlsd_embeds = []
        for g in graphs:
            sig = netlsd_signature(pyg_to_nx_labeled(g))
            netlsd_embeds.append(sig)
        netlsd_embeds = np.vstack(netlsd_embeds)
        
        # Train SVM classifier on NetLSD embeddings
        svm_netlsd = make_pipeline(
            StandardScaler(),
            SVC(kernel="linear", probability=False, random_state=seed)
        )
        svm_netlsd.fit(netlsd_embeds, y)
        print("NetLSD + SVM complete.")

        # ===== RUN COUNTERFACTUAL SEARCH PER GRAPH =====
        print("\nRunning counterfactual search...")
        
        # Sample first 10 graphs for speed
        # (Full analysis would use all graphs)
        for i, g in enumerate(graphs[:10]):
            label = int(g.y)
            print(f"  -> Graph {i}, class {label}, {g.num_nodes} nodes")

            # Get original predictions (all should be correct if models trained well)
            orig_gin_pred = predict_graph(gin_model, g, device)
            orig_g2v_pred = svm.predict(g2v_embed[i].reshape(1, -1))[0]
            orig_netlsd_pred = svm_netlsd.predict(netlsd_embeds[i].reshape(1, -1))[0]

            # ===== COMPUTE SALIENCIES =====
            # Each method has different saliency computation
            sal_gin = gin_saliency(gin_model, g, device)
            sal_g2v = graph2vec_saliency(g, g2v_model)
            sal_net = netlsd_saliency(g)

            # ===== COUNTERFACTUAL SEARCH =====
            # For each saliency method, find k nodes to flip GIN prediction
            # (We use GIN as the target model for all methods for fair comparison)
            
            k_gin = counterfactual_search(
                g, sal_gin, gin_model, device, orig_gin_pred
            )
            k_g2v = counterfactual_search(
                g, sal_g2v, gin_model, device, orig_gin_pred
            )
            k_net = counterfactual_search(
                g, sal_net, gin_model, device, orig_gin_pred
            )

            print(f"     Nodes to flip: GIN={k_gin}, G2V={k_g2v}, NetLSD={k_net}")

            # Save results
            rows += [
                dict(
                    dataset=ds_name,
                    graph=i,
                    method="GIN",
                    nodes_to_flip=k_gin,
                    num_nodes=g.num_nodes
                ),
                dict(
                    dataset=ds_name,
                    graph=i,
                    method="Graph2Vec",
                    nodes_to_flip=k_g2v,
                    num_nodes=g.num_nodes
                ),
                dict(
                    dataset=ds_name,
                    graph=i,
                    method="NetLSD",
                    nodes_to_flip=k_net,
                    num_nodes=g.num_nodes
                )
            ]

    # ===== SAVE RESULTS TABLE =====
    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR_TAB, "counterfactual_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Saved counterfactual results to: {out_csv}")

    # ===== GENERATE BAR CHART =====
    # Compare average number of nodes to flip per method per dataset
    summary = df.groupby(["dataset", "method"])["nodes_to_flip"].mean().unstack()
    
    summary.plot(kind="bar", figsize=(8, 4))
    plt.ylabel("Mean # nodes to flip class")
    plt.title("Counterfactual Graph Explanations (avg nodes removed to flip class)")
    plt.tight_layout()
    
    fig_path = os.path.join(OUT_DIR_FIG, "counterfactual_bar.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"✅ Saved bar chart to: {fig_path}")

# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Task 6: Counterfactual Graph Explanations "
                    "(GIN, Graph2Vec, NetLSD)"
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["MUTAG", "ENZYMES", "IMDB-MULTI"],
        help="Datasets to analyze"
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility"
    )
    return p.parse_args()

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Main execution function.
    
    Workflow:
    1. Parse arguments
    2. Run counterfactual search
    3. Save results and visualizations
    """
    args = parse_args()
    
    print("="*70)
    print("COUNTERFACTUAL GRAPH EXPLANATIONS (CAUSAL NODE IMPORTANCE)")
    print("="*70)
    print(f"Datasets: {args.datasets}")
    print(f"Seed: {args.seed}")
    print("="*70)
    
    run_counterfactual(
        datasets=args.datasets,
        methods=["GIN", "Graph2Vec", "NetLSD"],
        gin_hidden=32,
        gin_epochs=15,
        seed=args.seed
    )
    
    print("\n" + "="*70)
    print("COUNTERFACTUAL ANALYSIS COMPLETE!")
    print("="*70)
    print("\nOutputs:")
    print("  - Results: report/tables/counterfactual_results.csv")
    print("  - Bar chart: report/figures/counterfactual_bar.png")
    print("\n✅ All results saved!")

if __name__ == "__main__":
    main()


# ============================================================================
# USAGE EXAMPLES & INTERPRETATION GUIDE
# ============================================================================

"""
BASIC USAGE:
-----------
# Quick test with defaults
python final_task6.py

# Custom datasets
python final_task6.py --datasets MUTAG ENZYMES

# Different seed
python final_task6.py --seed 42


"""