#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task 5 (Extra): Graph Embedding Explainability & Attention

PURPOSE:
- Understand WHICH NODES are most important for graph embeddings
- Provide interpretability for black-box embedding methods
- Compare saliency patterns across different methods
- Visualize node importance with colored graphs

KEY QUESTION:
"Which nodes in a graph contribute most to its embedding?"

WHY THIS MATTERS:
- Explainability: Understand what the embedding "sees"
- Trust: Validate that embeddings focus on meaningful structures
- Debugging: Identify if method captures relevant patterns
- Domain insight: Discover important substructures

METHODS:
1. GIN (Supervised): TRUE gradient-based saliency (Grad-CAM-like)
   - Compute ∂||embedding||/∂x (gradient of embedding norm w.r.t. node features)
   - Analogous to neural network visualization techniques
   
2. Graph2Vec (Unsupervised): PSEUDO-saliency via node deletion
   - Remove each node, recompute embedding, measure change
   - Larger change → More important node
   
3. NetLSD (Spectral): PSEUDO-saliency via spectral signature change
   - Remove each node, recompute heat trace, measure distance
   - Nodes affecting global structure → High saliency

WORKFLOW:
1. Load dataset and sample graphs
2. For each graph:
   a. Compute base embedding
   b. For each node:
      - Compute node saliency (method-specific)
   c. Normalize saliency [0,1] for visualization
   d. Plot graph colored by node importance
3. Aggregate statistics across graphs
4. Generate heatmap comparing methods

NORMALIZATION:
- Raw saliency values vary by method (different scales)
- Normalize to [0,1] within each graph for fair comparison
- Allows cross-method aggregation and visualization
"""

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

import matplotlib
matplotlib.use("Agg")  # Headless plotting (no display needed)

import os, random, warnings, argparse
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # Color maps for visualization
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

# ============================================================================
# COMPATIBILITY PATCH
# ============================================================================

# Patch SciPy errstate for NetLSD safety
import numpy as _np
import scipy as _sp
if not hasattr(_sp, "errstate"):
    _sp.errstate = _np.errstate

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================

OUT_DIR_FIG = "report/figures"
OUT_DIR_TAB = "report/tables"
os.makedirs(OUT_DIR_FIG, exist_ok=True)
os.makedirs(OUT_DIR_TAB, exist_ok=True)

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
    
    If a graph lacks node attributes, use degree as a simple 1D feature.
    This is essential for GIN which requires node features.
    
    Args:
        graphs: List of PyG Data objects
        
    Returns:
        graphs: Same list with x attribute added if missing
    """
    out = []
    for g in graphs:
        if getattr(g, "x", None) is None:
            # Compute node degree as feature
            deg = torch.bincount(
                g.edge_index[0],
                minlength=g.num_nodes
            ).float().view(-1, 1)
            g.x = deg
        out.append(g)
    return out

def nx_with_degree_labels(G: nx.Graph) -> nx.Graph:
    """
    Add discrete 'label' attribute to nodes (required by Graph2Vec).
    
    Graph2Vec expects nodes to have categorical labels.
    We use node degree as the label (standard practice).
    
    Args:
        G: NetworkX graph
        
    Returns:
        G: Same graph with 'label' attribute on each node
    """
    degs = dict(G.degree())
    for n in G.nodes:
        G.nodes[n]['label'] = int(degs[n])
    return G

def reindex_contiguously(G: nx.Graph) -> nx.Graph:
    """
    Relabel nodes to 0, 1, 2, ..., n-1.
    
    KarateClub's Graph2Vec expects contiguous integer node IDs.
    NetworkX graphs might have arbitrary node IDs after manipulations.
    
    Args:
        G: NetworkX graph
        
    Returns:
        G: Graph with nodes relabeled 0..n-1
    """
    return nx.convert_node_labels_to_integers(G, ordering='default')

def pyg_to_nx_labeled(graph) -> nx.Graph:
    """
    Convert PyG graph to NetworkX with proper labeling.
    
    Pipeline:
    1. Convert to NetworkX (undirected)
    2. Add degree labels to nodes
    3. Reindex nodes contiguously
    
    Args:
        graph: PyG Data object
        
    Returns:
        G: NetworkX graph ready for Graph2Vec/NetLSD
    """
    G = to_networkx(graph, to_undirected=True)
    G = nx_with_degree_labels(G)
    G = reindex_contiguously(G)
    return G

# ============================================================================
# WEISFEILER-LEHMAN DOCUMENT GENERATOR
# ============================================================================

def wl_document(G: nx.Graph, wl_iterations: int = 2):
    """
    Generate a "document" of substructure labels using WL kernel.
    
    Weisfeiler-Lehman (WL) algorithm:
    1. Start with initial node labels
    2. For each iteration:
       a. Aggregate neighbor labels
       b. Create new label: old_label + sorted(neighbor_labels)
       c. Add to document
    
    This creates a bag-of-words representation where "words" are
    substructure patterns (node + neighborhood context).
    
    Graph2Vec uses this document as input to doc2vec.
    
    Args:
        G: NetworkX graph with 'label' attribute on nodes
        wl_iterations: Number of WL refinement iterations
        
    Returns:
        doc_words: List of substructure label strings
    """
    # Initialize labels from node attributes
    labels = {n: str(G.nodes[n].get('label', G.degree[n])) for n in G.nodes}
    doc_words = []
    
    # WL refinement iterations
    for it in range(wl_iterations):
        new_labels = {}
        for n in G.nodes:
            # Get labels of all neighbors
            neigh = sorted(labels[nb] for nb in G.neighbors(n))
            
            # Create new label: current_label | neighbor_labels
            new_label = labels[n] + "|" + "|".join(neigh)
            new_labels[n] = new_label
            
            # Add to document with iteration prefix
            doc_words.append(f"it{it}:{new_label}")
        
        labels = new_labels
    
    return doc_words

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_graph_saliency(graph, saliency, dataset, method, idx, 
                        topk_frac=0.0, add_colorbar=True):
    """
    Plot a single graph with nodes colored by saliency.
    
    Visualization details:
    - Node color: Viridis colormap (blue=low, yellow=high)
    - Node size: Fixed (80)
    - Edges: Gray
    - Optional: Red outline for top-k most salient nodes
    - Colorbar: Shows saliency scale [0,1]
    
    Args:
        graph: PyG Data object
        saliency: Node saliency values [num_nodes]
        dataset: Dataset name (for title)
        method: Method name (for title)
        idx: Graph index (for title and filename)
        topk_frac: Fraction of nodes to outline in red (0.0 = none)
        add_colorbar: Whether to add colorbar
    """
    # Convert to NetworkX for visualization
    G = to_networkx(graph, to_undirected=True)
    
    # Spring layout for node positions
    pos = nx.spring_layout(G, seed=0)

    # Normalize saliency to [0,1] for colormap
    s = np.asarray(saliency, dtype=float)
    s = (s - s.min()) / (s.ptp() + 1e-9)  # ptp = peak-to-peak (max-min)
    
    # Map saliency to colors using Viridis colormap
    colors = cm.viridis(s)

    # Create figure
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Draw graph
    nx.draw_networkx(
        G, pos=pos,
        node_color=colors,
        node_size=80,
        edge_color="gray",
        with_labels=False,  # No node labels (too cluttered)
        alpha=0.9,
        ax=ax
    )
    
    # Title
    title = f"{dataset} | {method} | Graph #{idx}"
    ax.set_title(title)
    ax.set_axis_off()

    # Optional: Outline top-k most salient nodes
    if topk_frac and topk_frac > 0:
        k = max(1, int(topk_frac * len(s)))
        topk = np.argsort(s)[-k:]  # Indices of top-k nodes
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=topk,
            node_size=180,
            node_color='none',
            edgecolors='red',
            linewidths=1.5,
            ax=ax
        )

    # Add colorbar
    if add_colorbar:
        sm = plt.cm.ScalarMappable(
            cmap='viridis',
            norm=plt.Normalize(vmin=0, vmax=1)
        )
        sm.set_array([])
        fig.colorbar(
            sm, ax=ax,
            fraction=0.046,
            pad=0.04,
            label="node saliency (normalized)"
        )

    # Save figure
    out_path = os.path.join(OUT_DIR_FIG, f"{dataset}_{method}_graph{idx}.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=220)
    plt.close(fig)

def plot_mean_saliency_bar(mean_value, dataset, method, idx):
    """
    Plot a simple bar chart showing mean saliency.
    
    This provides a quick summary statistic alongside the full graph.
    
    Args:
        mean_value: Mean saliency across all nodes
        dataset: Dataset name
        method: Method name
        idx: Graph index
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.bar([0], [mean_value])
    ax.set_title(f"{dataset} | {method} | mean saliency (graph {idx})")
    ax.set_ylabel("Mean node saliency (normalized)")
    ax.set_xticks([])
    
    out_path = os.path.join(OUT_DIR_FIG, f"{dataset}_{method}_graph{idx}_bar.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

# ============================================================================
# GIN MODEL & TRUE GRADIENT-BASED SALIENCY
# ============================================================================

class GINSmall(nn.Module):
    """
    Lightweight GIN encoder for graph embeddings.
    
    Architecture:
    - Stack of GIN convolution layers
    - Global mean pooling
    - Classification head (for supervised training)
    
    For saliency, we use the graph embedding (before classification head).
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
        
        # Graph-level embedding
        g = global_mean_pool(h, batch)
        
        # Classification logits
        out = self.lin(g)
        return out, g

@torch.no_grad()
def make_single_batch(n: int, device):
    """Create a batch tensor for a single graph with n nodes."""
    return torch.zeros(n, dtype=torch.long, device=device)

def gin_saliency(model: nn.Module, graph, device: torch.device, mode="grad"):
    """
    Compute TRUE gradient-based node saliency for GIN.
    
    ALGORITHM (Grad-CAM-like for graphs):
    1. Forward pass: x → embedding
    2. Compute embedding norm: ||embedding||₂
    3. Backward pass: ∂||embedding||/∂x
    4. Aggregate gradient per node (sum across features)
    
    INTUITION:
    - Gradient measures how much each node feature affects the embedding
    - High gradient → Node is important for the embedding
    - This is TRUE saliency (exact gradient computation)
    
    MODES:
    - "grad": |∂||z||/∂x| (gradient magnitude)
    - "gradxinput": |∂||z||/∂x * x| (gradient × input, like Grad-CAM)
    
    Args:
        model: Trained GIN model
        graph: PyG Data object (single graph)
        device: torch device
        mode: "grad" or "gradxinput"
        
    Returns:
        saliency: Node saliency scores [num_nodes]
    """
    model.eval()
    g = graph.to(device)
    
    # Ensure batch attribute exists
    if getattr(g, "batch", None) is None:
        g.batch = make_single_batch(g.num_nodes, device)
    
    # Enable gradient tracking for node features
    x = g.x.clone().detach().requires_grad_(True)
    
    # Forward pass to get embedding
    _, emb = model(x, g.edge_index, g.batch)
    
    # Compute L2 norm of embedding (scalar)
    score = emb.norm(p=2)
    
    # Backward pass: compute ∂score/∂x
    grads = torch.autograd.grad(
        score, x,
        retain_graph=False,
        create_graph=False
    )[0]
    
    # Aggregate gradient per node
    if mode.lower() == "gradxinput":
        # Gradient × Input (like Grad-CAM)
        sal = (grads * x).abs().sum(dim=1).detach().cpu().numpy()
    else:
        # Just gradient magnitude
        sal = grads.abs().sum(dim=1).detach().cpu().numpy()
    
    return sal

# ============================================================================
# NETLSD PSEUDO-SALIENCY VIA NODE DELETION
# ============================================================================

def netlsd_signature(G: nx.Graph, times=None):
    """
    Compute NetLSD heat trace signature.
    
    NetLSD characterizes graphs through heat diffusion:
    - Normalized Laplacian eigenvalues: λ₁, λ₂, ..., λₙ
    - Heat trace at time t: h(t) = Σᵢ exp(-t·λᵢ)
    
    This signature captures global spectral properties.
    
    Args:
        G: NetworkX graph
        times: Array of diffusion times (log-spaced)
        
    Returns:
        signature: Heat trace values [len(times)]
    """
    if times is None:
        times = np.logspace(-2, 2, 128)  # 128 time points
    
    n = G.number_of_nodes()
    if n == 0:
        return np.zeros_like(times)
    
    # Normalized Laplacian
    L = nx.normalized_laplacian_matrix(G).astype(float).toarray()
    
    # Eigenvalues
    lam = np.linalg.eigvalsh(L)
    
    # Heat trace: h(t) = sum_i exp(-t * lambda_i)
    return np.exp(-np.outer(times, lam)).sum(axis=1)

def _dist(a, b, metric="cosine"):
    """
    Compute distance between two vectors.
    
    METRICS:
    - "cosine": 1 - cosine_similarity (0=same, 2=opposite)
    - "relative_l2": ||a-b|| / ||a|| (relative change)
    - "l2": ||a-b|| (absolute change)
    
    Args:
        a, b: Vectors to compare
        metric: Distance metric
        
    Returns:
        distance: Scalar distance value
    """
    if metric == "cosine":
        return 1.0 - float(
            cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0]
        )
    elif metric == "relative_l2":
        return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-9))
    else:  # 'l2'
        return float(np.linalg.norm(a - b))

def netlsd_saliency(graph, metric="cosine"):
    """
    Compute PSEUDO-saliency for NetLSD via node deletion.
    
    ALGORITHM:
    1. Compute base NetLSD signature for full graph
    2. For each node:
       a. Remove node from graph
       b. Recompute NetLSD signature
       c. Measure distance between base and new signature
       d. Distance = node saliency
    
    INTUITION:
    - Removing important node → Signature changes a lot
    - Removing unimportant node → Signature changes little
    - This is PSEUDO-saliency (approximate, not true gradient)
    
    METRICS:
    - Cosine distance: Measures direction change (best for spectral)
    - L2 distance: Measures magnitude change
    - Relative L2: Normalized magnitude change
    
    Args:
        graph: PyG Data object
        metric: Distance metric ("cosine", "relative_l2", "l2")
        
    Returns:
        saliency: Node saliency scores [num_nodes]
    """
    # Convert to NetworkX
    G_base = pyg_to_nx_labeled(graph)
    
    # Compute base signature
    base = netlsd_signature(G_base)
    
    # Compute saliency per node
    sal = []
    for n in range(G_base.number_of_nodes()):
        # Create copy and remove node
        G_del = G_base.copy()
        G_del.remove_node(n)
        G_del = reindex_contiguously(G_del)  # Fix node IDs
        
        # Compute new signature
        sig = netlsd_signature(G_del)
        
        # Measure distance (higher = more important node)
        sal.append(_dist(base, sig, metric=metric))
    
    return np.array(sal, dtype=float)

# ============================================================================
# GRAPH2VEC FAST PSEUDO-SALIENCY VIA DOC2VEC INFERENCE
# ============================================================================

def graph2vec_train_one(graph, dim=32, wl_iterations=2, epochs=8, seed=0):
    """
    Train Graph2Vec on a single graph.
    
    We train on just ONE graph to get a Doc2Vec model.
    Then we'll use this model's infer_vector() for fast saliency.
    
    Args:
        graph: PyG Data object
        dim: Embedding dimension
        wl_iterations: WL kernel depth
        epochs: Training epochs
        seed: Random seed
        
    Returns:
        model: Trained Graph2Vec model
    """
    G = pyg_to_nx_labeled(graph)
    model = Graph2Vec(
        dimensions=dim,
        wl_iterations=wl_iterations,
        epochs=epochs,
        seed=seed,
        workers=1,
        min_count=1  # Accept all substructures
    )
    model.fit([G])
    return model

def wl_infer_embedding(model: Graph2Vec, G: nx.Graph, wl_iterations=2):
    """
    Infer embedding for a modified graph using existing Doc2Vec model.
    
    Instead of retraining Graph2Vec from scratch (slow),
    we use the trained model's infer_vector() method (fast).
    
    ALGORITHM:
    1. Generate WL document for modified graph
    2. Use Doc2Vec.infer_vector() to get embedding
    3. This is approximate (not retrained), but much faster
    
    Args:
        model: Trained Graph2Vec model
        G: NetworkX graph (possibly modified)
        wl_iterations: Must match training
        
    Returns:
        embedding: Inferred embedding vector
    """
    # Ensure proper labeling
    G = nx_with_degree_labels(reindex_contiguously(G.copy()))
    
    # Generate WL document
    words = wl_document(G, wl_iterations=wl_iterations)
    
    # Infer embedding using existing Doc2Vec model
    vec = model.model.infer_vector(words, epochs=20)
    
    return np.asarray(vec, dtype=float)

def graph2vec_saliency_fast(graph, base_model: Graph2Vec, 
                            wl_iterations=2, metric="cosine"):
    """
    Compute PSEUDO-saliency for Graph2Vec via fast node deletion.
    
    ALGORITHM:
    1. Get base embedding from trained model
    2. For each node:
       a. Remove node from graph
       b. Infer embedding for modified graph (fast!)
       c. Measure distance between base and new embedding
       d. Distance = node saliency
    
    SPEED OPTIMIZATION:
    - Instead of retraining Graph2Vec n times (very slow)
    - We use infer_vector() which reuses the trained model
    - ~100x faster than retraining
    
    CAVEAT:
    - This is approximate (not exact retraining)
    - But empirically works well for saliency
    
    Args:
        graph: PyG Data object
        base_model: Pre-trained Graph2Vec model on this graph
        wl_iterations: Must match training
        metric: Distance metric
        
    Returns:
        saliency: Node saliency scores [num_nodes]
    """
    # Convert to NetworkX
    G_base = pyg_to_nx_labeled(graph)
    
    # Get base embedding from trained model
    base = base_model.get_embedding()[0]
    
    n = G_base.number_of_nodes()
    sal = np.zeros(n, dtype=float)
    
    # Compute saliency per node
    for node in range(n):
        # Create copy and remove node
        G_del = G_base.copy()
        G_del.remove_node(node)
        
        # Infer new embedding (fast!)
        emb_new = wl_infer_embedding(base_model, G_del, wl_iterations=wl_iterations)
        
        # Measure distance
        sal[node] = _dist(base, emb_new, metric=metric)
    
    return sal

# ============================================================================
# NORMALIZATION HELPERS
# ============================================================================

def normalize_saliency(sal, mode="minmax"):
    """
    Normalize saliency values for visualization and comparison.
    
    MODES:
    - "minmax": Scale to [0, 1] using (x - min) / (max - min)
      → Good for visualization (full color range)
      
    - "mean": Divide by mean: x / mean(x)
      → Preserves relative magnitudes
      
    - "none": No normalization
      → Raw values (not comparable across methods)
    
    WHY NORMALIZE?
    - Different methods produce different scales
    - GIN gradients might be in [0, 10]
    - NetLSD distances might be in [0, 0.5]
    - Normalization allows fair comparison
    
    Args:
        sal: Raw saliency values
        mode: Normalization mode
        
    Returns:
        Normalized saliency values
    """
    if mode == "minmax":
        s = sal.astype(float)
        return (s - s.min()) / (s.ptp() + 1e-9)
    elif mode == "mean":
        m = sal.mean() + 1e-9
        return sal / m
    else:  # 'none'
        return sal

# ============================================================================
# MAIN RUNNER
# ============================================================================

def run_saliency(datasets, methods, sample_k=6, seed=0,
                 # GIN parameters
                 gin_hidden=32, gin_epochs=15, gin_mode="grad",
                 # Graph2Vec parameters
                 g2v_dim=32, g2v_wl=2, g2v_epochs=8, g2v_metric="cosine",
                 # NetLSD parameters
                 netlsd_metric="cosine",
                 # Visualization parameters
                 topk_frac=0.0, norm_mode="minmax"):
    """
    Run complete saliency analysis across datasets and methods.
    
    WORKFLOW:
    For each dataset:
        1. Load dataset
        2. Train GIN model (if needed)
        3. Sample k graphs
        
        For each sampled graph:
            For each method:
                a. Compute raw saliency per node
                b. Normalize saliency
                c. Plot colored graph
                d. Save statistics
        
    4. Aggregate results
    5. Generate heatmap comparing methods
    
    PARAMETERS:
    - norm_mode: How to normalize saliency ('minmax' recommended)
    - g2v_metric/netlsd_metric: Distance metric for pseudo-saliency
    - topk_frac: Fraction of nodes to outline (e.g., 0.1 = top 10%)
    - gin_mode: 'grad' (gradient) or 'gradxinput' (gradient × input)
    
    Args:
        datasets: List of dataset names
        methods: List of methods (subset of ["GIN", "Graph2Vec", "NetLSD"])
        sample_k: Number of graphs per dataset to analyze
        seed: Random seed
        (see parameters above for others)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    summary_rows = []

    for ds_name in datasets:
        print(f"\n=== Dataset: {ds_name} ===")
        
        # Load dataset
        ds = TUDataset(root="data", name=ds_name)
        graphs = ensure_node_features([ds[i] for i in range(len(ds))])
        y_all = np.array([int(g.y) for g in graphs])
        num_classes = int(y_all.max()) + 1

        # ===== TRAIN GIN MODEL (if needed) =====
        gin_model = None
        if "GIN" in methods:
            print(f"Training GIN model for {ds_name}...")
            
            # Initialize model
            gin_model = GINSmall(
                graphs[0].x.size(1),
                hidden=gin_hidden,
                layers=3,
                n_classes=num_classes
            ).to(device)
            
            # Optimizer
            opt = torch.optim.Adam(
                gin_model.parameters(),
                lr=1e-3,
                weight_decay=1e-4
            )
            
            # Data loader
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
            print(f"GIN training complete.")

        # ===== SAMPLE GRAPHS =====
        k = min(sample_k, len(graphs))
        sample_ids = np.random.choice(len(graphs), k, replace=False)

        # ===== PROCESS EACH SAMPLED GRAPH =====
        for i in sample_ids:
            g = graphs[i]
            label = int(g.y)
            print(f"  -> graph {i} (class {label}, {g.num_nodes} nodes)")

            # Train Graph2Vec model for THIS specific graph (if needed)
            if "Graph2Vec" in methods:
                g2v_model = graph2vec_train_one(
                    g,
                    dim=g2v_dim,
                    wl_iterations=g2v_wl,
                    epochs=g2v_epochs,
                    seed=seed
                )

            # ===== COMPUTE SALIENCY PER METHOD =====
            for method in methods:
                print(f"     method: {method}")
                try:
                    # Compute raw saliency (method-specific)
                    if method == "GIN":
                        sal_raw = gin_saliency(
                            gin_model, g, device, mode=gin_mode
                        )
                    
                    elif method == "Graph2Vec":
                        sal_raw = graph2vec_saliency_fast(
                            g, g2v_model,
                            wl_iterations=g2v_wl,
                            metric=g2v_metric
                        )
                    
                    elif method == "NetLSD":
                        sal_raw = netlsd_saliency(g, metric=netlsd_metric)
                        
                        # Guard against size mismatch
                        if len(sal_raw) != g.num_nodes:
                            sal_raw = np.pad(
                                sal_raw,
                                (0, max(0, g.num_nodes - len(sal_raw))),
                                constant_values=0.0
                            )[:g.num_nodes]
                    
                    else:
                        continue

                    # ===== NORMALIZE SALIENCY =====
                    # Normalize to [0,1] for visualization and comparison
                    sal_norm = normalize_saliency(sal_raw, mode=norm_mode)

                    # ===== GENERATE VISUALIZATIONS =====
                    # Plot colored graph
                    plot_graph_saliency(
                        g, sal_norm,
                        ds_name, method, i,
                        topk_frac=topk_frac,
                        add_colorbar=True
                    )
                    
                    # Plot mean saliency bar
                    plot_mean_saliency_bar(
                        float(np.mean(sal_norm)),
                        ds_name, method, i
                    )

                    # ===== SAVE STATISTICS =====
                    # Store both raw and normalized summaries
                    summary_rows.append({
                        "dataset": ds_name,
                        "method": method,
                        "graph_id": int(i),
                        "label": int(label),
                        "num_nodes": int(g.num_nodes),
                        # Raw statistics (before normalization)
                        "mean_saliency_raw": float(np.mean(sal_raw)),
                        "max_saliency_raw": float(np.max(sal_raw)),
                        "std_saliency_raw": float(np.std(sal_raw)),
                        # Normalized statistics (after normalization)
                        "mean_saliency_norm": float(np.mean(sal_norm)),
                        "max_saliency_norm": float(np.max(sal_norm)),
                        "std_saliency_norm": float(np.std(sal_norm)),
                    })

                except Exception as e:
                    print(f"     [warn] saliency failed ({method}, graph {i}): {e}")
                    # Record failure to keep table consistent
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

    # ===== SAVE SUMMARY CSV =====
    import pandas as pd
    df = pd.DataFrame(summary_rows)
    out_csv = os.path.join(OUT_DIR_TAB, "saliency_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Saved saliency summary to: {out_csv}")

    # ===== GENERATE COMPARISON HEATMAP =====
    # Compare average normalized saliency across datasets and methods
    try:
        # Pivot table: dataset × method → mean of normalized mean-saliency
        pivot = df.pivot_table(
            values="mean_saliency_norm",
            index="dataset",
            columns="method",
            aggfunc="mean"
        )
        
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(7, 4.2))
        
        # Heatmap with annotations
        sns.heatmap(
            pivot,
            annot=True,
            cmap="YlGnBu",
            fmt=".3f",
            vmin=0.0,
            vmax=1.0,
            ax=ax
        )
        
        ax.set_title("Average Mean-Node Saliency (normalized, dataset × method)")
        fig.tight_layout()
        
        heat_path = os.path.join(OUT_DIR_FIG, "saliency_mean_heatmap.png")
        fig.savefig(heat_path, dpi=180)
        plt.close(fig)
        print(f"✅ Saved heatmap to: {heat_path}")
    
    except Exception as e:
        print(f"[warn] heatmap failed: {e}")

# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Task 5: Graph saliency for GIN, Graph2Vec, NetLSD "
                    "across datasets (normalized for comparison)."
    )
    
    # Dataset and method selection
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["MUTAG", "ENZYMES", "IMDB-MULTI"],
        help="Datasets to analyze"
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=["GIN", "Graph2Vec", "NetLSD"],
        help="Methods to compute saliency for"
    )
    p.add_argument(
        "--sample_k",
        type=int,
        default=6,
        help="Number of graphs per dataset to visualize"
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility"
    )

    # GIN-specific parameters
    p.add_argument(
        "--gin_hidden",
        type=int,
        default=32,
        help="Hidden dimension for GIN encoder"
    )
    p.add_argument(
        "--gin_epochs",
        type=int,
        default=15,
        help="Training epochs for GIN"
    )
    p.add_argument(
        "--gin_mode",
        choices=["grad", "gradxinput"],
        default="grad",
        help="Saliency mode: 'grad' (gradient) or 'gradxinput' (gradient×input)"
    )

    # Graph2Vec-specific parameters
    p.add_argument(
        "--g2v_dim",
        type=int,
        default=32,
        help="Embedding dimension for Graph2Vec"
    )
    p.add_argument(
        "--g2v_wl",
        type=int,
        default=2,
        help="WL iterations for Graph2Vec"
    )
    p.add_argument(
        "--g2v_epochs",
        type=int,
        default=8,
        help="Training epochs for Graph2Vec"
    )
    p.add_argument(
        "--g2v_metric",
        choices=["cosine", "relative_l2", "l2"],
        default="cosine",
        help="Distance metric for Graph2Vec saliency"
    )

    # NetLSD-specific parameters
    p.add_argument(
        "--netlsd_metric",
        choices=["cosine", "relative_l2", "l2"],
        default="cosine",
        help="Distance metric for NetLSD saliency"
    )

    # Visualization and aggregation parameters
    p.add_argument(
        "--topk_frac",
        type=float,
        default=0.0,
        help="Outline top-k fraction of salient nodes (e.g., 0.1 for top 10%%)"
    )
    p.add_argument(
        "--norm_mode",
        choices=["minmax", "mean", "none"],
        default="minmax",
        help="Per-graph normalization mode for cross-method comparison"
    )
    
    return p.parse_args()

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Main execution function.
    
    Workflow:
    1. Parse command-line arguments
    2. Run saliency analysis
    3. Generate visualizations
    4. Save summary statistics
    """
    args = parse_args()
    
    print("="*70)
    print("GRAPH EMBEDDING EXPLAINABILITY & SALIENCY ANALYSIS")
    print("="*70)
    print(f"Datasets: {args.datasets}")
    print(f"Methods: {args.methods}")
    print(f"Graphs per dataset: {args.sample_k}")
    print(f"Normalization: {args.norm_mode}")
    print("="*70)
    
    # Run saliency analysis
    run_saliency(
        datasets=args.datasets,
        methods=args.methods,
        sample_k=args.sample_k,
        seed=args.seed,
        # GIN parameters
        gin_hidden=args.gin_hidden,
        gin_epochs=args.gin_epochs,
        gin_mode=args.gin_mode,
        # Graph2Vec parameters
        g2v_dim=args.g2v_dim,
        g2v_wl=args.g2v_wl,
        g2v_epochs=args.g2v_epochs,
        g2v_metric=args.g2v_metric,
        # NetLSD parameters
        netlsd_metric=args.netlsd_metric,
        # Visualization parameters
        topk_frac=args.topk_frac,
        norm_mode=args.norm_mode,
    )
    
    print("\n" + "="*70)
    print("SALIENCY ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  - Colored graphs: {OUT_DIR_FIG}/")
    print(f"  - Summary table: {OUT_DIR_TAB}/saliency_summary.csv")
    print(f"  - Comparison heatmap: {OUT_DIR_FIG}/saliency_mean_heatmap.png")
    print(f"\nVisualization files:")
    print(f"  - {{dataset}}_{{method}}_graph{{id}}.png : Colored by saliency")
    print(f"  - {{dataset}}_{{method}}_graph{{id}}_bar.png : Mean saliency bar")
    print("\n✅ All saliency figures saved!")

if __name__ == "__main__":
    main()


# ============================================================================
# USAGE EXAMPLES & INTERPRETATION GUIDE
# ============================================================================

"""
BASIC USAGE:
-----------
# Quick test with defaults
python final_task5.py

# Custom configuration
python final_task5.py \
  --datasets MUTAG ENZYMES \
  --methods GIN Graph2Vec \
  --sample_k 10

# Outline top 20% most salient nodes in red
python final_task5.py \
  --topk_frac 0.2


COMPLETE EVALUATION:
-------------------
python final_task5.py \
  --datasets MUTAG ENZYMES IMDB-MULTI \
  --methods GIN Graph2Vec NetLSD \
  --sample_k 12 \
  --gin_epochs 30 \
  --topk_frac 0.15 \
  --norm_mode minmax


METHOD-SPECIFIC TUNING:
----------------------

GIN:
  --gin_hidden 64        # Larger hidden dimension
  --gin_epochs 30        # More training
  --gin_mode gradxinput  # Gradient × Input (like Grad-CAM)

Graph2Vec:
  --g2v_dim 64           # Larger embedding
  --g2v_wl 3             # Deeper WL kernel
  --g2v_metric relative_l2  # Different distance metric

NetLSD:
  --netlsd_metric l2     # Absolute distance instead of cosine


INTERPRETING VISUALIZATIONS:
---------------------------

Graph Coloring:
  - Blue nodes: Low saliency (less important)
  - Green nodes: Medium saliency
  - Yellow nodes: High saliency (most important)
  
Red Outlines (if --topk_frac > 0):
  - Highlight the most salient nodes
  - Example: --topk_frac 0.1 outlines top 10%

Bar Charts:
  - Show average saliency per graph
  - Higher bar = Graph has more important nodes overall


INTERPRETING SALIENCY PATTERNS:
-------------------------------

HIGH SALIENCY NODES (Yellow):
  - Central hubs in social networks
  - Functional groups in molecules
  - Key residues in proteins
  - Bottleneck nodes in infrastructure

LOW SALIENCY NODES (Blue):
  - Peripheral nodes
  - Redundant connections
  - Background structure

UNIFORM SALIENCY (All similar colors):
  - Homogeneous graph structure
  - All nodes equally important
  - Or: Method doesn't discriminate well


NORMALIZATION IMPORTANCE:
------------------------

WHY normalize?
  - GIN gradients: Range [0, 100]
  - Graph2Vec distances: Range [0, 2]
  - NetLSD distances: Range [0, 0.5]
  → Cannot compare directly!

After normalization [0, 1]:
  - All methods on same scale
  - Can aggregate across methods
  - Can compare saliency distributions

Modes:
  - "minmax": Best for visualization (full color range)
  - "mean": Best for statistics (preserves ratios)
  - "none": Best for debugging (raw values)


TROUBLESHOOTING:
---------------

1. All nodes same color:
   → Try different normalization mode
   → Method might not discriminate for this graph
   → Increase sample size

2. GIN saliency is very noisy:
   → Train longer (--gin_epochs 50)
   → Use gradxinput mode instead of grad
   → Increase hidden dimension

3. Graph2Vec fails on some graphs:
   → Graphs too small (< 5 nodes)
   → Increase --g2v_epochs
   → Check min_count parameter

4. NetLSD very slow:
   → Expected for large graphs (eigendecomposition)
   → Sample fewer graphs
   → Skip NetLSD for large datasets


PERFORMANCE TIPS:
----------------

Fast mode (for testing):
  --sample_k 3
  --gin_epochs 10
  --g2v_epochs 5

Quality mode (for publication):
  --sample_k 12
  --gin_epochs 30
  --g2v_epochs 20
  --topk_frac 0.1

GPU acceleration (GIN only):
  - Automatically uses CUDA if available
  - 5-10x speedup for GIN training


RESEARCH APPLICATIONS:
---------------------

Drug Discovery:
  "Which atoms are critical for drug activity?"
  → High-saliency atoms = pharmacophore

Social Networks:
  "Who are the key influencers?"
  → High-saliency nodes = important users

Protein Analysis:
  "Which residues define protein function?"
  → High-saliency residues = active site

Infrastructure:
  "Which roads are critical for traffic flow?"
  → High-saliency nodes = bottlenecks


KEY OUTPUTS:
-----------

1. Colored Graph Visualizations:
   - Visual inspection of node importance
   - Identify important substructures
   
2. Summary CSV:
   - Quantitative saliency statistics
   - Compare across graphs and methods
   
3. Comparison Heatmap:
   - Overall method behavior
   - Dataset-specific patterns

4. Bar Charts:
   - Quick summary per graph
   - Spot anomalies


SCIENTIFIC VALIDATION:
---------------------

To validate saliency is meaningful:

1. Perturbation Test:
   Remove high-saliency nodes → Embedding should change more
   
2. Correlation Analysis:
   Saliency vs graph properties (degree, betweenness)
   
3. Classification Impact:
   Mask high-saliency nodes → Accuracy should drop
   
4. Domain Expert Validation:
   "Do highlighted structures make sense?"
   
5. Consistency Check:
   Similar graphs should have similar saliency patterns


LIMITATIONS:
-----------

1. Saliency ≠ Causality
   - High saliency = correlation, not causation
   - Node important for embedding ≠ important for task
   
2. Method-Specific Biases
   - GIN: Focuses on features used by neural network
   - Graph2Vec: Focuses on frequent substructures
   - NetLSD: Focuses on structural centrality
   
3. Visualization Challenges
   - Large graphs hard to visualize
   - Overlapping nodes obscure colors
   - Need domain knowledge to interpret



"""