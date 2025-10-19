# Evaluation of Graph Embedding Methods on Real-World Datasets
The first step is to install **Miniconda** and create a virtual enviroment for accomodating the whole project.
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh # get the latest version for Linux
bash Miniconda3-latest-Linux-x86_64.sh # install miniconda
conda --version # check the correctness of the installation
conda create --name myenv python=3.9 # create an a venv with name my env and python version 3.9
conda activate myenv # activate conda enviroment to currently work on
conda install numpy pandas matplotlib # install some basic python libraries
conda deactivate myenv # exit the myenv enviroment
```
Then in order to create an enviroment with the necessary python modules we created an environment.yml file, we executed:
``` bash
# 1) Create the env
conda env create -f environment.yml
conda activate graphbench
# 2) (Optional) upgrade pip
python -m pip install -U pip
# 3) If an error occured due to mismatch of sympy please do this
pip install "sympy==1.13.1"
# 4) Install PyG + companions for cpu only
# set the variable automatically
TV=$(python -c "import torch; print(torch.__version__.split('+')[0])")
pip install -U fsspec torch-geometric
pip install -U pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-${TV}+cpu.html
```
# Task A) Classification 
In order to run the script for the first task, please run the following:
```bash
python final_task1.py \
  --datasets ENZYMES MUTAG IMDB-MULTI \
  --methods graph2vec netlsd \
  --dims 32 64 \
  --seeds 0 1 2 \
  --test_size 0.2 \
  --out report/tables/classification_eval.csv \
  --embed_plots first_seed
```
| **Argument** | **Example Value** | **Description** |
|---------------|------------------|-----------------|
| `--datasets` | `ENZYMES MUTAG IMDB-MULTI` | The **graph datasets** to run experiments on. You can include one or multiple datasets from the [TUDataset](https://chrsmrrs.github.io/datasets/docs/datasets/) collection used by PyTorch Geometric.<br> |
| `--methods` | `graph2vec netlsd` | The **unsupervised graph embedding algorithms** to use.<br>• **graph2vec**: learns embeddings by aggregating graph substructures.<br>• **netlsd**: computes embeddings from Laplacian spectral signatures. |
| `--dims` | `32 64` | The **embedding dimensions** (vector lengths) to evaluate. You can specify one or several values.<br> |
| `--seeds` | `0 1 2` | Random seeds for reproducibility. Each seed gives a different train/test split and initialization, allowing you to compute **mean ± std** performance metrics. |
| `--test_size` | `0.2` | Fraction of the dataset reserved for **testing** (20% test, 80% train by default). Can be changed (e.g. `0.3` for 30% test). |
| `--out` | `report/tables/classification_eval.csv` | Path to save the **per-run CSV** results. Contains all metrics (accuracy, F1, AUC) and resource usage per (dataset, method, dimension, seed). |
| `--embed_plots` | `first_seed` | Controls how many **embedding visualizations** (t-SNE, UMAP) are generated.<br>Options:<br>• `none` → skip all embedding plots <br>• `first_seed` → only plot for the first random seed <br>• `all` → plot for every seed. |

# Task B) Clustering 
In order to run the script of the second task, please run the following:

```bash
python3 final_task2.py \
  --datasets MUTAG ENZYMES IMDB-MULTI \
  --methods graph2vec netlsd gin \
  --dims 32 64 128 \
  --seeds 0 1 2 \
  --plot_policy first_seed \
  --gin_hidden 64 \
  --gin_layers 3 \
  --gin_dropout 0.2 \
  --gin_epochs 30 \
  --gin_batch 64 \
  --device cpu
```

| **Argument** | **Example Value** | **Description** |
|---------------|------------------|-----------------|
| `--datasets` | `MUTAG ENZYMES IMDB-MULTI` | The **graph datasets** to run clustering on. You can include one or multiple datasets from the TUDataset collection. |
| `--methods` | `graph2vec netlsd gin` | The **embedding methods** to compute graph representations.<br>• **graph2vec**: unsupervised embedding using substructure aggregation.<br>• **netlsd**: Laplacian spectral descriptor-based embedding.<br>• **gin**: supervised Graph Isomorphism Network (GIN) trained as an encoder. |
| `--dims` | `32 64 128` | The **embedding dimensions** (vector lengths) to evaluate. You can include multiple values to compare performance. |
| `--seeds` | `0 1 2` | Random seeds for reproducibility — used for random splits, initialization, and model training. |
| `--plot_policy` | `first_seed` | Controls which runs generate **t-SNE/UMAP plots** of the embeddings.<br>Options:<br>• `none` → no plots (fastest)<br>• `first_seed` → plots only for the first random seed (default)<br>• `all` → plots for all seeds (slowest) |
| `--gin_hidden` | `64` | Number of hidden units per layer in the **GIN encoder**. |
| `--gin_layers` | `3` | Number of **GIN layers** used in the encoder network. |
| `--gin_dropout` | `0.2` | Dropout probability applied between GIN layers during training. |
| `--gin_epochs` | `30` | Number of training epochs for the GIN encoder. |
| `--gin_batch` | `64` | Batch size used when training the GIN encoder. |
| `--device` | `cpu` | Compute device to use (`cpu` or `cuda`). Defaults to CPU; set to `cuda` for GPU acceleration. |

# Task C) Stability Analysis
In order run the code for the third task, please run the following:
```bash
python3 final_task3.py \
  --datasets MUTAG ENZYMES IMDB-MULTI \
  --methods graph2vec netlsd gin \
  --dims 32 64 \
  --seeds 0 1 \
  --levels_edges 0.5 1.0 \
  --levels_attrs 0.5 1.0
```

| **Argument** | **Example Value** | **Description** |
|---------------|------------------|-----------------|
| `--datasets` | `MUTAG ENZYMES IMDB-MULTI` | The **graph datasets** to run stability analysis on. You can specify one or multiple datasets from the [TUDataset](https://chrsmrrs.github.io/datasets/docs/datasets/) collection. |
| `--methods` | `graph2vec netlsd gin` | The **embedding methods** to compute graph representations.<br>• **graph2vec** — substructure-based unsupervised embeddings.<br>• **netlsd** — Laplacian spectral descriptor embeddings.<br>• **gin** — supervised Graph Isomorphism Network encoder (hidden = dim). |
| `--dims` | `32 64` | The **embedding dimensions** (vector lengths) to test. You can specify one or more dimensions. |
| `--seeds` | `0 1` | Random seeds for reproducibility — used for random splits, initialization, and perturbation randomness. |
| `--levels_edges` | `0.5 1.0` | List of **edge perturbation levels** (fractions of edges to add/remove). Example: `0.5` means ~50% of edges will be altered. |
| `--levels_attrs` | `0.5 1.0` | List of **attribute perturbation levels** (fractions of node features to shuffle). Example: `1.0` means all node features are shuffled. |

# Extra Task 1) Cross-Dataset Transferability
In order to run the first extra task code, please run the following:
```bash
python3 final_task4.py \
  --datasets MUTAG ENZYMES IMDB-MULTI \
  --methods graph2vec netlsd gin \
  --dims 32 64 \
  --seeds 0
```
| **Argument** | **Example Value** | **Description** |
|---------------|------------------|-----------------|
| `--datasets` | `MUTAG ENZYMES IMDB-MULTI` | The **graph datasets** used as both sources and targets for transfer evaluation. You can include one or multiple datasets from the [TUDataset](https://chrsmrrs.github.io/datasets/docs/datasets/). |
| `--methods` | `graph2vec netlsd gin` | The **graph embedding methods** to evaluate.<br>• **graph2vec** — unsupervised substructure-based embeddings.<br>• **netlsd** — Laplacian spectral descriptor embeddings.<br>• **gin** — supervised Graph Isomorphism Network encoder. |
| `--dims` | `32 64` | The **embedding dimensions** to evaluate. Each dimension controls the vector length of the graph embeddings. |
| `--seeds` | `0` | Random seeds for reproducibility (used for random initialization, splits, and embedding randomness). You can specify multiple seeds, e.g. `0 1 2`. |

# Extra Task 2) Graph Embeddings Explainability and Attention
In order to run the second extra task, please run the following:
```bash
python final_task5.py \
  --datasets MUTAG ENZYMES IMDB-MULTI \
  --methods GIN Graph2Vec NetLSD \
  --sample_k 6 \
  --seed 0 \
  --gin_hidden 32 \
  --gin_epochs 15 \
  --gin_mode grad \
  --g2v_dim 32 \
  --g2v_wl 2 \
  --g2v_epochs 8 \
  --g2v_metric cosine \
  --netlsd_metric cosine \
  --topk_frac 0.0 \
  --norm_mode minmax
```

| **Argument** | **Type** | **Default** | **Description** |
|---------------|-----------|--------------|------------------|
| `--datasets` | list | `["MUTAG", "ENZYMES", "IMDB-MULTI"]` | Names of TUDatasets to process. |
| `--methods` | list | `["GIN", "Graph2Vec", "NetLSD"]` | Graph embedding methods to compute saliency. |
| `--sample_k` | int | `6` | Number of graphs per dataset to visualize and analyze. |
| `--seed` | int | `0` | Random seed for reproducibility. |
| `--gin_hidden` | int | `32` | Hidden layer dimension for the GIN model. |
| `--gin_epochs` | int | `15` | Number of epochs to train the GIN model. |
| `--gin_mode` | str | `"grad"` | Type of gradient-based saliency: `"grad"` or `"gradxinput"`. |
| `--g2v_dim` | int | `32` | Embedding dimension for Graph2Vec. |
| `--g2v_wl` | int | `2` | Number of Weisfeiler-Lehman iterations in Graph2Vec. |
| `--g2v_epochs` | int | `8` | Number of training epochs for Graph2Vec model. |
| `--g2v_metric` | str | `"cosine"` | Distance metric for Graph2Vec saliency (`"cosine"`, `"relative_l2"`, `"l2"`). |
| `--netlsd_metric` | str | `"cosine"` | Distance metric for NetLSD saliency. |
| `--topk_frac` | float | `0.0` | Fraction of most salient nodes to outline in red (e.g., `0.1` = top 10%). |
| `--norm_mode` | str | `"minmax"` | Normalization type for saliency values (`"minmax"`, `"mean"`, `"none"`). |

# Extra Task 3) Causal Node Importance
In order to run the third extra task, please run the following: 
```bash
python3 counterfactual_graph_explanations.py \
  --datasets MUTAG ENZYMES IMDB-MULTI \
  --seed 0
```
*(All other model parameters such as GIN hidden size and epochs are defined internally.)*
| **Argument** | **Type** | **Default** | **Description** |
|---------------|-----------|--------------|------------------|
| `--datasets` | list | `["MUTAG", "ENZYMES", "IMDB-MULTI"]` | Names of TUDatasets to analyze for counterfactual explanations. |
| `--seed` | int | `0` | Random seed for reproducibility of model training and saliency computation. |
