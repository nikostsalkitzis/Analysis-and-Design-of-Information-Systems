### Understanding the Datasets for EMB1

The datasets in EMB1 (MUTAG, ENZYMES, and IMDB-MULTI) are standard benchmarks from the TUDataset collection, commonly used to evaluate graph embedding and classification algorithms in machine learning. They represent different domains and graph characteristics, allowing you to test how well methods like Graph2Vec, NetLSD, and GIN handle varying sizes, features, and complexities. Below, I'll provide a comparison table of key statistics, followed by detailed explanations for each dataset.

#### Comparison of Dataset Statistics

| Dataset      | Domain                  | Number of Graphs | Number of Classes | Average Nodes per Graph | Average Edges per Graph | Node Features/Attributes                  | Edge Features/Attributes          |
|--------------|-------------------------|------------------|-------------------|-------------------------|-------------------------|-------------------------------------------|-----------------------------------|
| MUTAG       | Small molecules (bioinformatics/chemistry) | 188             | 2 (binary)       | 17.93                  | 19.79                  | Categorical labels (e.g., atom types); no continuous attributes | Categorical labels (e.g., bond types); no continuous attributes |
| ENZYMES     | Bioinformatics (proteins/enzymes) | 600             | 6 (multi-class)  | 32.63                  | 62.14                  | Categorical labels (e.g., amino acid types); 18-dimensional continuous attributes (e.g., physicochemical properties) | None                              |
| IMDB-MULTI  | Social networks (movie collaborations) | 1500            | 3 (multi-class)  | 13.00                  | 65.94                  | None                                      | None                              |

These stats highlight differences: MUTAG is small and feature-rich for chemistry; ENZYMES is medium-sized with detailed node info for biology; IMDB-MULTI is larger and structure-only for social analysis. When implementing your project, note that datasets without features (like IMDB-MULTI) rely purely on graph topology, which might favor certain embedding methods.




#### MUTAG
This dataset consists of 188 graphs representing nitroaromatic and heteroaromatic compounds, focused on predicting mutagenicity (a property related to whether a chemical can cause mutations in DNA). It's a binary classification task: graphs are labeled as mutagenic (positive) or non-mutagenic (negative). Each graph models a molecule, where nodes represent atoms (with categorical labels indicating atom types, such as carbon, oxygen, etc.) and edges represent chemical bonds (with categorical labels for bond types, like single, double, or aromatic). There are no continuous features on nodes or edges, so embeddings must capture structural patterns effectively.

MUTAG is compact and straightforward, making it ideal for quick experiments in your project. Its small size (average ~18 nodes and ~20 edges per graph) means low computational demands, but the chemical domain tests how well embeddings preserve molecular topology. Common sources cite it from references like [1,23] in the TUDataset docs.

#### ENZYMES
ENZYMES includes 600 graphs derived from protein tertiary structures, aimed at classifying enzymes into one of 6 functional categories (based on the Enzyme Commission hierarchy, e.g., hydrolases, transferases). This is a multi-class problem. Graphs model proteins, with nodes typically representing secondary structure elements (like alpha helices or beta sheets) or amino acids (categorical labels for types), connected by edges indicating spatial proximity or interactions (no edge labels or features). Uniquely, nodes have 18-dimensional continuous attributes, which might include properties like hydrophobicity, charge, or other physicochemical metrics.

With moderate size (average ~33 nodes and ~62 edges), it's suitable for testing scalability in your benchmarks. The bioinformatics domain emphasizes biological relevance, and methods like GIN (which handle node features well) might perform strongly here. It's sourced from references [4,5] in the dataset collection.

#### IMDB-MULTI
This dataset features 1500 graphs from the Internet Movie Database (IMDB), representing collaboration networks in movies for multi-class classification into 3 genres (typically action, romance, and sci-fi). Each graph is an ego-network centered on a director or genre, where nodes are actors and edges indicate co-appearances in films. Notably, there are no node or edge features—purely structural data—so embeddings must rely on topology alone (e.g., connectivity patterns like cliques for tight collaborations).

It's the largest in EMB1 (average ~13 nodes but denser with ~66 edges), which could highlight efficiency differences among your methods, especially on clustering or classification tasks. The social network domain contrasts with the scientific ones, testing generalizability. Sourced from reference [14].

If you're loading these in code (e.g., via PyTorch Geometric), use `TUDataset(root='data', name='MUTAG')` and similar— they'll come pre-split for train/test. Let me know if you need help with code setup or the embedding methods next!