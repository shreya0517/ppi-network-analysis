Protein–Protein Interaction (PPI) Network Analysis Using Graph Centrality

This project analyzes a Protein–Protein Interaction (PPI) network using graph-based centrality algorithms. The goal is to identify hub proteins, understand interaction patterns, and evaluate protein importance within biological networks.


Project Overview

The script constructs a PPI-like network (synthetic or using a real dataset) and computes the following centrality measures:

Degree Centrality

Betweenness Centrality

Closeness Centrality

Eigenvector Centrality

PageRank

These metrics help identify influential or biologically important proteins, typically involved in essential pathways or regulatory processes.


How to Run the Project
1. Synthetic Network (demo mode)

Generates a scale-free graph resembling real PPI networks.

python src/ppi_centrality_project.py --synthetic --out output

2. Using a Real Dataset

Place your file inside the data/ folder, then run:

python src/ppi_centrality_project.py --input data/your_file.tsv --sep "\t"


Outputs Generated

All results are saved in the output/ directory:

centralities.csv – centrality scores for each protein

degree_distribution.png

betweenness_distribution.png

degree_vs_betweenness.png

network_visualization.png

These outputs provide a clear view of node importance and overall network structure.



Requirements

Install all dependencies:

pip install -r requirements.txt


Project Structure
ppi-network-analysis/
│
├── src/
│   └── ppi_centrality_project.py
├── data/
├── output/
├── README.md
└── requirements.txt


Key Applications

This analysis can support research in:

Biological network modeling

Hub protein identification

Drug target prediction

Systems biology

Graph-based data mining