# ppi-network-analysis
Protein–Protein Interaction (PPI) Network Analysis
Using Graph Centrality Algorithms in Python

This project analyzes Protein–Protein Interaction (PPI) networks using graph-based centrality algorithms. The goal is to identify hub proteins, understand interaction patterns, and explore the structure of biological networks using computational methods.

Overview

PPI networks help reveal how proteins collaborate within a biological system.
Using NetworkX, this project computes several centrality measures:

Degree Centrality — identifies highly connected proteins

Betweenness Centrality — finds proteins acting as bridges

Closeness Centrality — measures how quickly a protein can reach others

Eigenvector Centrality — detects influential proteins

PageRank — evaluates global importance

The script works with both synthetic PPI-like networks and real datasets.

How to Run the Project
1. Install dependencies
pip install -r requirements.txt

2. Run with a synthetic network
python src/ppi_centrality_project.py --synthetic --out output

3. Run using a real dataset

Place your file inside the data/ folder, then run:

python src/ppi_centrality_project.py --input data/yourfile.tsv --sep "\t"

Output

All results are saved in the output/ directory:

centralities.csv — ranked centrality scores

degree_distribution.png

betweenness_distribution.png

degree_vs_betweenness.png

network_visualization.png

These outputs help visualize the network and identify key proteins.

📁 Project Structure
ppi-network-analysis/
│
├── src/
│   └── ppi_centrality_project.py
├── data/
├── output/
├── README.md
└── requirements.txt

 Applications

Identifying hub or essential proteins

Understanding biological pathways

Systems biology research

Network modeling and graph theory analysis
