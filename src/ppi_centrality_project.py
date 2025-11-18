#!/usr/bin/env python3
"""
ppi_centrality_project.py
Compute centrality measures on a PPI graph (synthetic or from file).
Outputs centralities.csv and several PNG plots into an output directory.

Usage:
  # synthetic demo
  python src/ppi_centrality_project.py --synthetic --out output

  # using a file (tab-separated)
  python src/ppi_centrality_project.py --input data/yourfile.tsv --sep "\t" --colA "Official Symbol Interactor A" --colB "Official Symbol Interactor B" --out output
"""
import argparse
import os
import sys
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_edge_list(path: str, sep: str = "\t", colA: str = None, colB: str = None) -> pd.DataFrame:
    df_raw = pd.read_csv(path, sep=sep, header=0, dtype=str, low_memory=False)
    cols_map = {c.strip(): c for c in df_raw.columns}
    if colA and colB:
        if colA not in cols_map or colB not in cols_map:
            raise ValueError(f"Columns not found. Available: {list(df_raw.columns)}")
        df = df_raw[[cols_map[colA], cols_map[colB]]].copy()
    else:
        # pick first two columns
        df = df_raw.iloc[:, :2].copy()
    df.columns = ["A", "B"]
    df = df.dropna(subset=["A","B"])
    df = df[df["A"].astype(str) != df["B"].astype(str)]
    df = df.drop_duplicates()
    return df

def build_graph_from_edges(df_edges: pd.DataFrame) -> nx.Graph:
    G = nx.from_pandas_edgelist(df_edges, "A", "B")
    G.remove_nodes_from(list(nx.isolates(G)))
    return G

def generate_synthetic_graph(n_nodes: int = 200, m: int = 3) -> nx.Graph:
    G = nx.barabasi_albert_graph(n_nodes, m, seed=42)
    mapping = {node: f"P{node}" for node in G.nodes()}
    return nx.relabel_nodes(G, mapping)

def compute_centralities(G: nx.Graph) -> pd.DataFrame:
    print("Computing centralities ...")
    deg = nx.degree_centrality(G)
    bet = nx.betweenness_centrality(G, normalized=True)
    cls = nx.closeness_centrality(G)
    try:
        eig = nx.eigenvector_centrality(G, max_iter=1000)
    except Exception:
        eig = nx.eigenvector_centrality_numpy(G)
    try:
        pr = nx.pagerank(G)
    except Exception:
        # fallback simple pagerank if scipy backend missing
        pr = nx.pagerank_numpy(G)
    df = pd.DataFrame({
        "degree": pd.Series(deg),
        "betweenness": pd.Series(bet),
        "closeness": pd.Series(cls),
        "eigenvector": pd.Series(eig),
        "pagerank": pd.Series(pr)
    })
    return df.sort_values(by="degree", ascending=False)

def save_plots(df: pd.DataFrame, G: nx.Graph, out_dir: str) -> dict:
    plots = {}
    # degree distribution
    plt.figure(figsize=(6,4))
    plt.hist(df["degree"], bins=30)
    plt.title("Degree Centrality Distribution")
    plt.xlabel("Degree centrality")
    plt.ylabel("Frequency")
    p1 = os.path.join(out_dir, "degree_distribution.png")
    plt.tight_layout(); plt.savefig(p1); plt.close()
    plots["degree_distribution"] = p1

    # betweenness distribution
    plt.figure(figsize=(6,4))
    plt.hist(df["betweenness"], bins=30)
    plt.title("Betweenness Centrality Distribution")
    plt.xlabel("Betweenness")
    plt.ylabel("Frequency")
    p2 = os.path.join(out_dir, "betweenness_distribution.png")
    plt.tight_layout(); plt.savefig(p2); plt.close()
    plots["betweenness_distribution"] = p2

    # degree vs betweenness
    plt.figure(figsize=(6,4))
    plt.scatter(df["degree"], df["betweenness"], s=12)
    plt.title("Degree vs Betweenness")
    plt.xlabel("Degree")
    plt.ylabel("Betweenness")
    p3 = os.path.join(out_dir, "degree_vs_betweenness.png")
    plt.tight_layout(); plt.savefig(p3); plt.close()
    plots["degree_vs_betweenness"] = p3

    # network visualization (spring layout)
    plt.figure(figsize=(6,6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, node_size=20, edge_color="gray", linewidths=0.2)
    plt.title("Network visualization (spring layout)")
    p4 = os.path.join(out_dir, "network_visualization.png")
    plt.tight_layout(); plt.savefig(p4); plt.close()
    plots["network_visualization"] = p4

    return plots

def save_csv(df: pd.DataFrame, out_path: str):
    df.to_csv(out_path, index_label="protein")
    print(f"Wrote centralities CSV to: {out_path}")

def topk_print(df: pd.DataFrame, k: int = 10):
    print(f"\nTop {k} proteins by centrality:")
    for col in df.columns:
        topk = df[col].nlargest(k).index.tolist()
        print(f"{col}: {', '.join(map(str, topk))}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", type=str, default=None)
    p.add_argument("--sep", type=str, default="\t")
    p.add_argument("--colA", type=str, default=None)
    p.add_argument("--colB", type=str, default=None)
    p.add_argument("--out", "-o", type=str, default="output")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--nodes", type=int, default=200)
    p.add_argument("--m", type=int, default=3)
    p.add_argument("--topk", type=int, default=10)
    return p.parse_args()

def main():
    args = parse_args()
    ensure_dir(args.out)

    if args.input and not args.synthetic:
        print(f"Loading edges from: {args.input}")
        df_edges = load_edge_list(args.input, sep=args.sep, colA=args.colA, colB=args.colB)
        G = build_graph_from_edges(df_edges)
        print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    else:
        print("Generating synthetic PPI-like graph.")
        G = generate_synthetic_graph(n_nodes=args.nodes, m=args.m)
        print(f"Synthetic graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    df = compute_centralities(G)
    csv_path = os.path.join(args.out, "centralities.csv")
    save_csv(df, csv_path)
    plots = save_plots(df, G, args.out)
    print("Saved plots:", plots)
    topk_print(df, k=args.topk)
    print("\nDone. Check the output/ folder for results.")

if __name__ == "__main__":
    main()
