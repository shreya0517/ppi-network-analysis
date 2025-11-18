#!/usr/bin/env python3
"""
ppi_centrality_project.py

Usage examples:
# 1) Run with your edge list (tab-separated, header or no header):
python ppi_centrality_project.py --input path/to/edges.tsv --colA "ProteinA" --colB "ProteinB" --sep "\t"

# 2) If you don't have a file, create a synthetic PPI-like graph:
python ppi_centrality_project.py --synthetic

# 3) Make report (DOCX + PPTX):
python ppi_centrality_project.py --synthetic --make-report

Outputs (default):
- ./output/centralities.csv
- ./output/degree_distribution.png
- ./output/betweenness_distribution.png
- ./output/degree_vs_betweenness.png
- ./output/network_visualization.png
- optional ./output/PPI_Project_Report.docx
- optional ./output/PPI_Project_Presentation.pptx
"""

import argparse
import os
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# Optional imports for report generation
try:
    from docx import Document
    from docx.shared import Inches
    from pptx import Presentation
    from pptx.util import Inches as PPTInches
    REPORT_LIBS_AVAILABLE = True
except Exception:
    REPORT_LIBS_AVAILABLE = False


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def load_edge_list(path: str, sep: str = "\t", colA: str = None, colB: str = None) -> pd.DataFrame:
    """
    Load an edge list file and return a DataFrame with two columns ['A','B'].
    If colA/colB provided, use them; else try to infer common column names.
    """
    df_raw = pd.read_csv(path, sep=sep, header=0, dtype=str, low_memory=False)
    # normalize column names
    cols = {c.strip(): c for c in df_raw.columns}
    # if user provided names
    if colA and colB:
        if colA not in cols or colB not in cols:
            raise ValueError(f"Provided column names not found in file. Columns found: {list(df_raw.columns)}")
        df = df_raw[[cols[colA], cols[colB]]].copy()
    else:
        # Try common column names
        candidates = [
            ("protein1", "protein2"),
            ("Protein1", "Protein2"),
            ("interactor_a", "interactor_b"),
            ("Official Symbol Interactor A", "Official Symbol Interactor B"),
            ("A", "B"),
        ]
        found = False
        for a, b in candidates:
            if a in cols and b in cols:
                df = df_raw[[cols[a], cols[b]]].copy()
                found = True
                break
        if not found:
            # fall back: take the first two columns
            df = df_raw.iloc[:, :2].copy()
    df.columns = ["A", "B"]
    # drop NA and self-loops
    df = df.dropna(subset=["A", "B"])
    df = df[df["A"].astype(str) != df["B"].astype(str)]
    df = df.drop_duplicates()
    return df


def build_graph_from_edges(df_edges: pd.DataFrame) -> nx.Graph:
    G = nx.from_pandas_edgelist(df_edges, "A", "B")
    # remove isolated nodes if any (none usually)
    G.remove_nodes_from(list(nx.isolates(G)))
    return G


def generate_synthetic_graph(n_nodes: int = 200, m: int = 3) -> nx.Graph:
    """
    Generate a synthetic scale-free graph using Barabasi-Albert model.
    This resembles many PPI network degree distributions.
    """
    G = nx.barabasi_albert_graph(n_nodes, m, seed=42)
    # relabel nodes as "P0", "P1", ...
    mapping = {node: f"P{node}" for node in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    return G


def compute_centralities(G: nx.Graph) -> pd.DataFrame:
    print("Computing centralities ...")
    # degree centrality
    deg = nx.degree_centrality(G)
    # betweenness (could be slow for very large graphs)
    bet = nx.betweenness_centrality(G, normalized=True)
    # closeness
    cls = nx.closeness_centrality(G)
    # eigenvector (may fail to converge on some graphs; handle)
    try:
        eig = nx.eigenvector_centrality(G, max_iter=1000)
    except Exception as e:
        print("eigenvector_centrality failed to converge; falling back to numpy.linalg eig on adjacency.")
        eig = nx.eigenvector_centrality_numpy(G)
    # pagerank
    pr = nx.pagerank(G)

    df = pd.DataFrame({
        "degree": pd.Series(deg),
        "betweenness": pd.Series(bet),
        "closeness": pd.Series(cls),
        "eigenvector": pd.Series(eig),
        "pagerank": pd.Series(pr)
    })
    # sort descending by degree by default (helpful)
    df = df.sort_values(by="degree", ascending=False)
    return df


def save_plots(df: pd.DataFrame, G: nx.Graph, out_dir: str) -> dict:
    plots = {}
    # Degree distribution
    plt.figure(figsize=(6, 4))
    plt.hist(df["degree"], bins=30)
    plt.title("Degree Centrality Distribution")
    plt.xlabel("Degree centrality")
    plt.ylabel("Frequency")
    p1 = os.path.join(out_dir, "degree_distribution.png")
    plt.tight_layout()
    plt.savefig(p1)
    plt.close()
    plots["degree_distribution"] = p1

    # Betweenness distribution
    plt.figure(figsize=(6, 4))
    plt.hist(df["betweenness"], bins=30)
    plt.title("Betweenness Centrality Distribution")
    plt.xlabel("Betweenness centrality")
    plt.ylabel("Frequency")
    p2 = os.path.join(out_dir, "betweenness_distribution.png")
    plt.tight_layout()
    plt.savefig(p2)
    plt.close()
    plots["betweenness_distribution"] = p2

    # Scatter: degree vs betweenness
    plt.figure(figsize=(6, 4))
    plt.scatter(df["degree"], df["betweenness"], s=12)
    plt.title("Degree vs Betweenness")
    plt.xlabel("Degree centrality")
    plt.ylabel("Betweenness centrality")
    p3 = os.path.join(out_dir, "degree_vs_betweenness.png")
    plt.tight_layout()
    plt.savefig(p3)
    plt.close()
    plots["degree_vs_betweenness"] = p3

    # Network visualization (may be messy for large graphs)
    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, node_size=20, edge_color="grey", linewidths=0.2)
    plt.title("Network visualization (spring layout)")
    p4 = os.path.join(out_dir, "network_visualization.png")
    plt.tight_layout()
    plt.savefig(p4)
    plt.close()
    plots["network_visualization"] = p4

    return plots


def save_csv(df: pd.DataFrame, out_path: str):
    df.to_csv(out_path, index_label="protein")
    print(f"Wrote centralities CSV to: {out_path}")


# m

def topk_report(df: pd.DataFrame, k: int = 10) -> str:
    lines = []
    lines.append("Top proteins by centrality measures (top {}):\n".format(k))
    for col in df.columns:
        topk = df[col].nlargest(k).index.tolist()
        lines.append(f"{col} top-{k}: {', '.join(map(str, topk))}")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="PPI centrality analysis script")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Path to edge list file (tsv/csv). If omitted, synthetic graph is used.")
    parser.add_argument("--sep", type=str, default="\t", help="Separator for input file (default: tab). Use '\\t' or ','")
    parser.add_argument("--colA", type=str, default=None, help="Column name for protein A (if header present)")
    parser.add_argument("--colB", type=str, default=None, help="Column name for protein B (if header present)")
    parser.add_argument("--out", "-o", type=str, default="output", help="Output directory")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic graph (ignore --input)")
    parser.add_argument("--nodes", type=int, default=200, help="Nodes for synthetic graph (default 200)")
    parser.add_argument("--m", type=int, default=3, help="Edges per new node for synthetic BA graph (default 3)")
    parser.add_argument("--make-report", action="store_true", help="Generate DOCX and PPTX report (requires python-docx and python-pptx)")
    parser.add_argument("--topk", type=int, default=10, help="Top-k to print in terminal")
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = args.out
    ensure_dir(outdir)

    # Load or generate graph
    if args.input and not args.synthetic:
        print(f"Loading edge list from {args.input} (sep='{args.sep}')")
        try:
            df_edges = load_edge_list(args.input, sep=args.sep, colA=args.colA, colB=args.colB)
            G = build_graph_from_edges(df_edges)
            print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        except Exception as e:
            print(f"Failed to load input file: {e}")
            print("Falling back to synthetic graph.")
            G = generate_synthetic_graph(n_nodes=args.nodes, m=args.m)
            print(f"Synthetic graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    else:
        print("Generating synthetic PPI-like graph.")
        G = generate_synthetic_graph(n_nodes=args.nodes, m=args.m)
        print(f"Synthetic graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Compute centralities
    df_c = compute_centralities(G)

    # Save CSV
    csv_path = os.path.join(outdir, "centralities.csv")
    save_csv(df_c, csv_path)

    # Save plots
    plots = save_plots(df_c, G, outdir)
    print("Saved plots:", plots)

    # Print top-k to terminal
    print("\n" + topk_report(df_c, k=args.topk))

    # Optional report and slides
    if args.make_report:
        docx_path = os.path.join(outdir, "PPI_Project_Report.docx")
        pptx_path = os.path.join(outdir, "PPI_Project_Presentation.pptx")
        if not REPORT_LIBS_AVAILABLE:
            print("\nNote: docx/pptx libraries are not available in this environment.")
            print("Install python-docx and python-pptx to enable report generation:")
            print("pip install python-docx python-pptx")
        else:
            make_docx_report(plots, docx_path)
            make_pptx(plots, pptx_path)

    print("\nAll done. Outputs saved to:", os.path.abspath(outdir))


if __name__ == "__main__":
    main()
