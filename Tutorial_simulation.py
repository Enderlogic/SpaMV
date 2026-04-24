import argparse
import os
import sys
import anndata
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

from SpaMV.spamv import SpaMV
from SpaMV.utils import (
    clr_normalize_each_cell,
    clustering,
    plot_embedding_results,
)

# -------------------------
# Configuration
# -------------------------

DATA_DIR = "Data"
OUTPUT_DIR = "Outputs"
DEFAULT_CLUSTERS = 10

DATASET_CLUSTER_MAP = {
    "Simulation_1": 8,
    "Simulation_2": 6,
    "Simulation_3": 4,
}


# -------------------------
# Utilities
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SpaMV pipeline on a specified dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='Simulation_1',
        help="Dataset name (i.e., Simulation_1, Simulation_2, Simulation_3)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=OUTPUT_DIR,
        help="Directory to save Outputs (default: Outputs/)",
    )
    return parser.parse_args()


def validate_dataset(dataset: str):
    dataset_path = os.path.join(DATA_DIR, dataset)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")


def get_output_path(outdir: str, filename: str):
    os.makedirs(outdir, exist_ok=True)
    return os.path.join(outdir, filename)


def load_data(dataset: str):
    print(f"[INFO] Loading dataset: {dataset}")

    rna_path = os.path.join(DATA_DIR, dataset, "adata_RNA.h5ad")
    pro_path = os.path.join(DATA_DIR, dataset, "adata_ADT.h5ad")

    data_rna = sc.read_h5ad(rna_path)
    data_pro = sc.read_h5ad(pro_path)

    return data_rna, data_pro


def preprocess_data(data_rna, data_pro):
    print("[INFO] Preprocessing data...")

    # RNA preprocessing
    sc.pp.normalize_total(data_rna)
    sc.pp.log1p(data_rna)
    sc.pp.pca(data_rna)
    data_rna = anndata.AnnData(
        data_rna.obsm["X_pca"], obs=data_rna.obs, obsm=data_rna.obsm
    )

    # Protein preprocessing
    data_pro = clr_normalize_each_cell(data_pro)
    sc.pp.pca(data_pro)
    data_pro = anndata.AnnData(
        data_pro.obsm["X_pca"], obs=data_pro.obs, obsm=data_pro.obsm
    )

    return data_rna, data_pro


def get_num_clusters(dataset: str):
    return DATASET_CLUSTER_MAP.get(dataset, DEFAULT_CLUSTERS)


# -------------------------
# Core Pipeline
# -------------------------

def run_clustering_pipeline(data_rna, data_pro, dataset):
    print("[INFO] Training SpaMV (clustering mode)...")

    model = SpaMV(
        [data_rna, data_pro],
        interpretable=False,
        max_epochs_stage1=100,
    )
    model.train()
    # Full embedding
    data_rna.obsm["SpaMV"] = model.get_embedding()
    print('test point 0')
    clustering(data_rna, n_clusters=10, key="SpaMV")
    print('test point 1')
    # Shared embedding
    data_rna.obsm["SpaMV (Shared)"] = data_rna.obsm["SpaMV"][:, :model.zs_dim]
    n_clusters = get_num_clusters(dataset)
    clustering(
        data_rna,
        n_clusters=n_clusters,
        key="SpaMV (Shared)",
        add_key="SpaMV (Shared)",
    )
    return data_rna, model


def plot_results(data_rna, dataset, outdir):
    print("[INFO] Generating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    # --- Full embedding ---
    sc.pp.neighbors(data_rna, use_rep="SpaMV")
    sc.tl.umap(data_rna)

    sc.pl.umap(
        data_rna,
        color="SpaMV",
        ax=axes[0][0],
        show=False,
        legend_loc="none",
        s=20,
        title="UMAP",
    )

    ari_full = adjusted_rand_score(
        data_rna.obs["cluster"], data_rna.obs["SpaMV"]
    )

    sc.pl.embedding(
        data_rna,
        color="SpaMV",
        basis="spatial",
        s=200,
        show=False,
        ax=axes[0][1],
        title=f"SpaMV on {dataset}\nARI: {ari_full:.3f}",
    )

    # --- Shared embedding ---
    sc.pp.neighbors(data_rna, use_rep="SpaMV (Shared)")
    sc.tl.umap(data_rna)

    sc.pl.umap(
        data_rna,
        color="SpaMV (Shared)",
        ax=axes[1][0],
        show=False,
        legend_loc="none",
        s=20,
        title="UMAP",
    )

    ari_shared = adjusted_rand_score(
        data_rna.obs["cluster"], data_rna.obs["SpaMV (Shared)"]
    )

    sc.pl.embedding(
        data_rna,
        color="SpaMV (Shared)",
        basis="spatial",
        s=200,
        show=False,
        ax=axes[1][1],
        title=f"SpaMV (Shared) on {dataset}\nARI: {ari_shared:.3f}",
    )

    plt.tight_layout()

    filename = f"clustering_{dataset}.pdf"
    output_file = get_output_path(outdir, filename)

    plt.savefig(output_file)
    print(f"[INFO] Saved {output_file}")


def run_topic_modeling(dataset, outdir):
    print("[INFO] Training SpaMV (interpretable mode)...")

    data_rna = sc.read_h5ad(f"{DATA_DIR}/{dataset}/adata_RNA.h5ad")
    data_pro = sc.read_h5ad(f"{DATA_DIR}/{dataset}/adata_ADT.h5ad")

    sc.pp.normalize_total(data_rna)
    sc.pp.log1p(data_rna)
    data_pro = clr_normalize_each_cell(data_pro)

    model = SpaMV(
        [data_rna, data_pro],
        alphas=[3, 3],
        interpretable=True,
        neighborhood_embedding=5,
        threshold_background=5,
        max_epochs_stage1=400,
        max_epochs_stage2=200,
    )
    model.train()

    z, w = model.get_embedding_and_feature_by_topic(threshold=0.1)

    filename = f"topic_modeling_{dataset}.pdf"

    plot_embedding_results(
        [data_rna, data_pro],
        model.omics_names,
        z,
        w,
        save=True,
        show=False,
        size=350,
        folder_path=outdir + '/',
        file_name=filename,
    )


# -------------------------
# Main
# -------------------------

def main():
    args = parse_args()
    dataset = args.dataset
    outdir = args.outdir

    try:
        validate_dataset(dataset)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    data_rna, data_pro = load_data(dataset)
    data_rna, data_pro = preprocess_data(data_rna, data_pro)

    data_rna, model = run_clustering_pipeline(data_rna, data_pro, dataset)
    plot_results(data_rna, dataset, outdir)

    run_topic_modeling(dataset, outdir)

    print("[INFO] Pipeline completed successfully.")


if __name__ == "__main__":
    main()