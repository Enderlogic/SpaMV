import argparse
import os
import sys
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

from SpaMV.spamv import SpaMV
from SpaMV.utils import (
    preprocess_dc,
    preprocess_idr,
    clustering,
    plot_embedding_results,
)

# -------------------------
# Configuration
# -------------------------

DATA_DIR = "Data"
OUTPUT_DIR = "Outputs"

DATASET_CONFIG = {
    "Mouse_Embryo": {
        "files": ["adata_RNA.h5ad", "adata_peaks.h5ad", "adata_ATAC.h5ad"],
        "omics": ["Transcriptome", "Epigenome"],
        "clusters": 14,
        "scale": False,
    },
    "ME13_1": {
        "files": ["adata_H3K27ac_ATAC.h5ad", "adata_H3K27me3_ATAC.h5ad"],
        "omics": ["H3K27ac", "H3K27me3"],
        "clusters": 21,
        "scale": False,
    },
    "Mouse_Thymus": {
        "files": ["adata_RNA.h5ad", "adata_ADT.h5ad"],
        "omics": ["Transcriptome", "Proteome"],
        "clusters": 7,
        "scale": True,
    },
    "ccRCC_Y7_T": {
        "files": ["adata_RNA.h5ad", "adata_MET.h5ad"],
        "omics": ["Transcriptome", "Metabolome"],
        "clusters": 16,
        "scale": False,
    },
}


# -------------------------
# Utilities
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SpaMV on real-world datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Mouse_Embryo",
        help="Dataset name",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=OUTPUT_DIR,
        help="Output directory",
    )
    return parser.parse_args()


def validate_dataset(dataset: str):
    if dataset not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset: {dataset}")
    dataset_path = os.path.join(DATA_DIR, dataset)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")


def get_output_path(outdir: str, filename: str):
    os.makedirs(outdir, exist_ok=True)
    return os.path.join(outdir, filename)


def load_data(dataset: str):
    print(f"[INFO] Loading dataset: {dataset}")
    cfg = DATASET_CONFIG[dataset]

    data_paths = [
        os.path.join(DATA_DIR, dataset, f) for f in cfg["files"]
    ]
    datasets = [sc.read_h5ad(p) for p in data_paths]

    return datasets, cfg


def preprocess_data(datasets, cfg, task):
    print("[INFO] Preprocessing data...")
    if task == 'clustering':
        if len(datasets) == 3:
            datasets = datasets[:2]
        datasets = preprocess_dc(datasets, cfg["omics"], scale=cfg["scale"])
    elif task == 'dimension_reduction':
        if len(datasets) == 3:
            datasets = [datasets[0], datasets[2]]
        datasets = preprocess_idr(datasets, cfg['omics'])
    else:
        raise ValueError(f"Unsupported task: {task}")
    return datasets


# -------------------------
# Clustering Pipeline
# -------------------------

def run_clustering_pipeline(datasets, cfg):
    print("[INFO] Training SpaMV (clustering mode)...")

    model = SpaMV(
        datasets,
        interpretable=False,
    )
    model.train()

    datasets[0].obsm["SpaMV"] = model.get_embedding()

    clustering(
        datasets[0],
        n_clusters=cfg["clusters"],
        key="SpaMV",
    )

    return datasets, model


def plot_results(datasets, cfg, dataset, outdir):
    print("[INFO] Generating plots...")

    data = datasets[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sc.pp.neighbors(data, use_rep="SpaMV")
    sc.tl.umap(data)

    sc.pl.umap(
        data,
        color="SpaMV",
        ax=axes[0],
        show=False,
        s=20,
        title="UMAP",
    )

    if "spatial" in data.uns:
        title = (
            f"SpaMV\nARI: {adjusted_rand_score(data.obs['cluster'], data.obs['SpaMV']):.3f}"
            if "cluster" in data.obs else "SpaMV"
        )
        sc.pl.spatial(
            data,
            color="SpaMV",
            title=title,
            show=False,
            ax=axes[1],
        )
    else:
        sc.pl.embedding(
            data,
            basis="spatial",
            color="SpaMV",
            show=False,
            ax=axes[1],
            size=50,
        )

    plt.tight_layout()

    filename = f"clustering_{dataset}.pdf"
    output_file = get_output_path(outdir, filename)

    plt.savefig(output_file)
    print(f"[INFO] Saved {output_file}")


# -------------------------
# Interpretable Pipeline
# -------------------------

def run_interpretable_pipeline(dataset, outdir):
    print("[INFO] Training SpaMV (interpretable mode)...")

    cfg = DATASET_CONFIG[dataset]

    data_paths = [
        os.path.join(DATA_DIR, dataset, f) for f in cfg["files"]
    ]
    datasets = [sc.read_h5ad(p) for p in data_paths]
    data = preprocess_data(datasets, cfg, 'dimension_reduction')

    zs_dim = 10
    weights = [1, 1]
    alphas = [1, 1]
    threshold_background = 1

    if dataset == "Mouse_Thymus":
        zp_dims = [10, 10]
        weights = [1, 10]
        alphas = [5, 5]
    elif dataset == "Mouse_Embryo":
        zp_dims = [20, 5]
    elif dataset == "ME13_1":
        zs_dim = 20
        zp_dims = [5, 5]
        alphas = [3, 3]
        threshold_background = 5
    elif dataset == "ccRCC_Y7_T":
        zp_dims = [5, 5]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    model = SpaMV(
        data,
        zs_dim=zs_dim,
        zp_dims=zp_dims,
        interpretable=True,
        weights=weights,
        alphas=alphas,
        omics_names=cfg["omics"],
        threshold_background=threshold_background,
    )

    model.train()

    z, w = model.get_embedding_and_feature_by_topic()

    if dataset == "ccRCC_Y7_T":
        spatial_range = (
            data[0].obsm["spatial"].min(0).tolist() +
            data[0].obsm["spatial"].max(0).tolist()
        )
        width = spatial_range[2] - spatial_range[0]
        height = spatial_range[3] - spatial_range[1]

        crop_coord = [(
            spatial_range[0] - width / 20,
            spatial_range[1] - height / 20,
            spatial_range[2] + width / 20,
            spatial_range[3] + height / 20,
        )]

        plot_embedding_results(
            data,
            cfg["omics"],
            z,
            w,
            show=False,
            save=True,
            crop_coord=crop_coord,
            folder_path=outdir + "/",
            file_name=f"interpretable_{dataset}.pdf",
        )
    else:
        plot_embedding_results(
            data,
            cfg["omics"],
            z,
            w,
            show=False,
            save=True,
            size=100,
            folder_path=outdir + "/",
            file_name=f"interpretable_{dataset}.pdf",
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

    run_interpretable_pipeline(dataset, outdir)

    datasets, cfg = load_data(dataset)
    datasets = preprocess_data(datasets, cfg, 'clustering')

    datasets, model = run_clustering_pipeline(datasets, cfg)
    plot_results(datasets, cfg, dataset, outdir)

    print("[INFO] Pipeline completed successfully.")


if __name__ == "__main__":
    main()