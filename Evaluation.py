import argparse
import os
import random

import numpy
import numpy as np
import scanpy as sc
from numpy.linalg import norm
from pandas import DataFrame
from sklearn.metrics import adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, homogeneity_score, v_measure_score

import wandb
from SpaMV.metrics import compute_moranI, compute_jaccard
from SpaMV.spamv import SpaMV
from SpaMV.utils import ST_preprocess, clr_normalize_each_cell, clustering


def cosine_similarity(A, B):
    return np.dot(A, B) / (norm(A) * norm(B))


sc._settings.ScanpyConfig.figdir = '.'
# Argument to parse
parser = argparse.ArgumentParser(description='SpaMV Experiment')
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--zp_dim_omics1', type=int, default=32, help='modality 1-specific latent dimensionality')
parser.add_argument('--zp_dim_omics2', type=int, default=32, help='modality 2-specific latent dimensionality')
parser.add_argument('--zs_dim', type=int, default=32, help='shared latent dimensionality')
parser.add_argument('--heads', type=int, default=1, help='number of heads in GAT')
parser.add_argument('--hidden_size', type=int, default=256, help='hidden layer size')
parser.add_argument('--n_neighbors', type=int, default=20, help='number of neighbors in GNN')
parser.add_argument('--beta', type=float, default=1, help='beta hyperparameter in VAE objective')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--interpretable', type=bool, default=False, help='whether to use interpretable mode')
parser.add_argument('--reweight', type=bool, default=False, help='reweight the loss of different modalities')

# Args
args = parser.parse_args()
wandb.login()
root_path = "Data/"
result = DataFrame(
    columns=["data", "ari", "mi", "nmi", "ami", "hom", "vme", "average", "moran I", "jaccard 1", "jaccard 2"])
for _, dirs, _ in os.walk(root_path):
    for data in dirs:
        print(data)
        folder_path = 'results/' + data + '/'
        adata_combined = []
        adata_raw = []
        omics_names = []
        recon_types = []
        if 'Spleen' in data:
            n_cluster = 5
        elif 'Lymph_Node' in data:
            n_cluster = 10
        elif 'Thymus' in data:
            n_cluster = 8
        elif 'Brain' in data:
            n_cluster = 18
        else:
            raise ValueError
        for filename in os.listdir(root_path + data + "/"):
            adata = sc.read_h5ad(root_path + data + "/" + filename)
            adata.var_names_make_unique()
            sc.pp.filter_cells(adata, min_genes=50 if adata.n_vars > 100 else 5)
            adata_raw.append(adata.copy())
            if "RNA" in filename or "peak" in filename or 'SM' in filename:
                # sc.pp.filter_genes(adata, min_cells=round(adata.n_obs * .05))
                adata = ST_preprocess(adata)
                adata = adata[:, adata.var.highly_variable]
                adata_raw[-1] = adata_raw[-1][:, adata.var_names]
                recon_types.append('zinb')
                if 'RNA' in filename:
                    omics_names.append('RNA')
                elif 'peak' in filename:
                    omics_names.append('ATAC')
                else:
                    omics_names.append('SM')
            elif "ADT" in filename:
                adata = clr_normalize_each_cell(adata)
                sc.pp.pca(adata)
                omics_names.append("Protein")
                recon_types.append('nb')
            adata_combined.append(adata)
        adata_omics1 = adata_combined[0]
        adata_omics2 = adata_combined[1]
        recon_type_omics1 = recon_types[0]
        recon_type_omics2 = recon_types[1]
        for v in adata_omics1.var_names:
            if v in adata_omics2.var_names:
                adata_omics1.var.rename(index={v: v + '_' + omics_names[0]}, inplace=True)
                adata_raw[0].var.rename(index={v: v + '_' + omics_names[0]}, inplace=True)
        adata_omics1 = adata_omics1[adata_omics1.obs_names.intersection(adata_omics2.obs_names), :]
        adata_omics2 = adata_omics2[adata_omics2.obs_names.intersection(adata_omics1.obs_names), :]
        adata_raw[0] = adata_raw[0][adata_omics1.obs_names.intersection(adata_omics2.obs_names), :]
        adata_raw[1] = adata_raw[1][adata_omics1.obs_names.intersection(adata_omics2.obs_names), :]
        weight_omics1 = 1
        weight_omics2 = 1
        if args.reweight:
            if adata_omics1.n_vars > adata_omics2.n_vars:
                weight_omics2 = adata_omics1.n_vars / adata_omics2.n_vars
            else:
                weight_omics1 = adata_omics2.n_vars / adata_omics1.n_vars

        for _ in range(10):
            seed = random.randint(1, 10000)
            run = wandb.init(project=data, config=args,
                             name=str(args.zp_dim_omics1) + '_' + str(args.zs_dim) + '_' + str(
                                 args.zp_dim_omics2) + '_' + str(args.beta) + '_' + str(args.learning_rate) + '_' + str(
                                 args.reweight) + '_' + str(seed) + '_' + str(args.heads) + '_' + str(
                                 args.n_neighbors) + '_' + str(args.interpretable))
            model = SpaMV([adata_omics1, adata_omics2], zs_dim=args.zs_dim,
                          zp_dims=[args.zp_dim_omics1, args.zp_dim_omics2], weights=[weight_omics1, weight_omics2],
                          interpretable=args.interpretable, hidden_size=args.hidden_size, heads=args.heads,
                          n_neighbors=args.n_neighbors, random_seed=seed,
                          recon_types=[recon_type_omics1, recon_type_omics2], omics_names=omics_names, min_epochs=100,
                          max_epochs=args.epochs, min_kl=args.beta, max_kl=args.beta, learning_rate=args.learning_rate,
                          folder_path=folder_path, n_cluster=n_cluster)
            model.train()
            run.finish()
            z = model.get_embedding()
            adata_omics1.obsm['spamv'] = z
            adata_omics1.obsm['zs+zp1'] = z[:, :args.zs_dim + args.zp_dim_omics1]
            adata_omics2.obsm['zs+zp2'] = numpy.concatenate((z[:, :args.zs_dim], z[:, -args.zp_dim_omics2:]), axis=1)
            jaccard1 = compute_jaccard(adata_omics1, adata_omics1, 'zs+zp1', 'X_pca')
            jaccard2 = compute_jaccard(adata_omics2, adata_omics2, 'zs+zp2', 'X_pca')
            clustering(adata_omics1, key='spamv', add_key='spamv', n_clusters=n_cluster,
                       method='mclust', use_pca=True)
            moranI = compute_moranI(adata_omics1, 'spamv')
            if 'cluster' in adata_omics1.obs:
                cluster = adata_omics1.obs['cluster']
                cluster_learned = adata_omics1.obs['spamv']
                ari = adjusted_rand_score(cluster, cluster_learned)
                mi = mutual_info_score(cluster, cluster_learned)
                nmi = normalized_mutual_info_score(cluster, cluster_learned)
                ami = adjusted_mutual_info_score(cluster, cluster_learned)
                hom = homogeneity_score(cluster, cluster_learned)
                vme = v_measure_score(cluster, cluster_learned)
                ave = (ari + mi + nmi + ami + hom + vme) / 6
            else:
                ari = mi = nmi = hom = vme = ami = ave = 0
            result.loc[len(result)] = [data, ari, mi, nmi, ami, hom, vme, ave, moranI, jaccard1, jaccard2]
            print(result.tail(1).to_string())
            result.to_csv("results/Evaluation.csv", index=False)
