import argparse
import os
import random

import anndata
import numpy as np
import pandas
import scanpy as sc
import torch
from sklearn.metrics import adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, homogeneity_score, v_measure_score

from SpatialGlue.utils import clustering
from matplotlib import pyplot as plt
from numpy.linalg import norm

import wandb
from SpaMV.metrics import moranI_score, compute_supervised_scores, calculate_jaccard
from SpaMV.spamv import SpaMV
from SpaMV.utils import ST_preprocess, clr_normalize_each_cell, plot_results

sc._settings.ScanpyConfig.figdir = '.'
# Argument to parse
parser = argparse.ArgumentParser(description='SpaMV Experiment')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--zp_dim_omics1', type=int, default=10, help='latent modality 1-specific dimensionality')
parser.add_argument('--zp_dim_omics2', type=int, default=10, help='latent modality 2-specific dimensionality')
parser.add_argument('--zs_dim', type=int, default=10, help='latent shared dimensionality')
parser.add_argument('--heads', type=int, default=1, help='number of heads in GAT')
parser.add_argument('--hidden_size', type=int, default=128, help='hidden layer size')
parser.add_argument('--n_neighbors', type=int, default=20, help='number of neighbors in GNN')
parser.add_argument('--seed', type=int, default=1214, help='random seed')
parser.add_argument('--beta', type=float, default=1, help='beta hyperparameter in VAE objective')
parser.add_argument('--learning_rate', type=float, default=1e-2, help='learning rate')
parser.add_argument('--reweight', type=bool, default=False, help='reweight the loss of different modalities')

# Args
args = parser.parse_args()
wandb.login()
root_path = "Data/"
result = pandas.DataFrame(columns=['method', 'dataset', 'metrics', 'score'])

for _, dirs, _ in os.walk(root_path):
    for data in dirs:
        print(data)
        folder_path = 'results/' + data + '/'
        adata_combined = []
        adata_raw = []
        omics_names = []
        recon_types = []
        for filename in os.listdir(root_path + data + "/"):
            adata = sc.read_h5ad(root_path + data + "/" + filename)
            adata.var_names_make_unique()
            sc.pp.filter_cells(adata, min_genes=50 if adata.n_vars > 100 else 5)
            adata_raw.append(adata.copy())
            if "RNA" in filename or "peak" in filename:
                adata = ST_preprocess(adata, prune=True)
                adata = adata[:, adata.var.highly_variable]
                adata_raw[-1] = adata_raw[-1][:, adata.var_names]
                recon_types.append('nb')
                omics_names.append("RNA" if "RNA" in filename else "ATAC")
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

        run = wandb.init(project=data, config=args,
                         name=str(args.zp_dim_omics1) + '_' + str(args.zs_dim) + '_' + str(
                             args.zp_dim_omics2) + '_' + str(
                             args.beta) + '_' + str(args.learning_rate) + '_' + str(args.reweight) + '_' + str(
                             args.seed) + '_' + str(
                             args.heads) + '_' + str(args.n_neighbors) + '_' + str(True))
        model = SpaMV([adata_omics1, adata_omics2], zs_dim=args.zs_dim,
                      zp_dims=[args.zp_dim_omics1, args.zp_dim_omics2], weights=[weight_omics1, weight_omics2],
                      interpretable=True, hidden_size=args.hidden_size, heads=args.heads,
                      n_neighbors=args.n_neighbors, random_seed=args.seed,
                      recon_types=[recon_type_omics1, recon_type_omics2], omics_names=omics_names)
        model.train(min_epochs=100, max_epochs=args.epochs, min_kl=args.beta, max_kl=args.beta,
                    learning_rate=args.learning_rate, folder_path=folder_path)
        run.finish()
        z, w = model.get_embedding_and_feature_by_topic(map=True)
        plot_results([adata_omics1, adata_omics2], omics_names, z, w, save=True, show=False,
                     corresponding_features=False, folder_path=folder_path,
                     file_name='spamv_interpretable_without_features.pdf')
        plot_results([adata_omics1, adata_omics2], omics_names, z, w, save=True, show=False,
                     corresponding_features=True, folder_path=folder_path,
                     file_name='spamv_interpretable_with_features.pdf')
        adata_omics1.obs['spamv'] = z.idxmax(1)
        if 'cluster' in adata_omics1.obs:
            scores = compute_supervised_scores(adata_omics1, z)
            for metrics in scores.keys():
                result.loc[len(result.index)] = ['spamv', data, metrics, scores[metrics]]
        result.loc[len(result.index)] = ['spamv', data, 'moranI', moranI_score(adata_omics1, 'spamv')]
        adata_omics1.obsm['spamv_ps'] = z.loc[:, w[0].columns]
        result.loc[len(result.index)] = ['spamv', data, 'jaccard_' + omics_names[0],
                                         calculate_jaccard(adata_omics1, 'spamv_ps')]
        adata_omics2.obsm['spamv_ps'] = z.loc[:, w[1].columns]
        result.loc[len(result.index)] = ['spamv', data, 'jaccard_' + omics_names[1],
                                         calculate_jaccard(adata_omics2, 'spamv_ps')]
