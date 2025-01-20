import argparse
import os

import pandas
import scanpy as sc
from sklearn.preprocessing import MinMaxScaler

import wandb
from SpaMV.metrics import compute_moranI, compute_supervised_scores, compute_jaccard
from SpaMV.spamv import SpaMV
from SpaMV.utils import ST_preprocess, clr_normalize_each_cell, plot_embedding_results, plot_clustering_results

sc._settings.ScanpyConfig.figdir = '.'
# Argument to parse
parser = argparse.ArgumentParser(description='SpaMV Experiment')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--zp_dim_omics1', type=int, default=10, help='latent modality 1-specific dimensionality')
parser.add_argument('--zp_dim_omics2', type=int, default=10, help='latent modality 2-specific dimensionality')
parser.add_argument('--zs_dim', type=int, default=10, help='latent shared dimensionality')
parser.add_argument('--heads', type=int, default=1, help='number of heads in GAT')
parser.add_argument('--hidden_size', type=int, default=128, help='hidden layer size')
parser.add_argument('--n_neighbors', type=int, default=20, help='number of neighbors in GNN')
parser.add_argument('--seed', type=int, default=941214, help='random seed')
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
            if "RNA" in filename or "peak" in filename or "SM" in filename:
                adata = ST_preprocess(adata, prune=True)
                adata = adata[:, adata.var.highly_variable]
                adata_raw[-1] = adata_raw[-1][:, adata.var_names]
                recon_types.append('nb')
                if "RNA" in filename:
                    omics_names.append('Transcriptomics')
                elif "peak" in filename:
                    omics_names.append('Epigenomics')
                elif "SM" in filename:
                    omics_names.append('Metabonomics')
            elif "ADT" in filename:
                adata = clr_normalize_each_cell(adata)
                sc.pp.pca(adata)
                omics_names.append("Proteomics")
                recon_types.append('nb')
            adata_combined.append(adata)
        adata_omics1 = adata_combined[0]
        adata_omics2 = adata_combined[1]
        recon_type_omics1 = recon_types[0]
        recon_type_omics2 = recon_types[1]
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

        run = wandb.init(project=data, config=args, name=str(args.zp_dim_omics1) + '_' + str(args.zs_dim) + '_' + str(
            args.zp_dim_omics2) + '_' + str(args.beta) + '_' + str(args.learning_rate) + '_' + str(
            args.reweight) + '_' + str(args.seed) + '_' + str(args.heads) + '_' + str(args.n_neighbors) + '_' + str(
            True))
        model = SpaMV([adata_omics1, adata_omics2], zs_dim=args.zs_dim,
                      zp_dims=[args.zp_dim_omics1, args.zp_dim_omics2], weights=[weight_omics1, weight_omics2],
                      interpretable=True, hidden_size=args.hidden_size, heads=args.heads, n_neighbors=args.n_neighbors,
                      random_seed=args.seed, recon_types=[recon_type_omics1, recon_type_omics2],
                      omics_names=omics_names, min_epochs=100, max_epochs=args.epochs, min_kl=args.beta,
                      max_kl=args.beta, learning_rate=args.learning_rate, folder_path=folder_path)
        model.train()
        run.finish()
        z, w = model.get_embedding_and_feature_by_topic()
        plot_embedding_results(adata_raw, omics_names, z, w, save=True, show=False, corresponding_features=False,
                               folder_path=folder_path,
                               file_name='spamv_interpretable_without_features_without_map.pdf')
        plot_embedding_results(adata_raw, omics_names, z, w, save=True, show=False, corresponding_features=True,
                               folder_path=folder_path, file_name='spamv_interpretable_with_features_without_map.pdf')
        z, w = model.get_embedding_and_feature_by_topic(map=True)
        z.to_csv(folder_path + 'embedding.csv')
        w[0].to_csv(folder_path + 'weight_' + omics_names[0] + '.csv')
        w[1].to_csv(folder_path + 'weight_' + omics_names[1] + '.csv')
        plot_embedding_results(adata_raw, omics_names, z, w, save=True, show=False, corresponding_features=False,
                               folder_path=folder_path, file_name='spamv_interpretable_without_features_with_map.pdf')
        plot_embedding_results(adata_raw, omics_names, z, w, save=True, show=False, corresponding_features=True,
                               folder_path=folder_path, file_name='spamv_interpretable_with_features_with_map.pdf')

        for emb_type in ['spamv', 'spamv_scaled']:
            adata_omics1.obsm[emb_type] = z if emb_type == 'spamv' else pandas.DataFrame(
                MinMaxScaler().fit_transform(z), columns=z.columns, index=z.index)
            adata_omics1.obs[emb_type] = z.idxmax(1) if emb_type == 'spamv' else pandas.DataFrame(
                MinMaxScaler().fit_transform(z), columns=z.columns, index=z.index).idxmax(1)
            plot_clustering_results(adata_omics1, emb_type, omics_names, folder_path,
                                    suffix=None if emb_type == 'spamv' else 'scaled')
            if 'cluster' in adata_omics1.obs:
                scores = compute_supervised_scores(adata_omics1, emb_type)
                for metrics in scores.keys():
                    result.loc[len(result.index)] = [emb_type, data, metrics, scores[metrics]]
            result.loc[len(result.index)] = [emb_type, data, 'moranI', compute_moranI(adata_omics1, emb_type)]
            adata_omics1.obsm['spamv_ps'] = adata_omics1.obsm[emb_type].loc[:, w[0].columns]
            result.loc[len(result.index)] = [emb_type, data, 'jaccard_' + omics_names[0],
                                             compute_jaccard(adata_omics1, 'spamv_ps')]
            adata_omics2.obsm['spamv_ps'] = adata_omics1.obsm[emb_type].loc[:, w[1].columns]
            result.loc[len(result.index)] = ['spamv', data, 'jaccard_' + omics_names[1],
                                             compute_jaccard(adata_omics2, 'spamv_ps')]

result.to_csv('results/Evaluation_interpretable.csv', index=False)
