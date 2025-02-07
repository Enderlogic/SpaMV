import argparse
import os

import numpy as np
import pandas
import scanpy as sc
from matplotlib import pyplot as plt
from tqdm import tqdm

import wandb
from SpaMV.spamv import SpaMV
from SpaMV.utils import ST_preprocess, clr_normalize_each_cell, plot_embedding_results, cosine_similarity

sc._settings.ScanpyConfig.figdir = '.'
# Argument to parse
parser = argparse.ArgumentParser(description='SpaMV Experiment')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--data', type=str, default='Dataset12_Human_Lymph_Node_D1')
parser.add_argument('--zp_dim_omics1', type=int, default=10, help='latent modality 1-specific dimensionality')
parser.add_argument('--zp_dim_omics2', type=int, default=10, help='latent modality 2-specific dimensionality')
parser.add_argument('--zs_dim', type=int, default=10, help='latent shared dimensionality')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden layer dimensionality')
parser.add_argument('--heads', type=int, default=1, help='number of heads in GAT')
parser.add_argument('--n_neighbors', type=int, default=20, help='number of neighbors in GNN')
parser.add_argument('--seed', type=int, default=20, help='random seed')
parser.add_argument('--beta', type=float, default=1, help='beta hyperparameter in VAE objective')
parser.add_argument('--learning_rate', type=float, default=1e-2, help='learning rate')
parser.add_argument('--interpretable', type=bool, default=True, help='whether to use interpretable mode')
parser.add_argument('--reweight', type=bool, default=False, help='reweight the loss of different modalities')
parser.add_argument('--detach', type=bool, default=True, help='whether to use detach')
parser.add_argument('--distinguish', type=bool, default=True, help='whether to use distinguish')

# Args
args = parser.parse_args()
wandb.login()
root_path = "Data/"
files = [item for item in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, item)) and args.data in item]
if len(files) != 1:
    raise ValueError('Please provide correct and unique part of data name')
else:
    data = files[0]
folder_path = 'results/' + data + '/'
print(data)
adata_combined = []
adata_raw = []
omics_names = []
recon_types = []
for filename in os.listdir(root_path + data + "/"):
    adata = sc.read_h5ad(root_path + data + "/" + filename)
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=50 if adata.n_vars > 100 else 5)
    adata_raw.append(adata.copy())
    if "RNA" in filename or "peak" in filename or 'SM' in filename:
        adata = ST_preprocess(adata, prune=True)
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
adata_combined[0] = adata_combined[0][adata_combined[0].obs_names.intersection(adata_combined[1].obs_names), :]
adata_combined[1] = adata_combined[1][adata_combined[0].obs_names.intersection(adata_combined[1].obs_names), :]
adata_raw[0] = adata_raw[0][adata_combined[0].obs_names.intersection(adata_combined[1].obs_names), :]
adata_raw[1] = adata_raw[1][adata_combined[0].obs_names.intersection(adata_combined[1].obs_names), :]

weights = [1, 1]
if args.reweight:
    if adata_combined[0].n_vars > adata_combined[1].n_vars:
        weights[1] = adata_combined[0].n_vars / adata_combined[1].n_vars
    else:
        weights[0] = adata_combined[1].n_vars / adata_combined[0].n_vars

wandb.init(project=data, config=args, settings=dict(init_timeout=600),
           name=str(args.zp_dim_omics1) + '_' + str(args.zs_dim) + '_' + str(args.zp_dim_omics2) + '_' + str(
               args.beta) + '_' + str(args.learning_rate) + '_' + str(args.reweight) + '_' + str(args.seed) + '_' + str(
               args.heads) + '_' + str(args.n_neighbors) + '_' + str(args.interpretable))
model = SpaMV(adata_combined, zs_dim=args.zs_dim, zp_dims=[args.zp_dim_omics1, args.zp_dim_omics2],
              weights=weights, interpretable=args.interpretable, hidden_dim=args.hidden_dim,
              heads=args.heads, n_neighbors=args.n_neighbors, random_seed=args.seed,
              recon_types=recon_types, omics_names=omics_names, min_epochs=50,
              max_epochs=args.epochs, min_kl=args.beta, max_kl=args.beta, learning_rate=args.learning_rate,
              folder_path=folder_path, test_mode=False, detach=args.detach, distinguish=args.distinguish)
model.train()

z, w = model.get_embedding_and_feature_by_topic()
z.to_csv(folder_path + 'embedding_without_map.csv')
w[0].to_csv(folder_path + 'weight_' + omics_names[0] + '_without_map.csv')
w[1].to_csv(folder_path + 'weight_' + omics_names[1] + '_without_map.csv')
plot_embedding_results(adata_raw, omics_names, z, w, save=True, show=False, corresponding_features=False,
                       folder_path=folder_path, file_name='spamv_interpretable_' + str(args.detach) + '_' + str(
        args.distinguish) + '_without_features_without_map.pdf')
plot_embedding_results(adata_raw, omics_names, z, w, save=True, show=False, corresponding_features=True,
                       folder_path=folder_path, file_name='spamv_interpretable_' + str(args.detach) + '_' + str(
        args.distinguish) + '_with_features_without_map.pdf')

z, w = model.get_embedding_and_feature_by_topic(map=True)
z.to_csv(folder_path + 'embedding_map.csv')
w[0].to_csv(folder_path + 'weight_' + omics_names[0] + '_map.csv')
w[1].to_csv(folder_path + 'weight_' + omics_names[1] + '_map.csv')
plot_embedding_results(adata_raw, omics_names, z, w, save=True, show=False, corresponding_features=False,
                       folder_path=folder_path, file_name='spamv_interpretable_' + str(args.detach) + '_' + str(
        args.distinguish) + '_without_features_with_map.pdf')
plot_embedding_results(adata_raw, omics_names, z, w, save=True, show=False, corresponding_features=True,
                       folder_path=folder_path, file_name='spamv_interpretable_' + str(args.detach) + '_' + str(
        args.distinguish) + '_with_features_with_map.pdf')

plot_embedding_results(adata_raw, omics_names, z, w, folder_path=folder_path, file_name='spamv_interpretable.pdf')

cs_omics1 = pandas.DataFrame(np.zeros((adata_combined[0].shape[1], z.shape[1])), columns=z.columns,
                             index=adata_combined[0].var_names)
cs_omics2 = pandas.DataFrame(np.zeros((adata_combined[1].shape[1], z.shape[1])), columns=z.columns,
                             index=adata_combined[1].var_names)

for topic in tqdm(z.columns):
    for feature in adata_combined[0].var_names:
        cs_omics1.loc[feature, topic] = cosine_similarity(adata_combined[0][:, feature].X.toarray()[:, 0],
                                                          z.loc[:, topic])
    for feature in adata_combined[1].var_names:
        cs_omics2.loc[feature, topic] = cosine_similarity(adata_combined[1][:, feature].X.toarray()[:, 0],
                                                          z.loc[:, topic])
plot_df = pandas.DataFrame()
plot_df['label'] = cs_omics1.columns
plot_df[omics_names[0]] = [cs_omics1.nlargest(10, topic).loc[:, topic].mean() for topic in cs_omics1.columns]
plot_df[omics_names[0] + '_std'] = [cs_omics1.nlargest(10, topic).loc[:, topic].std() for topic in
                                    cs_omics1.columns]
plot_df[omics_names[1]] = [cs_omics2.nlargest(10, topic).loc[:, topic].mean() for topic in cs_omics2.columns]
plot_df[omics_names[1] + '_std'] = [cs_omics2.nlargest(10, topic).loc[:, topic].std() for topic in
                                    cs_omics2.columns]
plot_df.plot.bar(x='label', y=omics_names,
                 yerr=plot_df[[omics_names[0] + '_std', omics_names[1] + '_std']].T.values)
plt.savefig(folder_path + 'cosine_similarity.pdf')
