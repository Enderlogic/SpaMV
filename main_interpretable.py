import argparse
import os

import anndata
import numpy as np
import pandas
import scanpy as sc
from matplotlib import pyplot as plt
from numpy.linalg import norm
from sklearn.metrics import adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, homogeneity_score, v_measure_score
from tqdm import tqdm

from SpatialGlue.utils import clustering

import wandb
from SpaMV.spamv import SpaMV
from SpaMV.utils import ST_preprocess, clr_normalize_each_cell, plot_results


def cosine_similarity(A, B):
    return np.dot(A, B) / (norm(A) * norm(B))


sc._settings.ScanpyConfig.figdir = '.'
# Argument to parse
parser = argparse.ArgumentParser(description='SpaMV Experiment')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--zp_dim_omics1', type=int, default=8, help='latent modality 1-specific dimensionality')
parser.add_argument('--zp_dim_omics2', type=int, default=8, help='latent modality 2-specific dimensionality')
parser.add_argument('--zs_dim', type=int, default=8, help='latent shared dimensionality')
parser.add_argument('--hidden_size', type=int, default=128, help='hidden layer size')
parser.add_argument('--heads', type=int, default=1, help='number of heads in GAT')
parser.add_argument('--n_neighbors', type=int, default=20, help='number of neighbors in GNN')
parser.add_argument('--seed', type=int, default=1214, help='random seed')
parser.add_argument('--beta', type=float, default=1, help='beta hyperparameter in VAE objective')
parser.add_argument('--learning_rate', type=float, default=1e-2, help='learning rate')
parser.add_argument('--interpretable', type=bool, default=True, help='whether to use interpretable mode')
parser.add_argument('--reweight', type=bool, default=False, help='reweight the loss of different modalities')

# Args
args = parser.parse_args()
wandb.login()
root_path = "Data/"
data = "Dataset11_Human_Lymph_Node_A1"
folder_path = 'results/' + data + '/'
print(data)
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
    if "RNA" in filename or "peak" in filename:
        adata = ST_preprocess(adata, prune=True)
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

wandb.init(project=data, config=args,
           name=str(args.zp_dim_omics1) + '_' + str(args.zs_dim) + '_' + str(args.zp_dim_omics2) + '_' + str(
               args.beta) + '_' + str(args.learning_rate) + '_' + str(args.reweight) + '_' + str(args.seed) + '_' + str(
               args.heads) + '_' + str(args.n_neighbors) + '_' + str(args.interpretable))
model = SpaMV([adata_omics1, adata_omics2], zs_dim=args.zs_dim, zp_dims=[args.zp_dim_omics1, args.zp_dim_omics2],
              weights=[weight_omics1, weight_omics2], interpretable=args.interpretable, hidden_size=args.hidden_size,
              heads=args.heads, n_neighbors=args.n_neighbors, random_seed=args.seed,
              recon_types=[recon_type_omics1, recon_type_omics2])
model.train(min_epochs=100, max_epochs=args.epochs, min_kl=args.beta, max_kl=args.beta,
            learning_rate=args.learning_rate, folder_path=folder_path, n_cluster=n_cluster, omics_names=omics_names,
            test_mode=True)
z = model.get_embedding()

adata = anndata.concat([adata_raw[0], adata_raw[1]], join='outer', axis=1)
adata.obsm['spatial'] = adata_raw[0].obsm['spatial']
if 'cluster' in adata_raw[0].obs:
    adata.obs['cluster'] = adata_raw[0].obs['cluster']
elif 'cluster' in adata_raw[1].obs:
    adata.obs['cluster'] = adata_raw[1].obs['cluster']

plot_results(model, [adata_omics1, adata_omics2], omics_names, [args.zp_dim_omics1, args.zp_dim_omics2],
             args.zs_dim, folder_path, file_name='spamv_interpretable.pdf')
cs_omics1 = pandas.DataFrame(np.zeros((adata_omics1.shape[1], z.shape[1])), columns=z.columns,
                             index=adata_omics1.var_names)
cs_omics2 = pandas.DataFrame(np.zeros((adata_omics2.shape[1], z.shape[1])), columns=z.columns,
                             index=adata_omics2.var_names)

for topic in tqdm(z.columns):
    for feature in adata_omics1.var_names:
        cs_omics1.loc[feature, topic] = cosine_similarity(adata_omics1[:, feature].X.toarray()[:, 0],
                                                          z.loc[:, topic])
    for feature in adata_omics2.var_names:
        cs_omics2.loc[feature, topic] = cosine_similarity(adata_omics2[:, feature].X.toarray()[:, 0],
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
if 'cluster' in adata.obs:
    cluster = adata.obs['cluster']
    if args.interpretable:
        cluster_learned = z.idxmax(1)
    else:
        adata.obsm['spamv'] = z.values()
        clustering(adata, key='spamv', add_key='spamv', n_clusters=n_cluster, method='mclust', use_pca=True)
        cluster_learned = adata.obs['spamv']
    ari = adjusted_rand_score(cluster, cluster_learned)
    mi = mutual_info_score(cluster, cluster_learned)
    nmi = normalized_mutual_info_score(cluster, cluster_learned)
    ami = adjusted_mutual_info_score(cluster, cluster_learned)
    hom = homogeneity_score(cluster, cluster_learned)
    vme = v_measure_score(cluster, cluster_learned)
    print("ari: " + str(ari) + "\naverage: " + str((ari + mi + nmi + ami + hom + vme) / 6))
