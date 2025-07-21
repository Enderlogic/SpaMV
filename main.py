import argparse
import os

import matplotlib.pyplot as plt
import scanpy as sc
import torch
import torch.nn.functional as F

import wandb
from SpaMV.metrics import compute_supervised_scores, compute_jaccard
from SpaMV.spamv import SpaMV
from SpaMV.utils import ST_preprocess, clr_normalize_each_cell, clustering

sc._settings.ScanpyConfig.figdir = '.'
# Argument to parse
parser = argparse.ArgumentParser(description='SpaMV Experiment')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--zp_dim_omics1', type=int, default=32, help='latent modality 1-specific dimensionality')
parser.add_argument('--zp_dim_omics2', type=int, default=32, help='latent modality 2-specific dimensionality')
parser.add_argument('--zs_dim', type=int, default=32, help='latent shared dimensionality')
parser.add_argument('--hidden_dim', type=int, default=256, help='hidden layer size')
parser.add_argument('--heads', type=int, default=1, help='number of heads in GAT')
parser.add_argument('--n_neighbors', type=int, default=20, help='number of neighbors in GNN')
parser.add_argument('--seed', type=int, default=10, help='random seed')
parser.add_argument('--kl', type=float, default=1, help='beta hyperparameter in VAE objective')
parser.add_argument('--beta_0', type=float, default=1)
parser.add_argument('--beta_1', type=float, default=1)
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--interpretable', type=bool, default=False, help='whether to use interpretable mode')
parser.add_argument('--reweight', type=bool, default=False, help='reweight the loss of different modalities')
parser.add_argument('--detach', type=bool, default=True, help='whether to use detach')
parser.add_argument('--distinguish', type=bool, default=True, help='whether to use distinguish')

# Args
args = parser.parse_args()
wandb.login()
root_path = "Data/"
data = "Dataset7_Mouse_Brain_ATAC"
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
folder_path = 'results/' + data + '/'
print(data)
adata_combined = []
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
    if "RNA" in filename or "peak" in filename or "SM" in filename:
        adata = ST_preprocess(adata)
        recon_types.append('zinb')
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
for v in adata_combined[0].var_names:
    if v in adata_combined[1].var_names:
        adata_combined[0].var.rename(index={v: v + '_' + omics_names[0]}, inplace=True)
adata_combined[0] = adata_combined[0][adata_combined[0].obs_names.intersection(adata_combined[1].obs_names), :]
adata_combined[1] = adata_combined[1][adata_combined[1].obs_names.intersection(adata_combined[0].obs_names), :]

weights = [1, 1]
if args.reweight:
    if adata_combined[0].n_vars > adata_combined[1].n_vars:
        weights[1] = adata_combined[0].n_vars / adata_combined[1].n_vars
    else:
        weights[0] = adata_combined[1].n_vars / adata_combined[0].n_vars

wandb.init(project=data, config=args,
           name=str(args.zp_dim_omics1) + '_' + str(args.zs_dim) + '_' + str(args.zp_dim_omics2) + '_' + str(
               args.kl) + '_' + str(args.learning_rate) + '_' + str(args.reweight) + '_' + str(args.seed) + '_' + str(
               args.heads) + '_' + str(args.n_neighbors) + '_' + str(args.interpretable) + '_' + str(
               args.detach) + '_' + str(args.distinguish))
model = SpaMV(adata_combined, zs_dim=args.zs_dim, zp_dims=[args.zp_dim_omics1, args.zp_dim_omics2], weights=weights,
              beta=[args.beta_0, args.beta_1], interpretable=args.interpretable, hidden_dim=args.hidden_dim,
              heads=args.heads, n_neighbors=args.n_neighbors, random_seed=args.seed, recon_types=recon_types,
              omics_names=omics_names, min_epochs=50, max_epochs_stage1=args.epochs, min_kl=args.kl, max_kl=args.kl,
              learning_rate=args.learning_rate, folder_path=folder_path, n_cluster=n_cluster, test_mode=True,
              detach=args.detach, distinguish=args.distinguish, device=device)
model.train()
z = model.get_embedding()
adata_combined[0].obsm['spamv'] = F.normalize(z, p=2, eps=1e-12, dim=1).detach().cpu().numpy()
clustering(adata_combined[0], n_clusters=n_cluster, key='spamv', add_key='spamv', method='mclust', use_pca=True)
sc.pl.embedding(adata_combined[0], color='spamv', basis='spatial', size=100, title='All embeddings', show=False)
plt.tight_layout()
plt.savefig('results/' + data + '/clustering_all.pdf')

adata_combined[0].obsm['spamv_shared'] = F.normalize(z[:, :args.zs_dim], p=2, eps=1e-12, dim=1).detach().cpu().numpy()
clustering(adata_combined[0], n_clusters=n_cluster, key='spamv_shared', add_key='spamv_shared', method='mclust',
           use_pca=True)
sc.pl.embedding(adata_combined[0], color='spamv_shared', basis='spatial', size=100, title='Shared embeddings',
                show=False)
plt.tight_layout()
plt.savefig('results/' + data + '/clustering_shared.pdf')
supervised_scores = compute_supervised_scores(adata_combined[0], 'spamv')
if 'cluster' in adata_combined[0].obs:
    print("ari: " + str(supervised_scores['ari']) + "\naverage: " + str(supervised_scores['average']))

for i in range(len(omics_names)):
    adata_combined[i].obsm['zs'] = F.normalize(z[:, :args.zs_dim], p=2, eps=1e-12, dim=1).detach().cpu().numpy()
    adata_combined[i].obsm['zs+zp1'] = F.normalize(z[:, :args.zs_dim + args.zp_dim_omics1], p=2, eps=1e-12,
                                                   dim=1).detach().cpu().numpy()
    adata_combined[i].obsm['zs+zp2'] = F.normalize(torch.cat((z[:, :args.zs_dim], z[:, -args.zp_dim_omics2:]), dim=1),
                                                   p=2, eps=1e-12, dim=1).detach().cpu().numpy()
    adata_combined[i].obsm['zp1'] = F.normalize(z[:, args.zs_dim:args.zs_dim + args.zp_dim_omics1], p=2, eps=1e-12,
                                                dim=1).detach().cpu().numpy()
    adata_combined[i].obsm['zp2'] = F.normalize(z[:, -args.zp_dim_omics2:], p=2, eps=1e-12,
                                                dim=1).detach().cpu().numpy()
    for emb_type in ['zs', 'zs+zp1', 'zs+zp2', 'zp1', 'zp2']:
        print(omics_names[i], emb_type)
        print(compute_jaccard(adata_combined[i], emb_type))
