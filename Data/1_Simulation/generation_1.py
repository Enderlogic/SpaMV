import numpy as np
import pandas
import scanpy as sc
from Methods.simulations.sim import sim

data_rna = sim('ggblocks_rna_1', nside=36, nzprob_nsp=.05, bkg_mean=.2, zi_prob=.5, Jmix=1000, seed=12)
data_pro = sim('ggblocks_pro_1', nside=36, nzprob_nsp=.05, bkg_mean=.2, Jmix=100, seed=14)
location_rna = pandas.DataFrame(data_rna.obsm['spatial'])
location_rna['idx'] = location_rna.index
location_pro = pandas.DataFrame(data_pro.obsm['spatial'])
location_pro['idx'] = location_pro.index

location_pro_reordered = location_rna[[0, 1]].merge(location_pro, on=[0, 1])
data_pro = data_pro[location_pro_reordered.idx, :]

data_gt = sim('ggblocks_gt', nside=36, nzprob_nsp=0, bkg_mean=0, zi_prob=0, Jmix=100, seed=0)
data_gt.obsm['spfac'][data_gt.obsm['spfac'].sum(1) == 0, 9] = 1
data_gt.obs['cluster'] = np.argmax(data_gt.obsm['spfac'], axis=1)
data_gt.obs['cluster'] = data_gt.obs['cluster'].astype('category')

data_rna.obs['cluster'] = data_gt.obs['cluster']
data_pro.obs['cluster'] = data_gt.obs['cluster']

# data_rna.obs['std'] = data_rna.X.std(1)
# data_pro.obs['std'] = data_pro.X.std(1)
# sc.pl.embedding(data_rna, basis='spatial', color='std', size=300)
# sc.pl.embedding(data_pro, basis='spatial', color='std', size=300)
#
# sc.pp.normalize_total(data_rna)
# sc.pp.log1p(data_rna)
# sc.pl.embedding(data_rna, basis='spatial', color=data_rna.var_names[:25], ncols=5, size=300)
# clr_normalize_each_cell(data_pro)
# sc.pl.embedding(data_pro, basis='spatial', color=data_pro.var_names[:25], ncols=5, size=300)
#
# adata_rna = anndata.AnnData(pca(data_rna, n_comps=50), obs=data_rna.obs, obsm=data_rna.obsm)
# sc.pl.embedding(adata_rna, basis='spatial', color=adata_rna.var_names[:25], ncols=5, size=300)
# adata_pro = anndata.AnnData(pca(data_pro, n_comps=50), obs=data_pro.obs, obsm=data_pro.obsm)
# sc.pl.embedding(adata_pro, basis='spatial', color=data_pro.var_names[:25], ncols=5, size=300)

sc.pl.embedding(data_rna, basis='spatial', color='cluster', size=300)
sc.pl.embedding(data_pro, basis='spatial', color='cluster', size=300)

data_rna.write_h5ad('adata_RNA.h5ad')
data_pro.write_h5ad('adata_ADT.h5ad')