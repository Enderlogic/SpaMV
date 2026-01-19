import numpy as np
import pandas

from SpaMV.sim import sim

data_rna = sim('ggblocks_rna_1', nside=36, nzprob_nsp=.05, bkg_mean=.2, zi_prob=.5, Jmix=1000, seed=12)
data_pro = sim('ggblocks_pro_1', nside=36, nzprob_nsp=.05, bkg_mean=.2, Jmix=100, seed=14)
location_rna = pandas.DataFrame(data_rna.obsm['spatial'])
location_rna['idx'] = location_rna.index
location_pro = pandas.DataFrame(data_pro.obsm['spatial'])
location_pro['idx'] = location_pro.index

location_pro_reordered = location_rna[[0, 1]].merge(location_pro, on=[0, 1])
data_pro = data_pro[location_pro_reordered.idx, :].copy()

data_gt = sim('ggblocks_gt', nside=36, nzprob_nsp=0, bkg_mean=0, zi_prob=0, Jmix=100, seed=0).copy()

spfac = data_gt.obsm['spfac'].copy()
mask = spfac.sum(1) == 0
spfac[mask, 9] = 1
data_gt.obsm['spfac'] = spfac
data_gt.obs['cluster'] = np.argmax(data_gt.obsm['spfac'], axis=1)
data_gt.obs['cluster'] = data_gt.obs['cluster'].astype('category')

data_rna = data_rna.copy()
data_pro = data_pro.copy()
data_rna.obs['cluster'] = data_gt.obs['cluster']
data_pro.obs['cluster'] = data_gt.obs['cluster']

data_rna.write_h5ad('adata_RNA.h5ad')
data_pro.write_h5ad('adata_ADT.h5ad')