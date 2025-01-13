from pandas import get_dummies
from sklearn.neighbors import kneighbors_graph
import scanpy as sc

def moranI_score(adata, key):
    # sc.pp.neighbors(adata, use_rep='spatial')
    g = kneighbors_graph(adata.obsm['spatial'], 6, mode='connectivity', metric='euclidean')
    one_hot = get_dummies(adata.obs[key])
    moranI = sc.metrics.morans_i(g, one_hot.values.T).mean()
    return moranI


def jaccard_scores(adata, adata_omics, emb, feat, k=50):
    sc.pp.neighbors(adata, use_rep=emb, key_added=emb, n_neighbors=k)
    sc.pp.neighbors(adata_omics, use_rep=feat, key_added=feat, n_neighbors=k)
    jaccard = ((adata.obsp[emb + '_distances'].toarray() * adata_omics.obsp[feat + '_distances'].toarray() > 0).sum(
        1) / (adata.obsp[emb + '_distances'].toarray() + adata_omics.obsp[feat + '_distances'].toarray() > 0).sum(
        1)).mean()
    return jaccard
