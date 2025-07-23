import pandas
from pandas import get_dummies
from scipy.sparse import issparse
from sklearn.neighbors import kneighbors_graph
import scanpy as sc
import numpy as np
from sklearn.metrics import adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, homogeneity_score, v_measure_score

from .utils import RankingSimilarity


def compute_moranI(adata, key):
    # sc.pp.neighbors(adata, use_rep='spatial')
    g = kneighbors_graph(adata.obsm['spatial'], 6, mode='connectivity', metric='euclidean')
    one_hot = get_dummies(adata.obs[key])
    moranI = sc.metrics.morans_i(g, one_hot.values.T).mean()
    return moranI


def compute_supervised_scores(adata, learned_cluster_name):
    if 'cluster' in adata.obs:
        cluster = adata.obs['cluster']
        cluster_learned = adata.obs[learned_cluster_name]
        ari = adjusted_rand_score(cluster, cluster_learned)
        mi = mutual_info_score(cluster, cluster_learned)
        nmi = normalized_mutual_info_score(cluster, cluster_learned)
        ami = adjusted_mutual_info_score(cluster, cluster_learned)
        hom = homogeneity_score(cluster, cluster_learned)
        vme = v_measure_score(cluster, cluster_learned)
        return {"ARI": ari, "MI": mi, "NMI": nmi, "AMI": ami, "HOM": hom, "VME": vme,
                "Average": (ari + mi + nmi + ami + hom + vme) / 6}


def compute_jaccard(adata, key, k=50, use_rep='X_pca'):
    sc.pp.neighbors(adata, use_rep=key, key_added=key, n_neighbors=k)
    if use_rep in adata.obsm:
        sc.pp.neighbors(adata, use_rep=use_rep, key_added='X', n_neighbors=k)
    else:
        sc.pp.neighbors(adata, use_rep='X', key_added='X', n_neighbors=k)
    jaccard = ((adata.obsp[key + '_distances'].toarray() * adata.obsp['X_distances'].toarray() > 0).sum(1) / (
            adata.obsp[key + '_distances'].toarray() + adata.obsp['X_distances'].toarray() > 0).sum(1)).mean()
    return jaccard


def get_cell_probability(data, wi, quantiles_df, wj=None, max=1):
    if wj is None:
        D_wi = (data[wi] >= np.max((quantiles_df.loc[wi, "quantiles"], max))).mean()
        return D_wi

    # Find probability that they are not both zero
    D_wj = (data[wj] >= np.max((quantiles_df.loc[wj, "quantiles"], max))).mean()
    D_wi_wj = ((data[wi] >= np.max((quantiles_df.loc[wi, "quantiles"], max))) & (
            data[wj] >= np.max((quantiles_df.loc[wj, "quantiles"], max)))).mean()

    return D_wj, D_wi_wj


def compute_topic_coherence(adata, beta, topk=20, quantile=0.75, individual=False):
    data = pandas.DataFrame(adata.X.toarray() if issparse(adata.X) else adata.X, index=adata.obs_names,
                            columns=adata.var_names)
    quantiles = np.quantile(data, q=quantile, axis=0)
    quantiles_df = pandas.DataFrame(quantiles, index=adata.var_names, columns=["quantiles"])

    TC = []
    topics = [beta.nlargest(topk, i).index.tolist() for i in beta.columns]
    for beta_topk in topics:
        TC_k = 0
        counter = 0
        for i, gene in enumerate(beta_topk):
            # get D(w_i)
            D_wi = get_cell_probability(data, gene, quantiles_df)
            j = i + 1
            tmp = 0
            while j < len(beta_topk) and j > i:
                # get D(w_j) and D(w_i, w_j)
                
                D_wj, D_wi_wj = get_cell_probability(data.copy(), gene, quantiles_df, beta_topk[j])
                # get f(w_i, w_j)
                if D_wi_wj == 0:
                    f_wi_wj = 0
                else:
                    f_wi_wj = (np.log2(D_wi_wj) - np.log2(D_wi) - np.log2(D_wj)) / (-np.log2(D_wi_wj))
                # update tmp:
                tmp += f_wi_wj
                j += 1
                counter += 1
            # update TC_k
            TC_k += tmp
        TC.append(TC_k / counter)

    if individual:
        return TC
    else:
        TC = np.mean(TC)
        return TC


def compute_topic_diversity(beta, topk=20):
    TD = np.ones(beta.shape[1])
    for i in range(beta.shape[1]):
        for j in range(beta.shape[1]):
            if i != j:
                li = beta.nlargest(topk, beta.columns[i]).index.tolist()
                lj = beta.nlargest(topk, beta.columns[j]).index.tolist()
                r = RankingSimilarity(li, lj).rbo()
                if 1 - r < TD[i]:
                    TD[i] = 1 - r
    return TD.mean()
