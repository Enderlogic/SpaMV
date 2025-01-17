from pandas import get_dummies
from sklearn.neighbors import kneighbors_graph
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, homogeneity_score, v_measure_score

def moranI_score(adata, key):
    # sc.pp.neighbors(adata, use_rep='spatial')
    g = kneighbors_graph(adata.obsm['spatial'], 6, mode='connectivity', metric='euclidean')
    one_hot = get_dummies(adata.obs[key])
    moranI = sc.metrics.morans_i(g, one_hot.values.T).mean()
    return moranI

def compute_supervised_scores(adata, z):
    if 'cluster' in adata.obs:
        cluster = adata.obs['cluster']
        cluster_learned = z.idxmax(1)
        ari = adjusted_rand_score(cluster, cluster_learned)
        mi = mutual_info_score(cluster, cluster_learned)
        nmi = normalized_mutual_info_score(cluster, cluster_learned)
        ami = adjusted_mutual_info_score(cluster, cluster_learned)
        hom = homogeneity_score(cluster, cluster_learned)
        vme = v_measure_score(cluster, cluster_learned)
        return {"ari": ari, "mi": mi, "nmi": nmi, "ami": ami, "hom": hom, "vme": vme,
                "average": (ari + mi + nmi + ami + hom + vme) / 6}

def calculate_jaccard(adata, key, k=50):
    sc.pp.neighbors(adata, use_rep=key, key_added=key, n_neighbors=k)
    sc.pp.neighbors(adata, use_rep='X_pca', key_added='X_pca', n_neighbors=k)
    jaccard = ((adata.obsp[key + '_distances'].toarray() * adata.obsp['X_pca_distances'].toarray() > 0).sum(1) / (
            adata.obsp[key + '_distances'].toarray() + adata.obsp['X_pca_distances'].toarray() > 0).sum(1)).mean()
    return jaccard