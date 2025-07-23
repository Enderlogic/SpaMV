import math
import os.path
import anndata
# import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import sklearn
import torch
from matplotlib import pyplot as plt
# from natsort import natsorted
from numpy.linalg import norm
from pandas import DataFrame
from scanpy.plotting import embedding, spatial
from scipy.sparse import issparse
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import coalesce, from_scipy_sparse_matrix
from squidpy.pl import spatial_scatter
from scipy.stats import pearsonr
import seaborn as sns
# from sklearn.decomposition import PCA
# from scipy.sparse.csr import csr_matrix
# from scipy.sparse.csc import csc_matrix
from typing import List, Optional, Union
from anndata import AnnData
from tqdm import tqdm


def construct_graph_by_coordinate(cell_position, neighborhood_depth=3, device='cpu'):
    '''
    Constructing spatial neighbor graph according to spatial coordinates.
    Args:
        cell_position: ndarray of shape (n_cells, 2)
        neighborhood_depth: The Neighborhood Depth parameter determines the number of layers of neighbors to consider
                            when calculating the neighborhood of each node in a network.
                            When set to 1, only the node itself is considered.
                            When set to 2, the node and its immediate neighbors (those directly connected to it) are included.
                            When set to 3, the node, its immediate neighbors, and the neighbors of those immediate neighbors (second-degree neighbors) are considered.
    Returns:
        ndarray of shape (n_edges, 2)
    '''
    dist_sort = np.sort(
        np.unique((cell_position[:, 0] - cell_position[0, 0]) ** 2 + (cell_position[:, 1] - cell_position[0, 1]) ** 2))
    threshold = dist_sort[neighborhood_depth]
    edge_index = []
    for i in tqdm(range(cell_position.shape[0])):
        adj = np.where((cell_position[:, 0] - cell_position[i, 0]) ** 2 + (
                cell_position[:, 1] - cell_position[i, 1]) ** 2 < threshold)[0]
        for j in adj:
            edge_index.append([j, i])
    edge_index = torch.tensor(edge_index, device=device).T
    return edge_index


def construct_graph_by_feature(adata, n_neighbors=20, mode="connectivity", metric="correlation", device="cpu"):
    """Constructing feature neighbor graph according to expresss profiles"""
    if 'X_lsi' in adata.obsm:
        feature_graph = kneighbors_graph(adata.obsm['X_lsi'], n_neighbors, mode=mode, metric=metric)
    elif 'X_pca' in adata.obsm:
        feature_graph = kneighbors_graph(adata.obsm['X_pca'], n_neighbors, mode=mode, metric=metric)
    else:
        feature_graph = kneighbors_graph(adata.X.toarray() if issparse(adata.X) else adata.X, n_neighbors, mode=mode,
                                         metric=metric)
    return from_scipy_sparse_matrix(feature_graph)[0].to(device=device)


def adjacent_matrix_preprocessing(adata, neighborhood_depth=2, neighborhood_embedding=20, device="cpu"):
    """Converting dense adjacent matrix to sparse adjacent matrix"""
    ######################################## construct spatial graph ########################################
    edge_index = construct_graph_by_coordinate(adata.obsm['spatial'], neighborhood_depth, device)

    ######################################## construct feature graph ########################################
    if neighborhood_embedding > 0:
        edge_index_feature = construct_graph_by_feature(adata, n_neighbors=neighborhood_embedding, device=device)
        edge_index = coalesce(torch.cat([edge_index, edge_index_feature], dim=1))
    return edge_index


def get_init_bg(x):
    bg_init = []
    for data in x:
        data = data / data.sum(axis=1, keepdims=True)
        means = data.nanmean(dim=0)
        bg_init.append((means + 1e-15).log())
    return bg_init


def remove_box(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_embedding_results(adatas, omics_names, topic_abundance, feature_topics, save=True, folder_path=None,
                           file_name=None, show=False, corresponding_features=True, size=350, crop_coord=None, rb=True,
                           img_alpha=1):
    element_names = []
    for omics_name in omics_names:
        if omics_name in ["Transcriptomics", "Transcriptome"] or "H3K27" in omics_name:
            element_names.append("Gene")
        elif omics_name in ["Proteomics", "Proteome"]:
            element_names.append("Protein")
        elif omics_name in ["Epigenomics", "Epigenome"]:
            element_names.append("Region of open chromatin")
        elif omics_name in ["Metabolomics", "Metabolome", "Metabonomics", "Metabonome"]:
            element_names.append("Metabolite")
        else:
            element_names.append(omics_name + '\'s feature')
    zs_dim = len([item for item in topic_abundance.columns if 'Shared' in item])
    n_omics = len(adatas)
    zp_dims = []
    for i in range(n_omics):
        zp_dims.append(len([item for item in topic_abundance.columns if omics_names[i] in item]))
    n_col = max(zp_dims + [zs_dim])
    n_row = n_omics * 3 + 1 if corresponding_features else n_omics + 1
    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 5, n_row * 5))
    if zs_dim < n_col:
        for i in range(n_col - zs_dim):
            for j in range(1 + n_omics if corresponding_features else 1):
                axes[j, zs_dim + i].axis('off')
    for i, zp_dim in enumerate(zp_dims):
        if zp_dim < n_col:
            for j in range(n_col - zp_dim):
                for k in range(2 if corresponding_features else 1):
                    axes[1 + (i + 1) * n_omics + k if corresponding_features else 1 + i + k, zp_dim + j].axis('off')
    adatas[0].obs[topic_abundance.columns] = topic_abundance.values
    adatas[1].obs[topic_abundance.columns] = topic_abundance.values
    for i in range(zs_dim):
        topic_name = topic_abundance.columns[i]
        if 'spatial' not in adatas[0].uns:
            embedding(adatas[0], color=topic_name, vmax='p99', size=size, show=False, basis='spatial', ax=axes[0, i])
        else:
            spatial_scatter(adatas[0], color=topic_name, ax=axes[0, i], crop_coord=crop_coord, img_alpha=img_alpha)
        if corresponding_features:
            for j in range(n_omics):
                mrf = feature_topics[omics_names[j]].nlargest(1, topic_name).index[0]
                if 'spatial' not in adatas[j].uns:
                    embedding(adatas[j], color=mrf, vmax='p99', basis='spatial', size=size, cmap='coolwarm', show=False,
                              ax=axes[1 + j, i],
                              title=mrf + '\nHighest ranking ' + element_names[j] + '\nw.r.t. ' + topic_name)
                else:
                    spatial(adatas[j], color=mrf, vmax='p99', cmap='coolwarm', show=False, ax=axes[1 + j, i],
                            title=mrf + '\nMost relevant ' + element_names[j] + '\nw.r.t. ' + topic_name)
    for i in range(zs_dim):
        for j in range(n_omics + 1 if corresponding_features else 1):
            axes[j, i].set_xlabel('')  # Remove x-axis label
            axes[j, i].set_ylabel('')  # Remove y-axis label
            axes[j, i].set_xticks([])  # Remove x-axis ticks
            axes[j, i].set_yticks([])  # Remove y-axis ticks
            if rb:
                remove_box(axes[j, i])
    for i in range(n_omics):
        for j in range(zp_dims[i]):
            topic_name = feature_topics[omics_names[i]].columns[zs_dim + j]
            if 'spatial' not in adatas[0].uns:
                embedding(adatas[0], color=topic_name, vmax='p99', size=size, show=False, basis='spatial',
                          ax=axes[1 + n_omics + i * n_omics if corresponding_features else 1 + i, j])
            else:
                spatial_scatter(adatas[0], color=topic_name,
                                ax=axes[1 + n_omics + i * n_omics if corresponding_features else 1 + i, j],
                                crop_coord=crop_coord, img_alpha=img_alpha)
            if corresponding_features:
                mrf = feature_topics[omics_names[i]].nlargest(1, topic_name).index[0]
                if 'spatial' not in adatas[i].uns:
                    embedding(adatas[i], color=mrf, vmax='p99', size=size, show=False, cmap='coolwarm', basis='spatial',
                              title=mrf + '\nHighest ranking ' + element_names[i] + '\nw.r.t. ' + topic_name,
                              ax=axes[1 + n_omics + i * n_omics + 1, j])
                else:
                    spatial(adatas[i], color=mrf, vmax='p99', cmap='coolwarm', show=False,
                            ax=axes[1 + n_omics + i * n_omics + 1, j],
                            title=mrf + '\nMost relevant ' + element_names[i] + '\nw.r.t. ' + topic_name)
    for i in range(n_omics):
        for j in range(zp_dims[i]):
            if corresponding_features:
                for k in range(2):
                    axes[1 + n_omics * (i + 1) + k, j].set_xlabel('')
                    axes[1 + n_omics * (i + 1) + k, j].set_ylabel('')
                    axes[1 + n_omics * (i + 1) + k, j].set_xticks([])
                    axes[1 + n_omics * (i + 1) + k, j].set_yticks([])
                    if rb:
                        remove_box(axes[1 + n_omics * (i + 1) + k, j])
            else:
                axes[1 + i, j].set_xlabel('')
                axes[1 + i, j].set_ylabel('')
                axes[1 + i, j].set_xticks([])
                axes[1 + i, j].set_yticks([])
                if rb:
                    remove_box(axes[1 + i, j])
    plt.tight_layout()
    if save:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plt.savefig(folder_path + 'spamv.pdf' if file_name is None else folder_path + file_name)
    if show:
        plt.show()
    plt.close()


def clr_normalize_each_cell(adata, inplace=True):
    """Normalize count vector for each cell, i.e. for each row of .X"""

    import numpy as np
    import scipy

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()

    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata


def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf


def lsi(adata: anndata.AnnData, n_components: int = 20, use_highly_variable: Optional[bool] = None, random_state=0,
        key_added='X_lsi', **kwargs) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)
    """
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, random_state=random_state, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm[key_added] = X_lsi[:, 1:]


def preprocess_dc(datasets: List[AnnData], omics_names: List[str], prune: bool = True, hvg: bool = True,
                  n_top_genes: int = 3000, normalization: bool = True, target_sum: float = 1e4, log1p: bool = True,
                  scale: bool = False):
    '''
    # preprocess step for domain clustering
    '''
    # prune low quality features and spots
    if prune:
        for i in range(len(datasets)):
            sc.pp.filter_genes(datasets[i], min_cells=round(datasets[i].n_obs / 100))
            sc.pp.filter_cells(datasets[i], min_genes=round(datasets[i].n_vars / 100))
            if any(substring.lower() in omics_names[i].lower() for substring in ['Transcriptomics', 'RNA', 'Gene']):
                datasets[i].var['mt'] = np.logical_or(datasets[i].var_names.str.startswith('MT-'),
                                                      datasets[i].var_names.str.startswith('mt-'))
                datasets[i].var['rb'] = datasets[i].var_names.str.startswith(('RP', 'Rp', 'rp'))
                sc.pp.calculate_qc_metrics(datasets[i], qc_vars=['mt'], inplace=True)
                mask_cell = datasets[i].obs['pct_counts_mt'] < 100
                mask_gene = np.logical_and(~datasets[i].var['mt'], ~datasets[i].var['rb'])
                datasets[i] = datasets[i][mask_cell, mask_gene]
    remained_spots = datasets[0].obs_names
    n_comps = min(50, datasets[0].n_vars - 1)
    for i in range(1, len(datasets)):
        remained_spots = remained_spots.intersection(datasets[i].obs_names)
        n_comps = min(n_comps, datasets[i].n_vars - 1)
    for i in range(len(datasets)):
        datasets[i] = datasets[i][remained_spots, :]
        if any(substring.lower() in omics_names[i].lower() for substring in ['Transcriptomics', 'RNA', 'Gene']):
            if hvg:
                sc.pp.highly_variable_genes(datasets[i], flavor='seurat_v3', n_top_genes=n_top_genes, subset=False)
            if normalization:
                sc.pp.normalize_total(datasets[i], target_sum=target_sum)
            if hvg:
                datasets[i] = datasets[i][:, datasets[i].var.highly_variable]
            if log1p:
                sc.pp.log1p(datasets[i])
            if scale:
                sc.pp.scale(datasets[i])
            sc.pp.pca(datasets[i], n_comps=n_comps, key_added='embedding')
        elif any(substring.lower() in omics_names[i].lower() for substring in ['Proteomics', 'ADT', 'Protein']):
            if normalization:
                datasets[i] = clr_normalize_each_cell(datasets[i])
            if scale:
                sc.pp.scale(datasets[i])
            sc.pp.pca(datasets[i], n_comps=n_comps, key_added='embedding')
        elif any(substring.lower() in omics_names[i].lower() for substring in ['Epigenomics', 'peaks']):
            lsi(datasets[i], use_highly_variable=False, n_components=n_comps + 1, key_added='embedding')
        elif any(substring.lower() in omics_names[i].lower() for substring in ['Metabolomics', 'Metabolite']):
            if normalization:
                sc.pp.normalize_total(datasets[i], target_sum=target_sum)
            if log1p:
                sc.pp.log1p(datasets[i])
            if scale:
                sc.pp.scale(datasets[i])
            sc.pp.pca(datasets[i], n_comps=n_comps, key_added='embedding')
        elif any(substring.lower() in omics_names[i].lower() for substring in ['H3K27ac', 'H3K27me3']):
            if scale:
                sc.pp.scale(datasets[i])
            sc.pp.pca(datasets[i], n_comps=n_comps, key_added='embedding')
        else:
            raise Exception("To be completed for " + omics_names[i] + ".")
        datasets[i] = anndata.AnnData(datasets[i].obsm['embedding'], obs=datasets[i].obs, obsm=datasets[i].obsm,
                                      uns=datasets[i].uns)
    return datasets


def log_mean_exp(value, dim=0, keepdim=False):
    return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))


def mclust_R(adata, num_cluster, used_obsm='emb_pca', add_key='SpaMV', random_seed=2025):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, 'EEE')
    mclust_res = np.array(res[-2])

    adata.obs[add_key] = mclust_res
    adata.obs[add_key] = adata.obs[add_key].astype('int')
    adata.obs[add_key] = adata.obs[add_key].astype('category')
    return adata


def pca(adata, use_reps=None, n_comps=10):
    """Dimension reduction with PCA algorithm"""
    # pca = PCA(n_components=n_comps, random_state=0)
    if use_reps is not None:
        tmp = AnnData(adata.obsm[use_reps])
        sc.pp.pca(tmp, n_comps=n_comps)
        feat_pca = tmp.obsm['X_pca']
    else:
        sc.pp.pca(adata, n_comps=n_comps)
        feat_pca = adata.obsm['X_pca']
        # if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
            # feat_pca = pca.fit_transform(adata.X.toarray())
        # else:
            # feat_pca = pca.fit_transform(adata.X)

    return feat_pca


def clustering(adata, n_clusters=7, key='emb', add_key='SpaMV', use_pca=True, n_comps=20):
    """\
    Spatial clustering based the latent representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    key : string, optional
        The key of the input representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'.
    start : float
        The start value for searching. The default is 0.1. Only works if the clustering method is 'leiden' or 'louvain'.
    end : float
        The end value for searching. The default is 3.0. Only works if the clustering method is 'leiden' or 'louvain'.
    increment : float
        The step size to increase. The default is 0.01. Only works if the clustering method is 'leiden' or 'louvain'.
    use_pca : bool, optional
        Whether use pca for dimension reduction. The default is false.

    Returns
    -------
    None.

    """

    if use_pca:
        adata.obsm[key + '_pca'] = pca(adata, use_reps=key,
                                       n_comps=n_comps if adata.obsm[key].shape[1] > n_comps else adata.obsm[key].shape[
                                           1])
    mclust_R(adata, used_obsm=key + '_pca' if use_pca else key, num_cluster=n_clusters, add_key=add_key)


def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    '''\
    Searching corresponding resolution according to given cluster number

    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float
        The end value for searching.
    increment : float
        The step size to increase.

    Returns
    -------
    res : float
        Resolution.

    '''
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label == 1, "Resolution is not found. Please try bigger range or smaller step!."

    return res


def cosine_similarity(A, B):
    return np.dot(A, B) / (norm(A) * norm(B))


def compute_similarity(z, w=None):
    similarity_spot = DataFrame(np.zeros((z.shape[1], z.shape[1])), columns=z.columns, index=z.columns)
    for i in z.columns:
        zi = z[i]
        for j in z.columns[np.where(z.columns == i)[0][0] + 1:]:
            zj = z[j]
            similarity_spot.loc[i, j] = cosine_similarity(zi.values, zj.values)

    if w is not None:
        similarity_feature = DataFrame(np.zeros((z.shape[1], z.shape[1])), columns=z.columns, index=z.columns)
        for wi in w:
            for i in wi.columns[:-1]:
                for j in wi.columns[np.where(wi.columns == i)[0][0] + 1:]:
                    similarity_feature.loc[i, j] += cosine_similarity(wi[i], wi[
                        j]) / 2 if i in z.columns and j in z.columns else cosine_similarity(wi[i], wi[j])
        return similarity_spot, similarity_feature
    else:
        return similarity_spot


def visualize_latent(z, location):
    data = anndata.AnnData(z)
    data.obsm['spatial'] = location
    data.obsm['emb'] = z
    clustering(data, key='emb', add_key='emb', n_clusters=5)
    sc.pl.embedding(data, color='emb', basis='spatial')


def compute_gene_topic_correlations(adata, z):
    """
    Compute Pearson correlation between genes and learned topics.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing expression data
    z : pandas.DataFrame
        Topic matrix where rows are cells and columns are topics
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing correlation coefficients between genes and topics
        Shape: (n_genes, n_topics)
    """
    # Validate input data
    if not isinstance(z, pd.DataFrame):
        try:
            z = pd.DataFrame(z)
        except:
            raise ValueError("z must be convertible to a pandas DataFrame")

    # Ensure z contains numerical data
    if not np.issubdtype(z.values.dtype, np.number):
        raise ValueError("Topic matrix 'z' must contain numerical values")

    # Get gene expression matrix
    if isinstance(adata.X, np.ndarray):
        gene_expr = adata.X
    else:
        gene_expr = adata.X.toarray()  # Convert sparse matrix to dense if needed

    # Ensure gene expression data is numerical
    if not np.issubdtype(gene_expr.dtype, np.number):
        raise ValueError("Gene expression matrix must contain numerical values")

    # Check for NaN or infinite values
    if np.any(np.isnan(gene_expr)) or np.any(np.isinf(gene_expr)):
        raise ValueError("Gene expression matrix contains NaN or infinite values")

    if np.any(z.isna()) or np.any(np.isinf(z.values)):
        raise ValueError("Topic matrix contains NaN or infinite values")

    # Initialize correlation matrix
    n_genes = gene_expr.shape[1]
    n_topics = z.shape[1]
    correlations = np.zeros((n_genes, n_topics))
    p_values = np.zeros((n_genes, n_topics))

    # Compute correlations for each gene-topic pair
    for i in range(n_genes):
        for j in range(n_topics):
            corr, p_val = pearsonr(gene_expr[:, i], z.iloc[:, j].values)
            correlations[i, j] = corr
            p_values[i, j] = p_val

    # Create DataFrames with gene names and topic names
    correlation_df = pd.DataFrame(
        correlations,
        index=adata.var_names,
        columns=z.columns if z.columns is not None else [f'Topic_{i}' for i in range(n_topics)]
    )

    pvalue_df = pd.DataFrame(
        p_values,
        index=adata.var_names,
        columns=z.columns if z.columns is not None else [f'Topic_{i}' for i in range(n_topics)]
    )

    return correlation_df, pvalue_df

    """
    Get top correlated genes for each topic.
    
    Parameters
    ----------
    correlation_df : pandas.DataFrame
        DataFrame containing correlation coefficients
    n_genes : int
        Number of top genes to return per topic
    absolute : bool
        If True, rank by absolute correlation values
        If False, rank by raw correlation values
    
    Returns
    -------
    dict
        Dictionary mapping topic names to top correlated genes
    """
    top_genes = {}
    for topic in correlation_df.columns:
        if absolute:
            # Get absolute correlation values
            abs_corr = correlation_df[topic].abs()
            # Sort and get top n genes
            top_genes[topic] = correlation_df[topic][abs_corr.nlargest(n_genes).index]
        else:
            # Sort by raw correlation values
            top_genes[topic] = correlation_df[topic].nlargest(n_genes)
    return top_genes


def plot_top_positive_correlations_boxplot(adata, z, omics_name, n_top=None, figsize=(12, 6)):
    """
    Create boxplots for each topic showing only the top n positive correlations.
    
    Parameters
    ----------
    correlation_df : pandas.DataFrame
        DataFrame containing correlation coefficients
    n_top : int
        Number of top positive correlations to include per topic
    figsize : tuple
        Figure size (width, height)
    """
    if n_top is None:
        n_top = 5 if omics_name == 'Proteomics' else 10
    corr_df, _ = compute_gene_topic_correlations(adata, z)
    # Get top positive correlations for each topic
    top_correlations_dict = {}
    for topic in corr_df.columns:
        # Get only positive correlations
        positive_corrs = corr_df[topic][corr_df[topic] > 0]
        # Get top n
        top_correlations_dict[topic] = positive_corrs.nlargest(n_top)

    # Convert to long format for plotting
    plot_data = []
    for topic, corrs in top_correlations_dict.items():
        for gene, corr in corrs.items():
            plot_data.append({
                'Topic': topic,
                'Gene': gene,
                'Correlation': corr
            })
    plot_df = pd.DataFrame(plot_data)

    # Create the plot
    plt.figure(figsize=figsize)

    # Create boxplot
    sns.boxplot(
        data=plot_df,
        x='Topic',
        y='Correlation',
        color='skyblue'
    )

    # Customize the plot
    # plt.title(f'Distribution of Top {n_top} Correlations with {omics_name} per Topic')
    plt.xlabel('')
    plt.ylabel('Pearson Correlation Coefficients')

    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')

    # Add gridlines
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    return plt.gcf()


def plot_topic_correlation_ratio_multimodal(data, omics_names, z, k_values=None, figsize=(10, 8)):
    """
    Plot the log2 fold change of mean top-k correlations between modalities for each topic.
    
    Parameters
    ----------
    data_list : list
        List of AnnData objects, one for each modality
    omics_names : list
        List of strings containing names of each modality
    z : pandas.DataFrame
        Topic matrix where rows are cells and columns are topics
    k_values : dict or None
        Dictionary mapping modality names to their k values for top correlations
        If None, defaults to k=20 for RNA and k=5 for others
    figsize : tuple
        Figure size (width, height)
    """
    # Validate inputs
    if len(data) != len(omics_names):
        raise ValueError("Number of datasets must match number of omics names")
    if len(data) != 2:
        raise ValueError("This function currently supports exactly 2 modalities")

    # Set default k values if not provided
    if k_values is None:
        k_values = {name: 5 if name in ['Proteomics'] else 10 for name in omics_names}

    # Compute correlations for each modality
    modality_corrs = {}
    for data, name in zip(data, omics_names):
        corr_df, _ = compute_gene_topic_correlations(data, z)
        modality_corrs[name] = corr_df

    # Calculate mean top-k correlations for each modality
    topic_means = {name: [] for name in omics_names}

    for topic in z.columns:
        for name in omics_names:
            # Get top k correlations for this modality
            top_k_corrs = np.sort(modality_corrs[name][topic].values)[-k_values[name]:]
            topic_means[name].append(np.mean(top_k_corrs))

    # Calculate log2 fold change
    log2_fold_changes = np.log2(np.array(topic_means[omics_names[0]]) /
                                np.array(topic_means[omics_names[1]]))

    # Create DataFrame with results
    result_df = pd.DataFrame({
        'Topic': z.columns,
        'Log2 Fold Change': log2_fold_changes
    })

    # Sort by log2 fold change
    result_df = result_df.sort_values('Log2 Fold Change', ascending=True)

    # Create horizontal bar plot
    plt.figure(figsize=figsize)
    bars = plt.barh(range(len(result_df)), result_df['Log2 Fold Change'])

    # Color bars based on which modality has stronger correlation
    for i, bar in enumerate(bars):
        if result_df['Log2 Fold Change'].iloc[i] > 0:
            bar.set_color('skyblue')  # First modality stronger
        else:
            bar.set_color('lightgreen')  # Second modality stronger

    # Customize the plot
    plt.title(f'Log2 Fold Change of Top Correlations\n({omics_names[0]} vs {omics_names[1]})')
    plt.xlabel(f'Log2 Fold Change')
    plt.ylabel('Topics')

    # Set topic names as y-axis labels
    plt.yticks(range(len(result_df)), result_df['Topic'])

    # Add grid for better readability
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', label=f'Stronger {omics_names[0]} correlation'),
        Patch(facecolor='lightgreen', label=f'Stronger {omics_names[1]} correlation')
    ]
    plt.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    return plt.gcf()


class RankingSimilarity:
    """
    This class will include some similarity measures between two different
    ranked lists.
    """

    def __init__(
            self,
            S: Union[List, np.ndarray],
            T: Union[List, np.ndarray],
            verbose: bool = False,
    ) -> None:
        """
        Initialize the object with the required lists.
        Examples of lists:
        S = ["a", "b", "c", "d", "e"]
        T = ["b", "a", 1, "d"]

        Both lists reflect the ranking of the items of interest, for example,
        list S tells us that item "a" is ranked first, "b" is ranked second,
        etc.

        Args:
            S, T (list or numpy array): lists with alphanumeric elements. They
                could be of different lengths. Both of the them should be
                ranked, i.e., each element"s position reflects its respective
                ranking in the list. Also we will require that there is no
                duplicate element in each list.
            verbose: If True, print out intermediate results. Default to False.
        """

        assert type(S) in [list, np.ndarray]
        assert type(T) in [list, np.ndarray]

        assert len(S) == len(set(S))
        assert len(T) == len(set(T))

        self.S, self.T = S, T
        self.N_S, self.N_T = len(S), len(T)
        self.verbose = verbose
        self.p = 0.5  # just a place holder

    def assert_p(self, p: float) -> None:
        """Make sure p is between (0, 1), if so, assign it to self.p.

        Args:
            p (float): The value p.
        """
        assert 0.0 < p < 1.0, "p must be between (0, 1)"
        self.p = p

    def _bound_range(self, value: float) -> float:
        """Bounds the value to [0.0, 1.0]."""

        try:
            assert (0 <= value <= 1 or np.isclose(1, value))
            return value

        except AssertionError:
            print("Value out of [0, 1] bound, will bound it.")
            larger_than_zero = max(0.0, value)
            less_than_one = min(1.0, larger_than_zero)
            return less_than_one

    def rbo(
            self,
            k: Optional[float] = None,
            p: float = 1.0,
            ext: bool = False,
    ) -> float:
        """
        This the weighted non-conjoint measures, namely, rank-biased overlap.
        Unlike Kendall tau which is correlation based, this is intersection
        based.
        The implementation if from Eq. (4) or Eq. (7) (for p != 1) from the
        RBO paper: http://www.williamwebber.com/research/papers/wmz10_tois.pdf

        If p = 1, it returns to the un-bounded set-intersection overlap,
        according to Fagin et al.
        https://researcher.watson.ibm.com/researcher/files/us-fagin/topk.pdf

        The fig. 5 in that RBO paper can be used as test case.
        Note there the choice of p is of great importance, since it
        essentially control the "top-weightness". Simply put, to an extreme,
        a small p value will only consider first few items, whereas a larger p
        value will consider more items. See Eq. (21) for quantitative measure.

        Args:
            k: The depth of evaluation.
            p: Weight of each agreement at depth d:
                p**(d-1). When set to 1.0, there is no weight, the rbo returns
                to average overlap.
            ext: If True, we will extrapolate the rbo, as in Eq. (23).

        Returns:
            The rbo at depth k (or extrapolated beyond).
        """

        if not self.N_S and not self.N_T:
            return 1  # both lists are empty

        if not self.N_S or not self.N_T:
            return 0  # one list empty, one non-empty

        if k is None:
            k = float("inf")
        k = min(self.N_S, self.N_T, k)

        # initialize the agreement and average overlap arrays
        A, AO = [0] * k, [0] * k
        if p == 1.0:
            weights = [1.0 for _ in range(k)]
        else:
            self.assert_p(p)
            weights = [1.0 * (1 - p) * p ** d for d in range(k)]

        # using dict for O(1) look up
        S_running, T_running = {self.S[0]: True}, {self.T[0]: True}
        A[0] = 1 if self.S[0] == self.T[0] else 0
        AO[0] = weights[0] if self.S[0] == self.T[0] else 0

        for d in tqdm(range(1, k), disable=~self.verbose):

            tmp = 0
            # if the new item from S is in T already
            if self.S[d] in T_running:
                tmp += 1
            # if the new item from T is in S already
            if self.T[d] in S_running:
                tmp += 1
            # if the new items are the same, which also means the previous
            # two cases did not happen
            if self.S[d] == self.T[d]:
                tmp += 1

            # update the agreement array
            A[d] = 1.0 * ((A[d - 1] * d) + tmp) / (d + 1)

            # update the average overlap array
            if p == 1.0:
                AO[d] = ((AO[d - 1] * d) + A[d]) / (d + 1)
            else:  # weighted average
                AO[d] = AO[d - 1] + weights[d] * A[d]

            # add the new item to the running set (dict)
            S_running[self.S[d]] = True
            T_running[self.T[d]] = True

        if ext and p < 1:
            return self._bound_range(AO[-1] + A[-1] * p ** k)

        return self._bound_range(AO[-1])

    def rbo_ext(self, p=0.98):
        """
        This is the ultimate implementation of the rbo, namely, the
        extrapolated version. The corresponding formula is Eq. (32) in the rbo
        paper.
        """

        self.assert_p(p)

        if not self.N_S and not self.N_T:
            return 1  # both lists are empty

        if not self.N_S or not self.N_T:
            return 0  # one list empty, one non-empty

        # since we are dealing with un-even lists, we need to figure out the
        # long (L) and short (S) list first. The name S might be confusing
        # but in this function, S refers to short list, L refers to long list
        if len(self.S) > len(self.T):
            L, S = self.S, self.T
        else:
            S, L = self.S, self.T

        s, l = len(S), len(L)  # noqa

        # initialize the overlap and rbo arrays
        # the agreement can be simply calculated from the overlap
        X, A, rbo = [0] * l, [0] * l, [0] * l

        # first item
        S_running, L_running = {S[0]}, {L[0]}  # for O(1) look up
        X[0] = 1 if S[0] == L[0] else 0
        A[0] = X[0]
        rbo[0] = 1.0 * (1 - p) * A[0]

        # start the calculation
        disjoint = 0
        ext_term = A[0] * p

        for d in tqdm(range(1, l), disable=~self.verbose):
            if d < s:  # still overlapping in length

                S_running.add(S[d])
                L_running.add(L[d])

                # again I will revoke the DP-like step
                overlap_incr = 0  # overlap increment at step d

                # if the new items are the same
                if S[d] == L[d]:
                    overlap_incr += 1
                else:
                    # if the new item from S is in L already
                    if S[d] in L_running:
                        overlap_incr += 1
                    # if the new item from L is in S already
                    if L[d] in S_running:
                        overlap_incr += 1

                X[d] = X[d - 1] + overlap_incr
                # Eq. (28) that handles the tie. len() is O(1)
                A[d] = 2.0 * X[d] / (len(S_running) + len(L_running))
                rbo[d] = rbo[d - 1] + 1.0 * (1 - p) * (p ** d) * A[d]

                ext_term = 1.0 * A[d] * p ** (d + 1)  # the extrapolate term

            else:  # the short list has fallen off the cliff
                L_running.add(L[d])  # we still have the long list

                # now there is one case
                overlap_incr = 1 if L[d] in S_running else 0

                X[d] = X[d - 1] + overlap_incr
                A[d] = 1.0 * X[d] / (d + 1)
                rbo[d] = rbo[d - 1] + 1.0 * (1 - p) * (p ** d) * A[d]

                X_s = X[s - 1]  # this the last common overlap
                # second term in first parenthesis of Eq. (32)
                disjoint += 1.0 * (1 - p) * (p ** d) * \
                            (X_s * (d + 1 - s) / (d + 1) / s)
                ext_term = 1.0 * ((X[d] - X_s) / (d + 1) + X[s - 1] / s) * \
                           p ** (d + 1)  # last term in Eq. (32)

        return self._bound_range(rbo[-1] + disjoint + ext_term)

    def top_weightness(
            self,
            p: Optional[float] = None,
            d: Optional[int] = None):
        """
        This function will evaluate the degree of the top-weightness of the
        rbo. It is the implementation of Eq. (21) of the rbo paper.

        As a sanity check (per the rbo paper),
        top_weightness(p=0.9, d=10) should be 86%
        top_weightness(p=0.98, d=50) should be 86% too

        Args:
            p (float), default None: A value between zero and one.
            d (int), default None: Evaluation depth of the list.

        Returns:
            A float between [0, 1], that indicates the top-weightness.
        """

        # sanity check
        self.assert_p(p)

        if d is None:
            d = min(self.N_S, self.N_T)
        else:
            d = min(self.N_S, self.N_T, int(d))

        if d == 0:
            top_w = 1
        elif d == 1:
            top_w = 1 - 1 + 1.0 * (1 - p) / p * (np.log(1.0 / (1 - p)))
        else:
            sum_1 = 0
            for i in range(1, d):
                sum_1 += 1.0 * p ** (i) / i
            top_w = 1 - p ** (i) + 1.0 * (1 - p) / p * (i + 1) * \
                    (np.log(1.0 / (1 - p)) - sum_1)  # here i == d-1

        if self.verbose:
            print("The first {} ranks have {:6.3%} of the weight of "
                  "the evaluation.".format(d, top_w))

        return self._bound_range(top_w)


def split_numbers(max_number, chunk_size):
    lists = []
    for start in range(0, max_number, chunk_size):
        end = min(start + chunk_size, max_number)
        lists.append(list(range(start, end)))
    return lists


def pairwise_distances(x):
    # x should be two dimensional
    instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()


def GaussianKernelMatrix(x, sigma=1):
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ / sigma)
