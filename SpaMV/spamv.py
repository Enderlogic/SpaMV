"""Main module."""
import math
from typing import List
import anndata
import numpy
import numpy as np
import pyro
import torch
import torch.nn.functional as F
import wandb
from anndata import AnnData
from pandas import DataFrame
from pyro.infer import TraceMeanField_ELBO
from pyro.infer.enum import get_importance_trace
from pyro.poutine import scale, trace
from scipy.sparse import issparse
from scipy.spatial.distance import cosine, cdist
from sklearn.metrics import adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, homogeneity_score, v_measure_score
from sklearn.preprocessing import MinMaxScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torchinfo import summary
from tqdm import tqdm
from .metrics import compute_moranI, compute_jaccard
from .model import spamv
from .utils import adjacent_matrix_preprocessing, get_init_bg, plot_embedding_results, log_mean_exp, cosine_similarity, \
    clustering, compute_similarity


class SpaMV:
    def __init__(self, adatas: List[AnnData], zp_dims: List[int] = None, zs_dim: int = 5, weights: List[float] = None,
                 recon_types: List[str] = None, omics_names: List[str] = None, device: torch.device = None,
                 hidden_size: int = 128, heads: int = 1, n_neighbors: int = 20, interpretable: bool = True,
                 verbose: bool = False, random_seed: int = 1214, min_epochs: int = 100, max_epochs: int = 1000,
                 min_kl: float = 1, max_kl: float = 1, learning_rate: float = 1e-3, folder_path: str = None,
                 early_stopping: bool = True, patience: int = 20, n_cluster: int = 10, test_mode: bool = False,
                 result: DataFrame = None):
        pyro.clear_param_store()
        pyro.set_rng_seed(random_seed)
        torch.manual_seed(random_seed)
        self.n_omics = len(adatas)
        if zs_dim is None:
            if interpretable:
                zs_dim = 5
            else:
                zs_dim = 32
        else:
            if zs_dim <= 0:
                raise ValueError("zs_dim must be positive")
        self.zs_dim = zs_dim
        if zp_dims is None:
            if interpretable:
                zp_dims = [5 for _ in range(self.n_omics)]
            else:
                zp_dims = [32 for _ in range(self.n_omics)]
        else:
            if min(zp_dims) < 0:
                raise ValueError("all elements in zp_dims must be non-negative")

        self.zp_dims = zp_dims
        if weights is None:
            weights = [1 for _ in range(self.n_omics)]
        else:
            if min(weights) < 0:
                raise ValueError("all elements in weights must be non-negative")
        self.weights = weights
        if recon_types is None:
            recon_types = ["nb" for _ in range(self.n_omics)]
        else:
            for recon_type in recon_types:
                if recon_type != "nb" and recon_type != "zinb":
                    raise ValueError("recon_type must be 'nb' or 'zinb'")
        self.omics_names = ["omics_{}".format(i) for i in range(self.n_omics)] if omics_names is None else omics_names
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") if device is None else device
        self.hidden_size = hidden_size
        self.heads = heads
        self.n_neighbors = n_neighbors

        self.adatas = adatas
        self.n_obs = adatas[0].shape[0]
        self.data_dims = [data.shape[1] for data in adatas]
        self.interpretable = interpretable
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.min_kl = min_kl
        self.max_kl = max_kl
        self.learning_rate = learning_rate
        self.folder_path = folder_path
        self.early_stopping = early_stopping
        self.patience = patience
        self.n_cluster = n_cluster
        self.test_mode = test_mode
        self.result = result

        self.x = [torch.tensor(data.X.toarray() if issparse(data.X) else data.X, device=self.device, dtype=torch.float)
                  for data in adatas]
        self.edge_index = adjacent_matrix_preprocessing(adatas, n_neighbors, self.device)

        self.init_bg_means = get_init_bg(self.x) if interpretable else None
        self.model = spamv(self.data_dims, zs_dim, zp_dims, self.init_bg_means, weights, hidden_size, recon_types,
                           heads, interpretable, self.device)
        if verbose:
            print(summary(self.model))

    def train(self):
        self.model = self.model.to(self.device)
        if self.early_stopping:
            self.early_stopper = EarlyStopper(patience=self.patience)

        pbar = tqdm(range(self.max_epochs), position=0, leave=True)
        loss_fn = lambda model, guide: TraceMeanField_ELBO(num_particles=1).differentiable_loss(
            scale(model, 1 / self.n_obs), scale(guide, 1 / self.n_obs), self.x, self.edge_index)
        with trace(param_only=True) as param_capture:
            loss = loss_fn(self.model.model, self.model.guide)
        params = set(site["value"].unconstrained() for site in param_capture.trace.nodes.values())
        optimizer = Adam(params, lr=self.learning_rate, betas=(0.9, 0.999), weight_decay=0)
        for epoch in pbar:
            self.model.train()
            optimizer.zero_grad()
            loss = self.get_elbo(epoch)
            loss.backward()
            clip_grad_norm_(params, 5)
            optimizer.step()
            pbar.set_description(f"Epoch Loss:{loss:.3f}")

            if self.early_stopping:
                if epoch > self.min_epochs:
                    if self.early_stopper.early_stop(loss):
                        print("Early Stopping")
                        break
            if (epoch + 1) % 50 == 0 and self.test_mode:
                if self.interpretable:
                    z, w = self.get_embedding_and_feature_by_topic(map=True)
                    self.adatas[0].obs['spamv'] = DataFrame(MinMaxScaler().fit_transform(z), columns=z.columns,
                                                    index=z.index).idxmax(1)
                    plot_embedding_results(self.adatas, self.omics_names, z, w, folder_path=self.folder_path,
                                           file_name='spamv_' + str(epoch + 1) + '.pdf')
                else:
                    z = self.get_embedding()
                    self.adatas[0].obsm['spamv'] = z
                    self.adatas[0].obsm['zs+zp1'] = z[:, :self.zs_dim + self.zp_dims[0]]
                    self.adatas[1].obsm['zs+zp2'] = numpy.concatenate((z[:, :self.zs_dim], z[:, -self.zp_dims[1]:]),
                                                                      axis=1)
                    jaccard1 = compute_jaccard(self.adatas[0], 'zs+zp1', 'X_pca')
                    jaccard2 = compute_jaccard(self.adatas[1], 'zs+zp2', 'X_pca')
                    wandb.log({"jaccard1": jaccard1}, step=epoch)
                    wandb.log({"jaccard2": jaccard2}, step=epoch)
                    print("jaccard 1: ", str(jaccard1), "jaccard 2:", str(jaccard2))
                    clustering(self.adatas[0], key='spamv', add_key='spamv', n_clusters=self.n_cluster, method='mclust',
                               use_pca=True)
                if 'cluster' in self.adatas[0].obs:
                    cluster = self.adatas[0].obs['cluster']
                    cluster_learned = self.adatas[0].obs['spamv']
                    ari = adjusted_rand_score(cluster, cluster_learned)
                    mi = mutual_info_score(cluster, cluster_learned)
                    nmi = normalized_mutual_info_score(cluster, cluster_learned)
                    ami = adjusted_mutual_info_score(cluster, cluster_learned)
                    hom = homogeneity_score(cluster, cluster_learned)
                    vme = v_measure_score(cluster, cluster_learned)
                    wandb.log({"ari": ari}, step=epoch)
                    wandb.log({"mi": mi}, step=epoch)
                    wandb.log({"nmi": nmi}, step=epoch)
                    wandb.log({"ami": ami}, step=epoch)
                    wandb.log({"hom": hom}, step=epoch)
                    wandb.log({"vme": vme}, step=epoch)
                    wandb.log({'ave': (ari + mi + nmi + ami + hom + vme) / 6}, step=epoch)
                    print("ari: ", str(ari), "\naverage: ", str((ari + mi + nmi + ami + hom + vme) / 6))
                    if self.result is not None:
                        self.result.loc[len(self.result)] = [self.zp_dims[0], self.zs_dim, self.hidden_size, self.heads,
                                                             self.n_neighbors, self.max_kl, self.learning_rate,
                                                             self.weights[0] == self.weights[1], epoch, "ari", ari]
                        self.result.loc[len(self.result)] = [self.zp_dims[0], self.zs_dim, self.hidden_size, self.heads,
                                                             self.n_neighbors, self.max_kl, self.learning_rate,
                                                             self.weights[0] == self.weights[1], epoch, "average",
                                                             (ari + mi + nmi + ami + hom + vme) / 6]
                moranI = compute_moranI(self.adatas[0], 'spamv')
                wandb.log({"spamv moran I": moranI}, step=epoch)
                print('moran I', moranI)
        return self.result

    def _kl_weight(self, iteration):
        kl = self.min_kl + iteration / self.max_epochs * (self.max_kl - self.min_kl)
        if kl > self.max_kl:
            kl = self.max_kl
        return kl

    def get_elbo(self, epoch):
        self.model = self.model.to(self.device)
        annealing_factor = self._kl_weight(epoch)
        elbo_particle = 0
        model_trace, guide_trace = get_importance_trace('flat', torch.inf, scale(self.model.model, 1 / self.n_obs),
                                                        scale(self.model.guide, 1 / self.n_obs),
                                                        (self.x, self.edge_index), {}, detach=False)
        for name, model_site in model_trace.nodes.items():
            if model_site["type"] == "sample":
                if model_site["is_observed"]:
                    elbo_particle = elbo_particle + model_site["log_prob_sum"]
                    wandb.log({name: -model_site["log_prob_sum"].item()}, step=epoch)
                else:
                    guide_site = guide_trace.nodes[name]
                    entropy_term = (log_mean_exp(torch.stack(
                        [guide_trace.nodes["zs_{}".format(i)]["fn"].log_prob(guide_site["value"]) for i in
                         range(len(self.data_dims))])) * guide_site['scale']).sum() if "zs" in name else guide_site[
                        'log_prob_sum']
                    elbo_particle += (model_site["log_prob_sum"] - entropy_term) * annealing_factor
                    wandb.log({name: (-model_site["log_prob_sum"] + entropy_term.sum()).item()}, step=epoch)
        return -elbo_particle

    def save(self, path):
        self.model.save(path)

    def load(self, path, map_location=torch.device('cpu')):
        self.model.load(path, map_location=map_location)

    def get_embedding(self, map=False):
        '''
        This function is used to get the embeddings. The returned embedding is stored in a pandas dataframe object if
        the model is in interpretable mode. Shared embeddings will be present in the first zs_dim columns, and private
        embeddings will be present in the following columns given their input orders.

        For example, if the input data is [data1, data2] and the shared latent dimension and both private latent
        dimensions are all 5, (i.e., zs_dim=5, zp_dim[0]=5, zp_dim[1]=5). Then the first 5 columns in returned dataframe
        will be the shared embeddings, and the following 5 columns will be the private embeddings for data1, and the
        last 5 columns will be the private embeddings for data2.
        '''
        z_mean = self.model.get_embedding(self.x, self.edge_index)
        if self.interpretable:
            columns_name = ["Shared topic {}".format(i + 1) for i in range(self.zs_dim)]
            for i in range(self.n_omics):
                columns_name += [self.omics_names[i] + ' private topic {}'.format(j + 1) for j in
                                 range(self.zp_dims[i])]
            spot_topic = DataFrame(z_mean.detach().cpu().numpy(), columns=columns_name)
            spot_topic.set_index(self.adatas[0].obs_names, inplace=True)
            return spot_topic
        else:
            return F.normalize(z_mean, p=2, eps=1e-12, dim=1).detach().cpu().numpy()
            # return z_mean

    def get_embedding_and_feature_by_topic(self, map=False):
        '''
        This function is used to get the feature by topic. The returned list contains feature by topic for each modality
        according to their input order. The row names in the returned dataframes are the feature names in the
        corresponding modality, and the column names are the topic names.

        For example, if the input data is [data1, data2] and the shared latent dimension and both private latent are all
        5. Assume, data1 is RNA modality and data2 is Protein modality. Then feature_topics[0] would be the feature by
        topic matrix for RNA, and each row represents a gene and each column represents a topic. The topic names are
        defined in the same way as the get_embedding() function. That is, Topics 1-5 are shared topics, Topics 6-10 are
        private topics for modality 1 (RNA), and Topics 11-15 are private topics for modality 2 (Protein).
        '''
        if self.interpretable:
            z_mean = self.model.get_embedding(self.x, self.edge_index)
            columns_name = ["Shared topic {}".format(i + 1) for i in range(self.zs_dim)]
            for i in range(self.n_omics):
                columns_name += [self.omics_names[i] + ' private topic {}'.format(j + 1) for j in
                                 range(self.zp_dims[i])]
            spot_topic = DataFrame(z_mean.detach().cpu().numpy(), columns=columns_name)
            spot_topic.set_index(self.adatas[0].obs_names, inplace=True)
            feature_topics = self.model.get_feature_by_topic()
            for i in range(len(self.zp_dims)):
                feature_topics[i] = DataFrame(feature_topics[i],
                                              columns=["Shared topic {}".format(j + 1) for j in range(self.zs_dim)] + [
                                                  self.omics_names[i] + " private topic {}".format(j + 1) for j in
                                                  range(self.zp_dims[i])], index=self.adatas[i].var_names)
            if map:
                spot_topic, feature_topics = self.merge_and_prune(spot_topic, feature_topics)
            return spot_topic, feature_topics
        else:
            raise Exception("This function can only be used with interpretable mode.")

    def merge_and_prune(self, topic_spot, feature_topics, threshold=0.7):
        topic_spot_update = topic_spot.copy()
        feature_topics_update = feature_topics.copy()
        while True:
            similarity_spot, similarity_feature = compute_similarity(topic_spot_update, feature_topics_update)
            if similarity_spot.stack().max() > threshold:
                topic1, topic2 = similarity_spot.stack().idxmax()
                print('The pattern between', topic1, 'and', topic2,
                      'is similar, with a cosine similarity of {:.3f}. Therefore, we decided to prune'.format(
                          similarity_spot.stack().max()), topic2 + '.')
            elif similarity_feature.stack().max() > threshold:
                topic1, topic2 = similarity_feature.stack().idxmax()
                print('The weight vectors between', topic1, 'and', topic2,
                      'are similar, with an average cosine similarity of {:.3f}. Therefore we decided to prune'.format(
                          similarity_feature.stack().max()), topic2 + '.')
            else:
                break
            if topic1.split(' ')[0] == topic2.split(' ')[0]:
                topic_spot_update[topic1] += topic_spot_update[topic2]
                if 'Shared' in topic2:
                    for i in range(self.n_omics):
                        feature_topics_update[i][topic1] += feature_topics_update[i][topic2]
                else:
                    for i in range(self.n_omics):
                        if topic2 in feature_topics_update[i].columns:
                            feature_topics_update[i][topic1] += feature_topics_update[i][topic2]
            topic_spot_update = topic_spot_update.drop(topic2, axis=1)
            if 'Shared' in topic2:
                for i in range(self.n_omics):
                    feature_topics_update[i] = feature_topics_update[i].drop(topic2, axis=1)
            else:
                for i in range(self.n_omics):
                    if topic2 in feature_topics_update[i].columns:
                        feature_topics_update[i] = feature_topics_update[i].drop(topic2, axis=1)
            continue
        return topic_spot_update, feature_topics_update


class EarlyStopper:
    def __init__(self, patience=50, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_training_loss = np.inf

    def early_stop(self, training_loss):
        if training_loss < self.min_training_loss:
            self.min_training_loss = training_loss
            self.counter = 0
        elif training_loss > (self.min_training_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def reset(self):
        self.counter = 0
        self.min_training_loss = np.Inf
