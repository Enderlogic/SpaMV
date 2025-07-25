"""Main module."""
from typing import List
import numpy as np
import pyro
import scanpy
import scanpy.plotting
import torch
import torch.nn.functional as F
from anndata import AnnData
from pandas import DataFrame
from pyro.infer import TraceMeanField_ELBO
from pyro.poutine import scale, trace
from pyro import poutine
from pyro.infer.util import is_validation_enabled
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_model_guide_match, check_site_shape
from scipy.sparse import issparse
from sklearn.cluster import KMeans
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from tqdm import tqdm
from torch_geometric import seed_everything
import os
from .model import spamv
from .layers import Measurement
from .utils import adjacent_matrix_preprocessing, get_init_bg, log_mean_exp, RankingSimilarity, split_numbers, \
    GaussianKernelMatrix

from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader, DataLoader


def set_seed(seed):
    """Set seed for all random number generators and ensure deterministic operations."""

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    # Pyro
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    seed_everything(seed)

    if torch.cuda.is_available():
        # CUDA
        torch.cuda.manual_seed_all(seed)

        # Ensure deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # Force deterministic algorithms
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)

    # Set environment variables for deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_importance_trace(graph_type, max_plate_nesting, model, guide, args, detach=False):
    """
    Returns a single trace from the guide, which can optionally be detached,
    and the model that is run against it.
    """
    # Dispatch between callables vs GuideMessengers.
    unwrapped_guide = poutine.unwrap(guide)
    if isinstance(unwrapped_guide, poutine.messenger.Messenger):
        if detach:
            raise NotImplementedError("GuideMessenger does not support detach")
        guide(args)
        model_trace, guide_trace = unwrapped_guide.get_traces()
    else:
        guide_trace = poutine.trace(guide, graph_type=graph_type).get_trace(args)
        if detach:
            guide_trace.detach_()
        model_trace = poutine.trace(
            poutine.replay(model, trace=guide_trace), graph_type=graph_type
        ).get_trace(args)

    if is_validation_enabled():
        check_model_guide_match(model_trace, guide_trace, max_plate_nesting)

    guide_trace = prune_subsample_sites(guide_trace)
    model_trace = prune_subsample_sites(model_trace)

    model_trace.compute_log_prob()
    guide_trace.compute_score_parts()
    if is_validation_enabled():
        for site in model_trace.nodes.values():
            if site["type"] == "sample":
                check_site_shape(site, max_plate_nesting)
        for site in guide_trace.nodes.values():
            if site["type"] == "sample":
                check_site_shape(site, max_plate_nesting)

    return model_trace, guide_trace


def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / e_x.sum()


class SpaMV:
    def __init__(self, adatas: List[AnnData], interpretable: bool, zp_dims: List[int] = None, zs_dim: int = None,
                 weights: List[float] = None, alphas: List[float] = None, recon_types: List[str] = None,
                 omics_names: List[str] = None, device: torch.device = None, hidden_dim: int = None,
                 batch_size: int = None, heads: int = 1, neighborhood_depth: int = 2, neighborhood_embedding: int = 10,
                 random_seed: int = 0, max_epochs_stage1: int = 400, max_epochs_stage2: int = 400,
                 learning_rate: float = None, early_stopping: bool = True, patience: int = 200,
                 threshold_noise: float = .3, threshold_background: float = 1):
        pyro.clear_param_store()
        set_seed(random_seed)

        # Store the random seed for use in training
        self.random_seed = random_seed

        self.n_omics = len(adatas)
        self.data_dims = [data.shape[1] for data in adatas]
        if zs_dim is None:
            self.zs_dim = 10 if interpretable else 32
        elif zs_dim <= 0:
            raise ValueError("zs_dim must be a positive integer")
        else:
            self.zs_dim = zs_dim
        if zp_dims is None:
            self.zp_dims = [10 if interpretable else 32 for _ in range(self.n_omics)]
        elif min(zp_dims) < 0:
            raise ValueError("all elements in zp_dims must be non-negative integers")
        else:
            self.zp_dims = zp_dims
        if weights is None:
            self.weights = [max(self.data_dims) / self.data_dims[i] for i in range(self.n_omics)]
        elif min(weights) < 0:
            raise ValueError("all elements in weights must be non-negative")
        else:
            self.weights = weights
        if alphas is None:
            self.alphas = [10 if interpretable else 1 for _ in range(self.n_omics)]
        elif min(alphas) < 0:
            raise ValueError("all elements in alphas must be non-negative")
        else:
            self.alphas = alphas
        if recon_types is None:
            recon_types = ["nb" if interpretable else "gauss" for _ in range(self.n_omics)]
        else:
            for recon_type in recon_types:
                if recon_type not in ['zinb', 'nb', 'gauss']:
                    raise ValueError("recon_type must be 'nb' or 'zinb' or 'gauss'")

        self.recon_types = recon_types
        if hidden_dim is None:
            self.hidden_dim = 128 if interpretable else 256
        elif hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        else:
            self.hidden_dim = hidden_dim
        self.omics_names = ["Omics_{}".format(i + 1) for i in
                            range(self.n_omics)] if omics_names is None else omics_names
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if learning_rate is None:
            self.learning_rate = 1e-2 if interpretable else 1e-3
        elif learning_rate <= 0:
            raise ValueError("learning_rate must be a positive number")
        else:
            self.learning_rate = learning_rate
        self.adatas = adatas
        self.n_obs = adatas[0].shape[0]
        if batch_size is None:
            self.batch_size = 10000 if self.n_obs > 10000 else self.n_obs
        elif batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        else:
            self.batch_size = batch_size
        self.neighborhood_depth = neighborhood_depth
        self.neighborhood_embedding = neighborhood_embedding

        self.interpretable = interpretable
        self.max_epochs_stage1 = max_epochs_stage1
        self.max_epochs_stage2 = max_epochs_stage2
        self.early_stopping = early_stopping
        self.patience = patience
        self.pretrain_epoch = 200 if interpretable else 10
        self.epoch_stage1 = 0
        self.epoch_stage2 = 0
        self.threshold_noise = threshold_noise
        self.threshold_background = threshold_background
        self.meaningful_dimensions = {}

        print("Building data and neighboring graphs...")
        self.data = [Data(
            x=torch.tensor(np.ascontiguousarray(adatas[i].X.toarray() if issparse(adatas[i].X) else adatas[i].X),
                           device=self.device, dtype=torch.float),
            edge_index=adjacent_matrix_preprocessing(adatas[i], neighborhood_depth, neighborhood_embedding,
                                                     self.device)) for i in range(self.n_omics)]
        self.root_node_indices = split_numbers(self.n_obs, self.batch_size)
        self.init_bg_means = get_init_bg([torch.tensor(
            np.ascontiguousarray(adatas[i].X.toarray() if issparse(adatas[i].X) else adatas[i].X), device=self.device,
            dtype=torch.float) for i in range(self.n_omics)]) if interpretable else None
        self.model = spamv(self.data_dims, self.zs_dim, self.zp_dims, self.init_bg_means, self.weights, self.hidden_dim,
                           self.recon_types, heads, interpretable, self.device, self.omics_names)

    def get_batch_data(self, batch_idx):
        if self.n_obs == self.batch_size:
            return self.data
        else:
            return [NeighborLoader(d, num_neighbors=[10, 5] if self.interpretable else [20], batch_size=self.batch_size,
                                   input_nodes=self.root_node_indices[batch_idx], shuffle=False).data for d in
                    self.data]

    def train(self):
        self.model = self.model.to(self.device)
        if self.early_stopping:
            self.early_stopper = EarlyStopper(patience=self.patience)

        pbar = tqdm(range(self.max_epochs_stage1), position=0, leave=True)

        loss_fn = lambda model, guide: TraceMeanField_ELBO(num_particles=1).differentiable_loss(model, guide,
                                                                                                self.get_batch_data(0))

        with trace(param_only=True) as param_capture:
            loss = loss_fn(self.model.model, self.model.guide)

        params = set(site["value"].unconstrained() for site in param_capture.trace.nodes.values())
        optimizer = Adam(params, lr=self.learning_rate, betas=(.9, .999), weight_decay=0, eps=1e-8)

        for self.epoch_stage1 in pbar:
            for batch_idx in range(len(self.root_node_indices)):
                batch_data = self.get_batch_data(batch_idx)
                if self.epoch_stage1 == self.pretrain_epoch:
                    self.early_stopper.min_training_loss = np.inf
                if self.epoch_stage1 >= self.pretrain_epoch:
                    if self.epoch_stage1 % 100 == 0 or self.epoch_stage1 == self.pretrain_epoch:
                        n_epochs = 100
                        self.measurement = Measurement(self.zp_dims, self.hidden_dim, self.data_dims, self.recon_types,
                                                       self.omics_names, self.interpretable).to(self.device)
                        optimizer_measurement = Adam(self.measurement.parameters(),
                                                     lr=1e-2 if self.interpretable else 1e-3,
                                                     betas=(.9, .999), weight_decay=0)
                    else:
                        n_epochs = 1 if self.interpretable else 10
                    self.zp = [z.detach() for z in self.model.get_private_embedding(batch_data)]
                    for epoch_measurement in range(n_epochs):
                        self.measurement.train()
                        optimizer_measurement.zero_grad()
                        measurement_loss = self.get_measurement_loss(batch_data)
                        measurement_loss.backward()
                        clip_grad_norm_(self.measurement.parameters(), 5)
                        optimizer_measurement.step()
                # train the model
                self.model.train()
                optimizer.zero_grad()
                loss = self.get_elbo(batch_data)
                loss.backward()
                clip_grad_norm_(params, 5)
                optimizer.step()
                pbar.set_description(f"Epoch Loss:{loss:.3f}")

            if self.early_stopping:
                if self.early_stopper.early_stop(loss):
                    print("Early Stopping")
                    break
        if self.interpretable:
            params_zs = set(site["value"].unconstrained() for site in param_capture.trace.nodes.values() if
                            'zp' not in site['name'])
            optimizer_zs = Adam(params_zs, lr=self.learning_rate, betas=(.9, .999), weight_decay=0, eps=1e-8)
            pbar = tqdm(range(self.epoch_stage1, self.max_epochs_stage2 + self.epoch_stage1), position=0, leave=True)
            self.early_stopper.min_training_loss = np.inf
            zps = self.model.get_private_latent(self.data, False)

            scanpy.pp.neighbors(self.adatas[0], use_rep='spatial')
            for i in range(self.n_omics):
                self.meaningful_dimensions['zp_' + self.omics_names[i]] = self.get_meaningful_dimensions(zps[i])
            for self.epoch_stage2 in pbar:
                for batch_idx in range(len(self.root_node_indices)):
                    batch_data = self.get_batch_data(batch_idx)
                    # train shared model
                    self.model.train()
                    optimizer_zs.zero_grad()
                    loss = self.get_elbo_shared(batch_data)
                    loss.backward()
                    clip_grad_norm_(params_zs, 5)
                    optimizer_zs.step()
                    pbar.set_description(f"Epoch Loss:{loss:.3f}")

                if self.early_stopping:
                    if self.early_stopper.early_stop(loss):
                        print("Early Stopping")
                        break
            zs = self.model.get_shared_embedding(self.data)
            self.meaningful_dimensions['zs'] = self.get_meaningful_dimensions(zs)

    def get_meaningful_dimensions(self, z):
        # prune noisy dimensions
        self.adatas[0].obsm['z'] = z.detach().cpu().numpy()
        morans_i = scanpy.metrics.morans_i(self.adatas[0], obsm='z')
        if morans_i.min() < self.threshold_noise:
            kmeans = KMeans(n_clusters=2)
            kmeans.fit(morans_i.reshape(-1, 1))
            z_pruned = np.where(kmeans.labels_ == np.argmax(kmeans.cluster_centers_))[0]
        else:
            z_pruned = np.array(range(z.shape[1]))

        # prune background dimensions
        z_exp_std = torch.exp(z[:, z_pruned]).std(0)
        z_pruned = z_pruned[z_exp_std.detach().cpu().numpy() > self.threshold_background]

        return z_pruned

    def HSIC(self, x, y, s_x=1, s_y=1):
        m, _ = x.shape  # batch size
        K = GaussianKernelMatrix(x, s_x)
        L = GaussianKernelMatrix(y, s_y)
        H = torch.eye(m, device=self.device) - 1.0 / m * torch.ones((m, m), device=self.device)
        HSIC = torch.trace(torch.mm(L, torch.mm(H, torch.mm(K, H)))) / ((m - 1) ** 2)
        return HSIC

    def get_elbo(self, datas):
        elbo_particle = 0
        batch_size = datas[0].num_nodes
        model_trace, guide_trace = get_importance_trace('flat', torch.inf, scale(self.model.model, 1 / batch_size),
                                                        scale(self.model.guide, 1 / batch_size), datas, detach=False)
        for name, model_site in model_trace.nodes.items():
            if model_site["type"] == "sample":
                if model_site["is_observed"]:
                    elbo_particle = elbo_particle + model_site["log_prob_sum"]
                else:
                    guide_site = guide_trace.nodes[name]
                    entropy_term = (log_mean_exp(torch.stack(
                        [guide_trace.nodes["zs_" + self.omics_names[i]]["fn"].log_prob(guide_site["value"]) for i in
                         range(len(self.data_dims))])) * guide_site['scale']).sum() if "zs" in name else guide_site[
                        'log_prob_sum']
                    elbo_particle += (model_site["log_prob_sum"] - entropy_term)
        if self.epoch_stage1 >= self.pretrain_epoch:
            self.measurement.eval()
            output = self.measurement(self.model.get_private_latent(datas, True))
            for i in range(self.n_omics):
                for j in range(self.n_omics):
                    if i != j:
                        name = "from_" + self.omics_names[i] + "_to_" + self.omics_names[j]
                        if self.interpretable:
                            loss_measurement = output[name].std(0).sum() * self.data_dims[i] / 100 * self.weights[i] * \
                                               self.alphas[i]
                        else:
                            data_std = datas[j].x[:datas[0].num_nodes].std(0)
                            loss_measurement = (output[name].std(
                                0) * data_std / data_std.sum()).sum() * data_std.mean() * self.weights[i] * self.alphas[
                                                   i]
                        elbo_particle -= loss_measurement
        return -elbo_particle

    def get_elbo_shared(self, datas):
        elbo_particle = 0
        batch_size = datas[0].num_nodes
        model_trace, guide_trace = get_importance_trace('flat', torch.inf, scale(self.model.model, 1 / batch_size),
                                                        scale(self.model.guide, 1 / batch_size), datas, detach=False)
        for name, model_site in model_trace.nodes.items():
            if model_site["type"] == "sample":
                if model_site["is_observed"]:
                    elbo_particle = elbo_particle + model_site["log_prob_sum"]
                elif 'zs' in name:
                    guide_site = guide_trace.nodes[name]
                    entropy_term = (log_mean_exp(torch.stack(
                        [guide_trace.nodes["zs_" + self.omics_names[i]]["fn"].log_prob(guide_site["value"]) for i in
                         range(len(self.data_dims))])) * guide_site['scale']).sum()
                    elbo_particle += (model_site["log_prob_sum"] - entropy_term)
                    omics_name = name[3:]
                    for on in self.omics_names:
                        if on != omics_name:
                            loss_hsic = self.HSIC(guide_site['fn'].mean,
                                                  guide_trace.nodes["zp_" + on]["fn"].mean.detach()[:,
                                                  self.meaningful_dimensions['zp_' + on]]) * batch_size * np.sqrt(
                                self.data_dims[self.omics_names.index(on)]) * self.alphas[self.omics_names.index(on)]
                            elbo_particle -= loss_hsic
        return -elbo_particle

    def get_measurement_loss(self, datas):
        batch_size = datas[0].num_nodes
        zps = [z.detach() if self.interpretable else z.detach() for z in
               self.model.get_private_latent(datas, False)]
        output = self.measurement(zps)
        loss = 0
        for i in range(self.n_omics):
            for j in range(self.n_omics):
                if i != j:
                    name = "from_" + self.omics_names[i] + "_to_" + self.omics_names[j]
                    if self.interpretable:
                        output[name] = datas[j].x[:batch_size].sum(1, keepdim=True) * output[name]
                        loss += mse_loss(output[name], datas[j].x[:batch_size]) * batch_size
                    else:
                        loss += mse_loss(output[name], datas[j].x[:batch_size]) * np.sqrt(batch_size)
        return loss

    def save(self, path):
        self.model.save(path)

    def load(self, path, map_location=torch.device('cpu')):
        self.model.load(path, map_location=map_location)

    def get_separate_embedding(self):
        return self.model.get_separate_embedding(self.x, self.edge_index)

    def get_embedding(self, use_softmax=True):
        '''
        This function is used to get the embeddings. The returned embedding is stored in a pandas dataframe object if
        the model is in interpretable mode. Shared embeddings will be present in the first zs_dim columns, and private
        embeddings will be present in the following columns given their input orders.

        For example, if the input data is [data1, data2] and the shared latent dimension and both private latent
        dimensions are all 5, (i.e., zs_dim=5, zp_dim[0]=5, zp_dim[1]=5). Then the first 5 columns in returned dataframe
        will be the shared embeddings, and the following 5 columns will be the private embeddings for data1, and the
        last 5 columns will be the private embeddings for data2.
        '''
        z_mean = self.model.get_embedding(self.data)
        if self.interpretable:
            columns_name = ["Shared topic {}".format(i + 1) for i in range(self.zs_dim)]
            for i in range(self.n_omics):
                columns_name += [self.omics_names[i] + ' private topic {}'.format(j + 1) for j in
                                 range(self.zp_dims[i])]
            spot_topic = DataFrame(z_mean.detach().cpu().numpy(), columns=columns_name)
            spot_topic.set_index(self.adatas[0].obs_names, inplace=True)
            if use_softmax:
                spot_topic = spot_topic.apply(lambda row: softmax(row), axis=1)
            return spot_topic
        else:
            return F.normalize(z_mean).detach().cpu().numpy()

    def get_embedding_and_feature_by_topic(self, use_softmax=True, merge=True, threshold=.4):
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
            z_mean = self.model.get_embedding(self.data)
            columns_name = ["Shared topic {}".format(i + 1) for i in range(self.zs_dim)]
            for i in range(self.n_omics):
                columns_name += [self.omics_names[i] + ' private topic {}'.format(j + 1) for j in
                                 range(self.zp_dims[i])]
            spot_topic = DataFrame(z_mean.detach().cpu().numpy(), columns=columns_name)
            spot_topic.set_index(self.adatas[0].obs_names, inplace=True)
            feature_topic = self.model.get_feature_by_topic()
            for i in range(self.n_omics):
                feature_topic[self.omics_names[i]] = DataFrame(feature_topic[self.omics_names[i]],
                                                               columns=["Shared topic {}".format(j + 1) for j in
                                                                        range(self.zs_dim)] + [
                                                                           self.omics_names[
                                                                               i] + " private topic {}".format(j + 1)
                                                                           for j in
                                                                           range(self.zp_dims[i])],
                                                               index=self.adatas[i].var_names)

            # prune noisy topics and background topics
            meaningful_topics = ["Shared topic {}".format(i + 1) for i in self.meaningful_dimensions['zs']]
            for i in range(self.n_omics):
                meaningful_topics += [self.omics_names[i] + " private topic {}".format(j + 1) for j in
                                      self.meaningful_dimensions['zp_' + self.omics_names[i]]]
            spot_topic = spot_topic[meaningful_topics]
            for i in range(self.n_omics):
                existing_topics = [col for col in meaningful_topics if
                                   col in feature_topic[self.omics_names[i]].columns]
                feature_topic[self.omics_names[i]] = feature_topic[self.omics_names[i]][existing_topics]

            if merge:
                spot_topic, feature_topic = self.merge(spot_topic, feature_topic, threshold)
            if use_softmax:
                spot_topic = spot_topic.apply(lambda row: softmax(row), axis=1)
            return spot_topic, feature_topic
        else:
            raise Exception("This function can only be used with interpretable mode.")

    def merge(self, spot_topic, feature_topic, threshold=.4):
        # merge topics with similar features
        topks = []
        for i in range(self.n_omics):
            if self.data_dims[i] < 200:
                topks.append(5)
            else:
                topks.append(50)
        merge = True
        while merge:
            merge = False
            for topic_i in spot_topic.columns:
                oi = 'Shared' if 'Shared' in topic_i else topic_i.split(' private', 1)[0]
                for topic_j in spot_topic.columns:
                    oj = 'Shared' if 'Shared' in topic_j else topic_j.split(' private', 1)[0]
                    if spot_topic.columns.get_loc(topic_j) > spot_topic.columns.get_loc(topic_i) and oi == oj:
                        if oi == 'Shared':
                            sim = min([RankingSimilarity(
                                feature_topic[self.omics_names[i]].nlargest(topks[i], topic_i).index.tolist(),
                                feature_topic[self.omics_names[i]].nlargest(topks[i], topic_j).index.tolist()).rbo() for
                                       i in range(self.n_omics)])
                        else:
                            sim = RankingSimilarity(
                                feature_topic[oi].nlargest(topks[self.omics_names.index(oi)],
                                                           topic_i).index.tolist(),
                                feature_topic[oj].nlargest(topks[self.omics_names.index(oj)],
                                                           topic_j).index.tolist()).rbo()
                        if sim > threshold:
                            print('merge', topic_i, 'and', topic_j)
                            spot_topic.loc[:, topic_i] = (spot_topic[topic_i] + spot_topic[topic_j]) / 2
                            spot_topic = spot_topic.drop(columns=topic_j)
                            for on in self.omics_names:
                                if topic_i in feature_topic[on].columns:
                                    feature_topic[on].loc[:, topic_i] = (feature_topic[on][topic_i] + feature_topic[on][
                                        topic_j]) / 2
                                    feature_topic[on] = feature_topic[on].drop(columns=topic_j)
                            merge = True
                            break
                if merge:
                    break
            if not merge:
                break
        return spot_topic, feature_topic


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
