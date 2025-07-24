import math
from typing import List

import pyro
import torch
import torch.nn.functional as F
from pyro import poutine
from pyro.distributions import constraints, InverseGamma, Normal, LogNormal
from pyro.nn import PyroModule, PyroParam
from torch import Tensor
from torch.distributions import HalfCauchy
from torch_geometric.nn import GAT

from .dist import NB, ZINB
from .layers import GATEncoder, Decoder, Decoder_zi_logits

scale_init = math.log(0.01)

class spamv(PyroModule):
    def __init__(self, data_dims: List[int], zs_dim: int, zp_dims: List[int], init_bg_means: List[Tensor],
                 weights: List[float], hidden_size: int, recon_types: List[str], heads: int, interpretable: bool,
                 device: torch.device, omics_names: List[str]):
        super().__init__()

        self.data_dims = data_dims
        self.zs_dim = zs_dim
        self.zp_dims = zp_dims
        self.latent_dims = [zs_dim + zp_dim for zp_dim in zp_dims]
        self.init_bg_means = init_bg_means
        self.weights = weights
        self.interpretable = interpretable
        self.prior = Normal
        self.recon_types = recon_types
        self.device = device
        self.omics_names = omics_names
        self.n_omics = len(data_dims)
        self.zs_plate = self.get_plate("zs")
        self.zp_plate = [self.get_plate("zp_" + omics_names[i]) if zp_dim > 0 else None for i, zp_dim in
                         enumerate(zp_dims)]

        for i in range(len(data_dims)):
            if interpretable:
                setattr(self, "disp_" + omics_names[i], PyroParam(self._zeros_init(data_dims[i])))
                if recon_types[i] == "zinb":
                    setattr(self, "decoder_zi_logits_" + omics_names[i],
                            Decoder_zi_logits(self.latent_dims[i], hidden_size, data_dims[i]))
                setattr(self, "zs_encoder_" + omics_names[i],
                        GATEncoder(data_dims[i], hidden_size, zs_dim, heads).to(device))
                if zp_dims[i] > 0:
                    if interpretable:
                        setattr(self, "zp_encoder_" + omics_names[i],
                                GATEncoder(data_dims[i], hidden_size, zp_dims[i], heads).to(device))
                self.set_attr(["c"], i, (1))
                self.set_attr(["delta", "bg"], i, data_dims[i])
                self.set_attr(["tau"], i, (self.latent_dims[i], 1))
                self.set_attr(["lambda", "beta"], i, (self.latent_dims[i], data_dims[i]))
            else:
                setattr(self, "disp_" + omics_names[i], PyroParam(self._zeros_init(data_dims[i])))
                setattr(self, "zs_encoder_" + omics_names[i],
                        GAT(in_channels=data_dims[i], hidden_channels=hidden_size, num_layers=1,
                            out_channels=zs_dim * 2, heads=heads, v2=True, concat=False))
                setattr(self, "decoder_" + omics_names[i],
                        Decoder(self.latent_dims[i], hidden_size, data_dims[i], recon_types[i]).to(device))
                if zp_dims[i] > 0:
                    setattr(self, "zp_encoder_" + omics_names[i],
                            GAT(in_channels=data_dims[i], hidden_channels=hidden_size, num_layers=1,
                                out_channels=zp_dims[i] * 2, heads=heads, v2=True, concat=False))
                    setattr(self, "zp_aux_std_" + omics_names[i],
                            PyroParam(self._ones_init(zp_dims[i]), constraint=constraints.positive))
        self.omics_plate = [self.get_plate("omics_" + omics_names[i]) for i in range(len(data_dims))]
        self.latent_omics_plate = [self.get_plate("latent_" + omics_names[i]) for i in range(len(data_dims))]

    def get_plate(self, name: str, **kwargs):
        """Get the sampling plate.

        Parameters
        ----------
        name : str
            Name of the plate

        Returns
        -------
        PlateMessenger
            A pyro plate.
        """
        plate_kwargs = {"zs": {"name": "zs", "size": self.zs_dim, "dim": -2}}
        for i in range(len(self.data_dims)):
            plate_kwargs["zp_" + self.omics_names[i]] = {"name": "zp_" + self.omics_names[i], "size": self.zp_dims[i],
                                                         "dim": -2}
            plate_kwargs["latent_" + self.omics_names[i]] = {"name": "latent_" + self.omics_names[i],
                                                             "size": self.latent_dims[i], "dim": -2}
            plate_kwargs["omics_" + self.omics_names[i]] = {"name": "omics_" + self.omics_names[i],
                                                            "size": self.data_dims[i], "dim": -1}
        return pyro.plate(**{**plate_kwargs[name], **kwargs})

    def _zeros_init(self, shape):
        return torch.zeros(shape, device=self.device)

    def _ones_init(self, shape, multiplier=0.1):
        return torch.ones(shape, device=self.device) * multiplier

    def set_attr(self, names, omics_id, shape):
        for name in names:
            setattr(self, name + '_mean_' + self.omics_names[omics_id], PyroParam(self._zeros_init(shape)))
            setattr(self, name + '_std_' + self.omics_names[omics_id],
                    PyroParam(self._ones_init(shape), constraint=constraints.positive))

    def model(self, datas):
        pyro.module("spamv", self)
        batch_size = datas[0].num_nodes
        sample_plate = pyro.plate("sample", batch_size)
        zss = []
        zps = []
        zp_auxs = []
        
        if self.interpretable:
            lss = []
            betas = []
        for i, data in zip(range(self.n_omics), datas):
            if self.interpretable:
                lss.append(data.x[:batch_size].sum(-1, keepdim=True))
                c = pyro.sample("c_" + self.omics_names[i],
                                InverseGamma(torch.ones(1, device=self.device) * 0.5, torch.ones(1, device=self.device) * 0.5))
                with self.omics_plate[i]:
                    delta = pyro.sample("delta_" + self.omics_names[i], HalfCauchy(torch.ones(1, device=self.device)))
                    bg = pyro.sample("bg_" + self.omics_names[i],
                                     Normal(torch.zeros(1, device=self.device), torch.ones(1, device=self.device)))
                bg = bg + self.init_bg_means[i]
                with self.latent_omics_plate[i]:
                    tau = pyro.sample("tau_" + self.omics_names[i], HalfCauchy(torch.ones(1, device=self.device)))
                    with self.omics_plate[i]:
                        lambda_ = pyro.sample("lambda_" + self.omics_names[i], HalfCauchy(torch.ones(1, device=self.device)))
                        beta = pyro.sample("beta_" + self.omics_names[i],
                                           Normal(torch.zeros(1, device=self.device), torch.ones(1, device=self.device)))
                lambda_tilde = (c ** 2 * tau ** 2 * delta ** 2 * lambda_ ** 2 / (
                        c ** 2 + tau ** 2 * delta ** 2 * lambda_ ** 2)).sqrt()
                betas.append(beta * lambda_tilde + bg)
            with sample_plate:
                zs = pyro.sample("zs_" + self.omics_names[i], self.prior(torch.zeros(self.zs_dim, device=self.device),
                                                                         torch.ones(self.zs_dim,
                                                                                    device=self.device)).to_event(1))
                zss.append(zs)
                if self.zp_dims[i] > 0:
                    zp = pyro.sample("zp_" + self.omics_names[i],
                                     self.prior(torch.zeros(self.zp_dims[i], device=self.device),
                                                torch.ones(self.zp_dims[i], device=self.device)).to_event(1))
                    zps.append(zp)
                    zp_auxs.append(
                        self._zeros_init((batch_size, self.zp_dims[i])) if self.interpretable else Normal(
                            torch.zeros(self.zp_dims[i], device=self.device),
                            getattr(self, "zp_aux_std_" + self.omics_names[i])).rsample((batch_size,)))
                else:
                    zps.append(None)
                    zp_auxs.append(None)
        for i in range(self.n_omics):
            for j in range(self.n_omics):
                if i == j:
                    # self reconstruction
                    z = torch.cat((zss[i].detach(), zps[j]), dim=1) if self.zp_dims[j] > 0 else zss[i].detach()
                else:
                    # cross reconstruction (using shared embedding from data i to reconstruct data j)
                    z = torch.cat((zss[i], zp_auxs[j]), dim=1) if self.zp_dims[j] > 0 else zss[i]
                if self.interpretable:
                    x_tilde = lss[j] * F.softmax(z, 1) @ F.softmax(betas[j], 1)
                    if self.recon_types[j] == 'zinb':
                        zi_logits = getattr(self, "decoder_zi_logits_" + self.omics_names[j])(z)
                else:
                    if self.recon_types[j] in ['nb', 'gauss']:
                        x_tilde = getattr(self, "decoder_" + self.omics_names[j])(z)
                    elif self.recon_types[j] in ['zinb']:
                        x_tilde, zi_logits = getattr(self, "decoder_" + self.omics_names[j])(z)
                    else:
                        raise NotImplementedError
                with sample_plate:
                    with poutine.scale(scale=self.weights[j]):
                        disp = getattr(self, "disp_" + self.omics_names[j]).exp()
                        if self.recon_types[j] == 'nb':
                            pyro.sample("recon_" + self.omics_names[j] + "_from_" + self.omics_names[i],
                                        NB(x_tilde, disp).to_event(1), obs=datas[j].x[:batch_size])
                        elif self.recon_types[j] == 'zinb':
                            pyro.sample("recon_" + self.omics_names[j] + "_from_" + self.omics_names[i],
                                        ZINB(x_tilde, disp, zi_logits).to_event(1), obs=datas[j].x[:batch_size])
                        elif self.recon_types[j] == 'gauss':
                            pyro.sample("recon_" + self.omics_names[j] + "_from_" + self.omics_names[i],
                                        Normal(x_tilde, disp).to_event(1), obs=datas[j].x[:batch_size])
                        else:
                            raise NotImplementedError

    def guide(self, datas):
        batch_size = datas[0].num_nodes
        sample_plate = pyro.plate("sample", batch_size)
        for i, data in zip(range(self.n_omics), datas):
            if self.interpretable:
                pyro.sample("c_" + self.omics_names[i], LogNormal(getattr(self, 'c_mean_' + self.omics_names[i]),
                                                                  getattr(self, 'c_std_' + self.omics_names[i])))
                with self.omics_plate[i]:
                    pyro.sample("delta_" + self.omics_names[i],
                                LogNormal(getattr(self, 'delta_mean_' + self.omics_names[i]),
                                          getattr(self, 'delta_std_' + self.omics_names[i])))
                    pyro.sample("bg_" + self.omics_names[i], Normal(getattr(self, 'bg_mean_' + self.omics_names[i]),
                                                                    getattr(self, 'bg_std_' + self.omics_names[i])))
                with self.latent_omics_plate[i]:
                    pyro.sample("tau_" + self.omics_names[i],
                                LogNormal(getattr(self, 'tau_mean_' + self.omics_names[i]),
                                          getattr(self, 'tau_std_' + self.omics_names[i])))
                    with self.omics_plate[i]:
                        pyro.sample("lambda_" + self.omics_names[i],
                                    LogNormal(getattr(self, 'lambda_mean_' + self.omics_names[i]),
                                              getattr(self, 'lambda_std_' + self.omics_names[i])))
                        pyro.sample("beta_" + self.omics_names[i],
                                    Normal(getattr(self, 'beta_mean_' + self.omics_names[i]),
                                           getattr(self, 'beta_std_' + self.omics_names[i])))
            with sample_plate:
                zs_mean, zs_scale = getattr(self, "zs_encoder_" + self.omics_names[i])(data.x, data.edge_index)[:batch_size].split(self.zs_dim, 1)
                pyro.sample("zs_" + self.omics_names[i], self.prior(zs_mean, zs_scale if self.interpretable else (
                        zs_scale.clamp(min=-10, max=5) / 2).exp()).to_event(1))
                if self.zp_dims[i] > 0:
                    zp_mean, zp_scale = getattr(self, "zp_encoder_" + self.omics_names[i])(data.x, data.edge_index)[:batch_size].split(self.zp_dims[i], 1)
                    pyro.sample("zp_" + self.omics_names[i], self.prior(zp_mean, zp_scale if self.interpretable else (
                            zp_scale.clamp(min=-10, max=5) / 2).exp()).to_event(1))

    def get_embedding(self, datas, train_eval=False):
        batch_size = datas[0].num_nodes
        if train_eval:
            self.train()
        else:
            self.eval()
        z_mean = torch.zeros((batch_size, self.zs_dim), device=self.device)
        for i, data in zip(range(self.n_omics), datas):
            zs_mean = getattr(self, "zs_encoder_" + self.omics_names[i])(data.x, data.edge_index)[:batch_size].split(self.zs_dim, 1)[0]
            z_mean += zs_mean / self.n_omics
        for i, data in zip(range(self.n_omics), datas):
            if self.zp_dims[i] > 0:
                zp_mean = getattr(self, "zp_encoder_" + self.omics_names[i])(data.x, data.edge_index)[:batch_size].split(self.zp_dims[i], 1)[0]
                z_mean = torch.cat((z_mean, zp_mean), dim=1)
        return z_mean

    def get_private_latent(self, datas, train_eval=False):
        batch_size = datas[0].num_nodes
        zp = []
        if train_eval:
            self.train()
        else:
            self.eval()
        for i, data in zip(range(self.n_omics), datas):
            if self.zp_dims[i] > 0:
                zp_mean, zp_scale = getattr(self, "zp_encoder_" + self.omics_names[i])(data.x, data.edge_index)[:batch_size].split(self.zp_dims[i], 1)
                zp.append(self.prior(zp_mean, zp_scale if self.interpretable else (
                            zp_scale.clamp(min=-10, max=5) / 2).exp()).rsample())
        return zp

    def get_private_embedding(self, datas):
        batch_size = datas[0].num_nodes
        zp = []
        for i, data in zip(range(self.n_omics), datas):
            if self.zp_dims[i] > 0:
                zp_mean = getattr(self, "zp_encoder_" + self.omics_names[i])(data.x, data.edge_index)[:batch_size].split(self.zp_dims[i], 1)[0]
                zp.append(zp_mean)
        return zp

    def get_shared_embedding(self, datas):
        batch_size = datas[0].num_nodes
        zs = torch.zeros((batch_size, self.zs_dim), device=self.device)
        for i, data in zip(range(self.n_omics), datas):
            zs += getattr(self, "zs_encoder_" + self.omics_names[i])(data.x, data.edge_index)[:batch_size].split(self.zs_dim, 1)[0] / self.n_omics
        return zs

    def get_separate_embedding(self, data, edge_index):
        self.eval()
        output = {}
        with torch.no_grad():
            for i, d, e in zip(range(len(data)), data, edge_index):
                zs_mean = getattr(self, "zs_encoder_" + self.omics_names[i])(d, e).split(self.zs_dim, 1)[0]
                output['zs_' + self.omics_names[i]] = zs_mean
                if self.zp_dims[i] > 0:
                    zp_mean = getattr(self, "zp_encoder_" + self.omics_names[i])(d, e).split(self.zp_dims[i], 1)[0]
                    output['zp_' + self.omics_names[i]] = zp_mean
        return output

    @torch.inference_mode()
    def get_feature_by_topic(self):
        if self.interpretable:
            betas = {}
            for i in range(len(self.data_dims)):
                tau = self.mean(getattr(self, 'tau_mean_' + self.omics_names[i]),
                                getattr(self, 'tau_std_' + self.omics_names[i]))
                delta = self.mean(getattr(self, 'delta_mean_' + self.omics_names[i]),
                                  getattr(self, 'delta_std_' + self.omics_names[i]))
                lambda_ = self.mean(getattr(self, 'lambda_mean_' + self.omics_names[i]),
                                    getattr(self, 'lambda_std_' + self.omics_names[i]))
                c = self.mean(getattr(self, 'c_mean_' + self.omics_names[i]),
                              getattr(self, 'c_std_' + self.omics_names[i]))
                lambda_tilde = (c ** 2 * tau ** 2 * delta ** 2 * lambda_ ** 2 / (
                        c ** 2 + tau ** 2 * delta ** 2 * lambda_ ** 2)).sqrt()
                beta = getattr(self, 'beta_mean_' + self.omics_names[i]) * lambda_tilde
                bg = getattr(self, 'bg_mean_' + self.omics_names[i]) + self.init_bg_means[i]
                bg = bg.exp()
                adj = (bg + torch.quantile(bg, .1)).log() - bg.log()
                betas[self.omics_names[i]] = (beta - adj).detach().cpu().numpy().transpose()
        else:
            raise Exception("Please set interpretable=True to use this function.")
        return betas

    def mean(self, loc, scale):
        return LogNormal(loc, scale).mean

    def variance(self, loc, scale):
        return LogNormal(loc, scale).variance

    def get_tide(self):
        tide = self.tide_loc
        tide = torch.cat(
            [
                torch.zeros_like(tide[:, :, None]).expand(-1, self.n_genes, 1),
                tide[:, :, None].expand(-1, self.n_genes),
            ],
            dim=-1,
        )
        return tide

    def get_bg(self):
        bg_omics1 = self.bg_loc_omics1 + self.init_bg_mean_omics1
        bg_omics2 = self.bg_loc_omics2 + self.init_bg_mean_omics2
        return bg_omics1.exp(), bg_omics2.exp()

    def save(self, path):
        pyro.get_param_store().save(path)

    def load(self, path, map_location=torch.device('cpu')):
        pyro.get_param_store().load(path, map_location=map_location)
        pyro.module("zs_encocder_0", self.zs_encoder_0, update_module_params=True)
        pyro.module("zs_encocder_1", self.zs_encoder_1, update_module_params=True)
