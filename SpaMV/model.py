import math
from typing import List

import pyro
import torch
import torch.nn.functional as F
from pandas import DataFrame
from pyro import poutine
from pyro.distributions import constraints, InverseGamma, Normal, LogNormal
from pyro.nn import PyroModule, PyroParam
from torch import nn, Tensor
from torch.distributions import HalfCauchy

from . import dist
from .dist import NB, ZINB
from .layers import MLPEncoder, Decoder

scale_init = math.log(0.01)


class spamv(PyroModule):
    def __init__(self, data_dims: List[int], zs_dim: int, zp_dims: List[int], init_bg_means: List[Tensor],
                 weights: List[float], hidden_size: int, recon_types: List[str], heads: int, interpretable: bool,
                 device: torch.device):
        super().__init__()

        self.data_dims = data_dims
        self.zs_dim = zs_dim
        self.zp_dims = zp_dims
        self.latent_dims = [zs_dim + zp_dim for zp_dim in zp_dims]
        self.init_bg_means = init_bg_means
        self.weights = weights
        self.interpretable = interpretable
        self.prior = LogNormal if interpretable else Normal
        self.recon_types = recon_types
        self.zs_plate = self.get_plate("zs")
        self.zp_plate = [self.get_plate("zp_{}".format(i)) if zp_dim > 0 else None for i, zp_dim in enumerate(zp_dims)]
        self.device = device
        for i in range(len(data_dims)):
            setattr(self, "disp_{}".format(i), PyroParam(torch.randn(data_dims[i], device=device)))
            setattr(self, "zs_encoder_{}".format(i),
                    MLPEncoder(data_dims[i], hidden_size, zs_dim, heads, interpretable).to(device))
            if zp_dims[i] > 0:
                setattr(self, "zp_encoder_{}".format(i),
                        MLPEncoder(data_dims[i], hidden_size, zp_dims[i], heads, interpretable).to(device))
                # setattr(self, "zp_aux_logstd_{}".format(i), PyroParam(torch.zeros(zp_dims[i], device=device)))
                setattr(self, "zp_aux_std_{}".format(i),
                        PyroParam(self._ones_init(zp_dims[i], device=device), constraint=constraints.positive))
            if interpretable:
                setattr(self, "c_mean_{}".format(i), PyroParam(torch.zeros(1, device=device)))
                setattr(self, "c_std_{}".format(i),
                        PyroParam(self._ones_init(1, device=device), constraint=constraints.positive))
                setattr(self, 'delta_mean_{}'.format(i), PyroParam(torch.zeros(data_dims[i], device=device)))
                setattr(self, 'delta_std_{}'.format(i),
                        PyroParam(self._ones_init(data_dims[i], device=device), constraint=constraints.positive))
                setattr(self, 'bg_mean_{}'.format(i), PyroParam(torch.zeros(data_dims[i], device=device)))
                setattr(self, 'bg_std_{}'.format(i),
                        PyroParam(self._ones_init(data_dims[i], device=device), constraint=constraints.positive))
                setattr(self, 'tau_mean_{}'.format(i), PyroParam(torch.zeros((self.latent_dims[i], 1), device=device)))
                setattr(self, 'tau_std_{}'.format(i),
                        PyroParam(self._ones_init((self.latent_dims[i], 1), device=device),
                                  constraint=constraints.positive))
                setattr(self, 'lambda_mean_{}'.format(i),
                        PyroParam(torch.zeros((self.latent_dims[i], data_dims[i]), device=device)))
                setattr(self, 'lambda_std_{}'.format(i),
                        PyroParam(self._ones_init((self.latent_dims[i], data_dims[i]), device=device)))
                setattr(self, 'beta_mean_{}'.format(i),
                        PyroParam(torch.zeros((self.latent_dims[i], data_dims[i]), device=device)))
                setattr(self, 'beta_std_{}'.format(i),
                        PyroParam(self._ones_init((self.latent_dims[i], data_dims[i]), device=device),
                                  constraint=constraints.positive))
                if recon_types[i] == "zinb":
                    setattr(self, 'zi_logits_{}'.format(i),
                            PyroParam(torch.zeros((self.latent_dims[i], data_dims[i]), device=device)))
            else:
                setattr(self, "decoder_{}".format(i),
                        Decoder(self.latent_dims[i], hidden_size, data_dims[i], recon_types[i]).to(device))
        self.omics_plate = [self.get_plate("omics_{}".format(i)) for i in range(len(data_dims))]
        self.latent_omics_plate = [self.get_plate("latent_{}".format(i)) for i in range(len(data_dims))]

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
            plate_kwargs["zp_{}".format(i)] = {"name": "zp_{}".format(i), "size": self.zp_dims[i], "dim": -2}
            plate_kwargs["latent_{}".format(i)] = {"name": "latent_{}".format(i), "size": self.latent_dims[i],
                                                   "dim": -2}
            plate_kwargs["omics_{}".format(i)] = {"name": "omics_{}".format(i), "size": self.data_dims[i], "dim": -1}
        return pyro.plate(**{**plate_kwargs[name], **kwargs})

    def _ones_init(self, shape, device=torch.device("cpu"), multiplier=0.1):
        return torch.ones(shape, device=device) * multiplier

    def model(self, data, edge_index):
        pyro.module("spamv", self)
        device = data[0].device
        sample_plate = pyro.plate("sample", data[0].shape[0])
        zss = []
        zps = []
        zp_auxs = []
        lss = []
        if self.interpretable:
            betas = []
        for i, d, e in zip(range(len(data)), data, edge_index):
            lss.append(d.sum(-1, keepdim=True))
            if self.interpretable:
                c = pyro.sample("c_{}".format(i),
                                InverseGamma(torch.ones(1, device=device) * 0.5, torch.ones(1, device=device) * 0.5))
                with self.omics_plate[i]:
                    delta = pyro.sample("delta_{}".format(i), HalfCauchy(torch.ones(1, device=device)))
                    bg = pyro.sample("bg_{}".format(i),
                                     Normal(torch.zeros(1, device=device), torch.ones(1, device=device)))
                bg = bg + self.init_bg_means[i]
                with self.latent_omics_plate[i]:
                    tau = pyro.sample("tau_{}".format(i), HalfCauchy(torch.ones(1, device=device)))
                    with self.omics_plate[i]:
                        lambda_ = pyro.sample("lambda_{}".format(i), HalfCauchy(torch.ones(1, device=device)))
                        beta = pyro.sample("beta_{}".format(i),
                                           Normal(torch.zeros(1, device=device), torch.ones(1, device=device)))
                lambda_tilde = (c ** 2 * tau ** 2 * delta ** 2 * lambda_ ** 2 / (
                        c ** 2 + tau ** 2 * delta ** 2 * lambda_ ** 2)).sqrt()
                betas.append(beta * lambda_tilde + bg)
            with sample_plate:
                zss.append(pyro.sample("zs_{}".format(i), self.prior(torch.zeros(self.zs_dim, device=device),
                                                                     torch.ones(self.zs_dim, device=device)).to_event(
                    1)))
                if self.zp_dims[i] > 0:
                    zps.append(pyro.sample("zp_{}".format(i), self.prior(torch.zeros(self.zp_dims[i], device=device),
                                                                         torch.ones(self.zp_dims[i],
                                                                                    device=device)).to_event(1)))
                    zp_auxs.append(torch.zeros((data[0].shape[0], self.zp_dims[i]),
                                               device=device) if self.interpretable else Normal(
                        torch.zeros(self.zp_dims[i], device=device), getattr(self, "zp_aux_std_{}".format(i))).rsample(
                        (d.shape[0],)))

        for i in range(len(data)):
            for j in range(len(data)):
                if i == j:
                    # self reconstruction
                    z = torch.cat((zss[i], zps[j]), dim=1) if self.zp_dims[j] > 0 else zss[i]
                else:
                    # cross reconstruction (using shared embedding from data i to reconstruct data j)
                    z = torch.cat((zss[i], zp_auxs[j]), dim=1) if self.zp_dims[j] > 0 else zss[i]
                if self.interpretable:
                    x_tilde = z @ F.softplus(betas[j]) + 1e-10
                    x_tilde = lss[j] * x_tilde / x_tilde.sum(1, keepdim=True)
                else:
                    if self.recon_types[j] == 'nb':
                        x_tilde = getattr(self, "decoder_{}".format(j))(z)
                    elif self.recon_types[j] == 'zinb':
                        x_tilde, zi_logits = getattr(self, "decoder_{}".format(j))(z)
                    else:
                        raise NotImplementedError
                with sample_plate:
                    with poutine.scale(scale=self.weights[j]):
                        if self.recon_types[j] == 'nb':
                            if (x_tilde <= 0).sum() > 0:
                                breakpoint()
                            pyro.sample("recon_{}_from_{}".format(j, i),
                                        NB(x_tilde, getattr(self, "disp_{}".format(j)).exp()).to_event(1), obs=data[j])
                        elif self.recon_types[j] == 'zinb':
                            pyro.sample("recon_{}_from_{}".format(j, i),
                                        ZINB(x_tilde, getattr(self, "disp_{}".format(j)).exp(), zi_logits).to_event(1),
                                        obs=data[j])
                        else:
                            raise NotImplementedError

    def guide(self, data, edge_index):
        sample_plate = pyro.plate("sample", data[0].shape[0])
        for i, d, e in zip(range(len(data)), data, edge_index):
            if self.interpretable:
                pyro.sample("c_{}".format(i),
                            LogNormal(getattr(self, 'c_mean_{}'.format(i)), getattr(self, 'c_std_{}'.format(i))))
                with self.omics_plate[i]:
                    pyro.sample("delta_{}".format(i), LogNormal(getattr(self, 'delta_mean_{}'.format(i)),
                                                                getattr(self, 'delta_std_{}'.format(i))))
                    pyro.sample("bg_{}".format(i),
                                Normal(getattr(self, 'bg_mean_{}'.format(i)), getattr(self, 'bg_std_{}'.format(i))))
                with self.latent_omics_plate[i]:
                    pyro.sample("tau_{}".format(i), LogNormal(
                        getattr(self, 'tau_mean_{}'.format(i)), getattr(self, 'tau_std_{}'.format(i))))
                    with self.omics_plate[i]:
                        pyro.sample("lambda_{}".format(i), LogNormal(
                            getattr(self, 'lambda_mean_{}'.format(i)), getattr(self, 'lambda_std_{}'.format(i))))
                        pyro.sample("beta_{}".format(i), Normal(
                            getattr(self, 'beta_mean_{}'.format(i)), getattr(self, 'beta_std_{}'.format(i))))
            with sample_plate:
                zs_mean, zs_scale = getattr(self, "zs_encoder_{}".format(i))(d, e)
                pyro.sample("zs_{}".format(i),
                            self.prior(zs_mean, zs_scale if self.interpretable else (zs_scale / 2).exp()).to_event(1))
                if self.zp_dims[i] > 0:
                    zp_mean, zp_scale = getattr(self, "zp_encoder_{}".format(i))(d, e)
                    pyro.sample("zp_{}".format(i),
                                self.prior(zp_mean, zp_scale if self.interpretable else (zp_scale / 2).exp()).to_event(
                                    1))

    def get_embedding(self, data, edge_index):
        self.eval()
        with torch.no_grad():
            z_mean = torch.zeros((data[0].shape[0], self.zs_dim), device=self.device)
            for i, d, e in zip(range(len(data)), data, edge_index):
                zs_mean_i, zs_std_i = getattr(self, "zs_encoder_{}".format(i))(d, e)
                if self.interpretable:
                    z_mean += self.mean(zs_mean_i, zs_std_i) / len(data)
                else:
                    z_mean += zs_mean_i / len(data)
            for i, d, e in zip(range(len(data)), data, edge_index):
                if self.zp_dims[i] > 0:
                    zp_mean_i, zp_std_i = getattr(self, "zp_encoder_{}".format(i))(d, e)
                    z_mean = torch.cat((z_mean, self.mean(zp_mean_i, zp_std_i) if self.interpretable else zp_mean_i),
                                       dim=1)
        return z_mean

    @torch.inference_mode()
    def get_feature_by_topic(self):
        if self.interpretable:
            betas = []
            for i in range(len(self.data_dims)):
                tau = self.mean(getattr(self, 'tau_mean_{}'.format(i)), getattr(self, 'tau_std_{}'.format(i)))
                delta = self.mean(getattr(self, 'delta_mean_{}'.format(i)), getattr(self, 'delta_std_{}'.format(i)))
                lambda_ = self.mean(getattr(self, 'lambda_mean_{}'.format(i)), getattr(self, 'lambda_std_{}'.format(i)))
                c = self.mean(getattr(self, 'c_mean_{}'.format(i)), getattr(self, 'c_std_{}'.format(i)))
                lambda_tilde = (c ** 2 * tau ** 2 * delta ** 2 * lambda_ ** 2 / (
                        c ** 2 + tau ** 2 * delta ** 2 * lambda_ ** 2)).sqrt()
                beta = getattr(self, 'beta_mean_{}'.format(i)) * lambda_tilde
                betas.append(beta.detach().cpu().numpy().transpose())
        else:
            raise Exception("Please set interpretable=True to use this function.")
        return betas

    def mean(self, loc, scale):
        return LogNormal(loc, scale).mean

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
