from __future__ import annotations

import warnings
import torch
import torch.nn.functional as F
from pyro.distributions import TorchDistribution
from torch.distributions import Gamma, constraints
from torch.distributions import Poisson as PoissonTorch
from torch.distributions.utils import broadcast_all, lazy_property, logits_to_probs, probs_to_logits

from scvi import settings

def log_zinb_positive(x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, zi_probs: torch.Tensor,
                      eps: float = 1e-8) -> torch.Tensor:
    """Log likelihood (scalar) of a minibatch according to a zinb model.

    Parameters
    ----------
    x
        Data
    mu
        mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
    theta
        inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
    pi
        logit of the dropout parameter (real support) (shape: minibatch x vars)
    eps
        numerical stability constant

    Notes
    -----
    We parametrize the bernoulli using the logits, hence the softplus functions appearing.
    """
    # theta is the dispersion rate. If .ndimension() == 1, it is shared for all cells (regardless
    # of batch or labels)
    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))  # In this case, we reshape theta for broadcasting

    # Uses log(sigmoid(x)) = -softplus(-x)
    softplus_pi = F.softplus(-zi_probs)
    log_theta_eps = (theta + eps).log()
    log_theta_mu_eps = (theta + mu + eps).log()
    pi_theta_log = -zi_probs + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = -softplus_pi + pi_theta_log + x * ((mu + eps).log() - log_theta_mu_eps) + (
                x + theta).lgamma() - theta.lgamma() - (x + 1).lgamma()
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero

    return res


def log_nb_positive(x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Log likelihood (scalar) of a minibatch according to a nb model.

    Parameters
    ----------
    x
        data
    mu
        mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
    theta
        inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
    eps
        numerical stability constant
    """
    log_theta_mu_eps = (theta + mu + eps).log()
    res = (theta * ((theta + eps).log() - log_theta_mu_eps) + x * ((mu + eps).log() - log_theta_mu_eps) + (
            x + theta).lgamma() - theta.lgamma() - (x + 1).lgamma())

    return res


def _gamma(theta: torch.Tensor, mu: torch.Tensor) -> Gamma:
    concentration = theta
    rate = theta / mu
    # Important remark: Gamma is parametrized by the rate = 1/scale!
    gamma_d = Gamma(concentration=concentration, rate=rate)
    return gamma_d


class NB(TorchDistribution):
    arg_constraints = {"mu": constraints.positive, "theta": constraints.positive}
    support = constraints.nonnegative

    def __init__(self, mu, theta, validate_args=None):
        mu, theta = broadcast_all(mu, theta)
        self.mu = mu
        self.theta = theta
        self._eps = 1e-8
        super().__init__(mu.shape, validate_args=validate_args)

    @property
    def mean(self):
        return self.mu

    @property
    def variance(self):
        return self.mean + (self.mean ** 2) / self.theta

    def _gamma(self):
        return _gamma(self.theta, self.mu)

    def sample(self, sample_shape=()):
        gamma_d = self._gamma()
        p_means = gamma_d.sample(sample_shape)
        l_train = torch.clamp(p_means, max=1e8)
        counts = PoissonTorch(l_train).sample()
        return counts

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return log_nb_positive(value, mu=self.mu, theta=self.theta, eps=self._eps)


class ZINB(TorchDistribution):
    arg_constraints = {"mu": constraints.positive, "theta": constraints.positive}
    support = constraints.nonnegative

    def __init__(self, mu, theta, zi_logits, validate_args=None):
        mu, theta, zi_logits = broadcast_all(mu, theta, zi_logits)
        self.mu = mu
        self.theta = theta
        self.zi_logits = zi_logits
        self._eps = 1e-8
        super().__init__(mu.shape, validate_args=validate_args)

    @property
    def mean(self):
        return (1 - self.zi_probs) * self.mu

    @property
    def variance(self):
        return (1 - self.zi_probs) * self.mu * (
                self.mu + self.theta + self.zi_probs * self.mu * self.theta) / self.theta

    @lazy_property
    def zi_logits(self) -> torch.Tensor:
        """ZI logits."""
        return probs_to_logits(self.zi_probs, is_binary=True)

    @lazy_property
    def zi_probs(self) -> torch.Tensor:
        return logits_to_probs(self.zi_logits, is_binary=True)

    def _gamma(self):
        return _gamma(self.theta, self.mu)

    def sample(self, sample_shape: torch.Size | tuple | None = None) -> torch.Tensor:
        """Sample from the distribution."""
        sample_shape = sample_shape or torch.Size()
        gamma_d = self._gamma()
        p_means = gamma_d.sample(sample_shape)
        # Clamping as distributions objects can have buggy behaviors when their parameters are too high
        l_train = torch.clamp(p_means, max=1e8)
        samp = PoissonTorch(l_train).sample()
        is_zero = torch.rand_like(samp) <= self.zi_probs
        samp_ = torch.where(is_zero, torch.zeros_like(samp), samp)
        return samp_

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Log probability."""
        try:
            self._validate_sample(value)
        except ValueError:
            warnings.warn("The value argument must be within the support of the distribution", UserWarning,
                          stacklevel=settings.warnings_stacklevel)
        return log_zinb_positive(value, self.mu, self.theta, self.zi_logits, eps=1e-08)
