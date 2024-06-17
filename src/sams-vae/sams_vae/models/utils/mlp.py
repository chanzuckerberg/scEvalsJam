from typing import Literal

import torch
from pyro.distributions import GammaPoisson
from torch import nn
from torch.distributions import Normal, Poisson

LIKELIHOOD_KEY_DTYPE = Literal["normal", "poisson", "library_nb"]


def get_likelihood_mlp(
    likelihood_key: LIKELIHOOD_KEY_DTYPE,
    n_input: int,
    n_output: int,
    n_layers: int,
    n_hidden: int,
    use_batch_norm: bool,
    activation_fn: nn.Module = nn.LeakyReLU,
) -> nn.Module:

    mlp_class: nn.Module
    if likelihood_key == "normal":
        mlp_class = GaussianLikelihoodResidualMLP
    elif likelihood_key == "normal_fixed_variance":
        mlp_class = GaussianLikelihoodFixedVarianceResidualMLP
    elif likelihood_key == "poisson":
        mlp_class = PoissonLikelihoodResidualMLP
    else:
        mlp_class = LibraryGammaPoissonSharedConcentrationResidualMLP

    mlp = mlp_class(
        n_input=n_input,
        n_output=n_output,
        n_layers=n_layers,
        n_hidden=n_hidden,
        use_batch_norm=use_batch_norm,
        activation_fn=activation_fn,
    )
    return mlp


class MLP(nn.Module):
    """Basic MLP with constant hidden dim for all layers"""

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int,
        n_hidden: int,
        use_batch_norm: bool,
        activation_fn: nn.Module = nn.LeakyReLU,
        use_activation: bool = True,
        last_layer_activation: bool = True,
    ):
        super().__init__()
        layer_dims = [n_input] + n_layers * [n_hidden] + [n_output]
        layers = []
        for i in range(1, len(layer_dims)):
            skip_activation = (not last_layer_activation) and (i == len(layer_dims) - 1)
            layer_in = layer_dims[i - 1]
            layer_out = layer_dims[i]
            sublayers = [
                nn.Linear(layer_in, layer_out),
                nn.BatchNorm1d(layer_out)
                if use_batch_norm and not skip_activation
                else None,
                activation_fn() if use_activation and not skip_activation else None,
            ]
            sublayers = [sl for sl in sublayers if sl is not None]
            layer = nn.Sequential(*sublayers)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = self.layers[0](x)
        for layer in self.layers[1:]:
            x = layer(x)
        return x


class ResidualMLP(MLP):
    """Basic MLP with constant hidden dimension and residual connections"""

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int,
        n_hidden: int,
        use_batch_norm: bool,
        activation_fn: nn.Module = nn.LeakyReLU,
        use_activation: bool = True,
        last_layer_activation: bool = True,
        last_layer_residual: bool = False,
    ):
        super().__init__(
            n_input,
            n_output,
            n_layers,
            n_hidden,
            use_batch_norm,
            activation_fn,
            use_activation,
            last_layer_activation,
        )
        self.last_layers_residual = last_layer_residual
        assert (not last_layer_residual) or (n_output == n_hidden)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(x)
            elif i == len(self.layers) - 1 and not self.last_layers_residual:
                x = layer(x)
            else:
                x = layer(x) + x
        return x


class BaseGaussianLikelihoodMLP(nn.Module):
    def forward(self, x: torch.Tensor):
        multiple_particles = len(x.shape) == 3
        if multiple_particles:
            n_particles, n, x_dim = x.shape
            x = x.reshape(n_particles * n, x_dim)

        z = self.mlp(x)
        mean = self.mean_encoder(z)
        var = torch.exp(self.log_var_encoder(z)) + self.var_eps

        if multiple_particles:
            mean = mean.reshape(n_particles, n, -1)
            var = var.reshape(n_particles, n, -1)
        dist = Normal(mean, var.sqrt())
        return dist


class BaseLibraryGammaPoissonSharedConcentrationLikelihood(nn.Module):
    def forward(self, x: torch.Tensor, library_size: torch.Tensor):
        multiple_particles = len(x.shape) == 3
        if multiple_particles:
            n_particles, n, x_dim = x.shape
            x = x.reshape(n_particles * n, x_dim)
            library_size = library_size.expand(n_particles, -1).reshape(-1, 1)

        z = self.mlp(x)
        normalized_mu = self.normalized_mean_decoder(z)
        mu = library_size * normalized_mu

        if multiple_particles:
            mu = mu.reshape(n_particles, n, -1)

        concentration = torch.exp(self.log_concentration)
        mu_eps = 1e-4
        dist = GammaPoisson(
            concentration=concentration, rate=concentration / (mu + mu_eps)
        )
        # dist = NegativeBinomial(mu=mu, theta=theta)
        return dist


class BasePoissonLikelihoodMLP(nn.Module):
    def forward(self, x: torch.Tensor):
        multiple_particles = len(x.shape) == 3
        if multiple_particles:
            n_particles, n, x_dim = x.shape
            x = x.reshape(n_particles * n, x_dim)

        z = self.mlp(x)
        rate = self.rate_decoder(z)

        if multiple_particles:
            rate = rate.reshape(n_particles, n, -1)

        dist = Poisson(rate=rate)
        return dist


class GaussianLikelihoodMLP(BaseGaussianLikelihoodMLP):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int,
        n_hidden: int,
        use_batch_norm: bool,
        activation_fn: nn.Module = nn.LeakyReLU,
        use_activation: bool = True,
        var_eps: float = 1e-4,
    ):
        super().__init__()
        self.mlp = MLP(
            n_input=n_input,
            n_output=n_hidden,
            n_layers=n_layers - 1,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm,
            activation_fn=activation_fn,
            use_activation=use_activation,
            last_layer_activation=True,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.log_var_encoder = nn.Linear(n_hidden, n_output)
        self.var_eps = var_eps


class GaussianLikelihoodResidualMLP(BaseGaussianLikelihoodMLP):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int,
        n_hidden: int,
        use_batch_norm: bool,
        activation_fn: nn.Module = nn.LeakyReLU,
        use_activation: bool = True,
        var_eps: float = 1e-4,
    ):
        super().__init__()
        self.mlp = ResidualMLP(
            n_input=n_input,
            n_output=n_hidden,
            n_layers=n_layers - 1,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm,
            activation_fn=activation_fn,
            use_activation=use_activation,
            last_layer_activation=True,
            last_layer_residual=True,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.log_var_encoder = nn.Linear(n_hidden, n_output)
        self.var_eps = var_eps


class FixedLogVarianceEncoder(torch.nn.Module):
    def __init__(self, n_input: int):
        super().__init__()
        self.log_var = torch.nn.Parameter(torch.zeros((n_input,)))

    def forward(self, z):
        expand_args = list(z.shape[:-1]) + [-1]
        var = self.log_var.expand(expand_args)
        return var


class GaussianLikelihoodFixedVarianceResidualMLP(BaseGaussianLikelihoodMLP):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int,
        n_hidden: int,
        use_batch_norm: bool,
        activation_fn: nn.Module = nn.LeakyReLU,
        use_activation: bool = True,
        var_eps: float = 1e-4,
    ):
        super().__init__()
        self.mlp = ResidualMLP(
            n_input=n_input,
            n_output=n_hidden,
            n_layers=n_layers - 1,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm,
            activation_fn=activation_fn,
            use_activation=use_activation,
            last_layer_activation=True,
            last_layer_residual=True,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.log_var_encoder = FixedLogVarianceEncoder(n_output)
        self.var_eps = var_eps


class LibraryGammaPoissonSharedConcentrationResidualMLP(
    BaseLibraryGammaPoissonSharedConcentrationLikelihood
):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int,
        n_hidden: int,
        use_batch_norm: bool,
        activation_fn: nn.Module = nn.LeakyReLU,
        use_activation: bool = True,
    ):
        super().__init__()
        self.mlp = ResidualMLP(
            n_input=n_input,
            n_output=n_hidden,
            n_layers=n_layers - 1,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm,
            activation_fn=activation_fn,
            use_activation=use_activation,
            last_layer_activation=True,
            last_layer_residual=True,
        )
        # predicts normalized mean expression (mean / library_size)
        self.normalized_mean_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            nn.Softmax(dim=-1),
        )
        # log concentration for gamma distribution
        self.log_concentration = nn.Parameter(torch.zeros((n_output,)))


class PoissonLikelihoodResidualMLP(BasePoissonLikelihoodMLP):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int,
        n_hidden: int,
        use_batch_norm: bool,
        activation_fn: nn.Module = nn.LeakyReLU,
        use_activation: bool = True,
    ):
        super().__init__()
        self.mlp = ResidualMLP(
            n_input=n_input,
            n_output=n_hidden,
            n_layers=n_layers - 1,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm,
            activation_fn=activation_fn,
            use_activation=use_activation,
            last_layer_activation=True,
            last_layer_residual=True,
        )
        self.rate_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            nn.Softplus(),
        )
