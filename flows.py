# Copyright (c) Facebook, Inc. and its affiliates.

import jax
import jax.numpy as jnp
from jax import random
from jax.nn import initializers as init
from jax.scipy.special import logsumexp

from flax import linen as nn

import hydra
import omegaconf

from manifolds import Manifold, Sphere, Product
import densities
import utils


def init_uniform(minval, maxval, dtype=jnp.float32):
    def init(key, shape, dtype=dtype):
        return random.uniform(key, shape, dtype, minval=minval, maxval=maxval)
    return init

def init_manifold_samples(dist, dtype=jnp.float32):
    def init(key, shape, dtype=dtype):
        D, N = shape
        samples = dist.sample(key, N).T
        assert samples.shape == (D, N)
        return samples
    return init

def init_full(val, dtype=jnp.float32):
    def init(key, shape, dtype=dtype):
        return jnp.full(shape, val)
    return init


class RadialPotential(nn.Module):
    n_radial_components: int
    init_beta_minval: float
    init_beta_range: float
    manifold: Manifold

    def setup(self):
        assert isinstance(self.manifold, Sphere)
        mu_init = densities.get_uniform(self.manifold)
        self.betas = self.param(
            'betas', init_uniform(
                minval=self.init_beta_minval,
                maxval=self.init_beta_minval+self.init_beta_range
            ), [self.n_radial_components])
        self.mus = self.param(
            'mus', init_manifold_samples(mu_init),
            [self.manifold.D, self.n_radial_components])
        self.alphas = self.param(
            'alphas', init_full(1./self.n_radial_components),
            [self.n_radial_components])


    def __call__(self, xs):
        single = xs.ndim == 1
        if single:
            xs = jnp.expand_dims(xs, 0)

        assert xs.ndim == 2
        assert xs.shape[1] == self.manifold.D
        n_batch = xs.shape[0]

        betas = nn.softplus(self.betas)
        mus = self.mus / jnp.linalg.norm(self.mus, axis=0, keepdims=True)
        alphas = nn.softmax(self.alphas)

        F = jnp.sum(
            (alphas/betas)*jnp.exp(betas * (jnp.matmul(xs, mus) - 1)),
            axis=-1
        )
        if single:
            F = jnp.squeeze(F, 0)

        return F

class InfAffine(nn.Module):
    n_components: int
    init_alpha_mode: str
    init_alpha_linear_scale: float
    init_alpha_minval: float
    init_alpha_range: float
    manifold: Manifold
    cost_gamma: float
    min_zero_gamma: float

    def setup(self):
        if self.cost_gamma == 'None': self.cost_gamma = None
        if self.min_zero_gamma == 'None': self.min_zero_gamma = None

        if isinstance(self.min_zero_gamma, str):
            self.min_zero_gamma = float(self.min_zero_gamma)

        if isinstance(self.manifold, Product):
            mu_init = densities.get(self.manifold, 'ProductUniformComponents')
        else:
            mu_init = densities.get_uniform(self.manifold)

        self.mus = self.param(
            'mus', init_manifold_samples(mu_init),
            [self.manifold.D, self.n_components])
        if self.init_alpha_mode == 'linear':
            alphas = self.init_alpha_linear_scale*self.mus[:,0].dot(self.mus)
            self.alphas = self.param(
                'alphas', lambda key, shape: alphas,
                [self.n_components])
        elif self.init_alpha_mode == 'uniform':
            self.alphas = self.param(
                'alphas', init_uniform(
                    minval=self.init_alpha_minval,
                    maxval=self.init_alpha_minval+self.init_alpha_range),
                [self.n_components])
        else:
            assert False

    def __call__(self, xs):
        single = xs.ndim == 1
        if single:
            xs = jnp.expand_dims(xs, 0)

        assert xs.ndim == 2
        assert xs.shape[1] == self.manifold.D
        n_batch = xs.shape[0]

        mus = self.manifold.projx(self.mus.T)
        mus = mus.T

        costs = self.manifold.cost(xs, mus) + self.alphas

        if self.cost_gamma is not None and self.cost_gamma > 0.:
            F = self.cost_gamma * logsumexp(
                -costs/self.cost_gamma, axis = 1)
        else:
            F = - jnp.min(costs, 1)

        if self.min_zero_gamma is not None and self.min_zero_gamma > 0.:
            Fz = jnp.stack((F, jnp.zeros_like(F)), axis=-1)
            F = self.min_zero_gamma * logsumexp(
                -Fz/self.min_zero_gamma, axis=-1)

        if single:
            F = jnp.squeeze(F, 0)
        return F


class MultiInfAffine(nn.Module):
    n_layers: int
    n_components: int
    init_alpha_minval: float
    init_alpha_range: float
    manifold: Manifold
    cost_gamma: float
    min_zero_gamma: float

    def setup(self):
        if self.cost_gamma == 'None': self.cost_gamma = None
        if self.min_zero_gamma == 'None': self.min_zero_gamma = None

        mu_init = densities.get_uniform(self.manifold)

        self.mus = []
        self.alphas = []
        self.ws = []
        input_sz = self.manifold.D
        for i in range(self.n_layers):

            key = f'mu{i:02d}'
            mu = self.param(
                key, init_manifold_samples(mu_init),
                [self.manifold.D, self.n_components])
            setattr(self, key, mu)

            key = f'alpha{i:02d}'
            alpha = self.param(
            key, init_uniform(
                minval=self.init_alpha_minval,
                maxval=self.init_alpha_minval+self.init_alpha_range),
            [self.n_components])
            setattr(self, key, alpha)

            key = f'w{i:02d}'
            w = self.param(
                key, init_uniform(minval=0., maxval=1.), [1])
            setattr(self, key, w)

            self.mus.append(mu)
            self.alphas.append(alpha)
            self.ws.append(w)


    def __call__(self, xs):
        single = xs.ndim == 1
        if single:
            xs = jnp.expand_dims(xs, 0)

        assert xs.ndim == 2
        assert xs.shape[1] == self.manifold.D

        F = 0.
        for i, (mu, alpha, w) in enumerate(
                zip(self.mus, self.alphas, self.ws)):

            mu = self.manifold.projx(mu.T)
            mu = mu.T

            costs = self.manifold.cost(xs, mu) + alpha

            w = jnp.exp(-w**2)[0]

            if self.cost_gamma is not None and self.cost_gamma > 0.:
                mincosts =  self.cost_gamma * logsumexp(
                    -costs/self.cost_gamma, axis = 1)
            else:
                mincosts = - jnp.min(costs, 1)

            F = w * nn.relu(F) + (1-w) * mincosts


        if self.min_zero_gamma is not None and self.min_zero_gamma > 0.:
            Fz = jnp.stack((F, jnp.zeros_like(F)), axis=-1)
            F = self.min_zero_gamma * logsumexp(
                -Fz/self.min_zero_gamma, axis=-1)

        if single:
            F = jnp.squeeze(F, 0)
        return F


class ExpMapFlow(nn.Module):
    potential_cfg: omegaconf.dictconfig.DictConfig
    manifold: Manifold

    def setup(self):
        self.potential_mod = hydra.utils.instantiate(
            self.potential_cfg,
            manifold=self.manifold
        )


    def __call__(self, xs, t = 1):
        assert xs.ndim == 2
        n_batch = xs.shape[0]

        def dF_riemannian(xs):
            assert xs.ndim == 1
            dF = jax.jacfwd(self.potential)(xs)
            dF = self.manifold.tangent_projection(xs, dF)
            return dF

        def flow(xs):
            assert xs.ndim == 1
            dF = dF_riemannian(xs)
            z = self.manifold.exponential_map(xs, t * dF)
            return z

        def flow_jacobian(xs):
            assert xs.ndim == 1
            J = jax.jacfwd(flow)(xs)
            return J

        def flow_and_jac(xs):
            z = flow(xs)
            dF = dF_riemannian(xs)
            J = flow_jacobian(xs)
            return z, dF, J

        z, dF, J = jax.vmap(flow_and_jac)(xs)

        E = self.manifold.tangent_orthonormal_basis(xs, dF)
        JE = jnp.matmul(J, E)
        JETJE = jnp.einsum('nji,njk->nik', JE, JE)

        sign, logdet = jnp.linalg.slogdet(JETJE)
        logdet *= 0.5

        return z, logdet, sign

    def potential(self, xs):
        F = self.potential_mod(xs)
        return F


class SequentialFlow(nn.Module):
    n_transforms: int
    manifold: Manifold
    single_transform_cfg: omegaconf.dictconfig.DictConfig

    def setup(self):
        self.transforms = []
        for i in range(self.n_transforms):
            mod = hydra.utils.instantiate(
                self.single_transform_cfg,
                manifold=self.manifold
            )
            self.transforms.append(mod)

            # hack for https://github.com/google/flax/issues/524
            key = f'transform{i:02d}'
            setattr(self, key, mod)

    def __call__(self, orig_xs, debug=False, t = 1):
        ldjs = 0.
        all_xs = []
        all_ldjs = []
        all_ldj_signs = []
        Fs = []

        xs = orig_xs
        for transform in self.transforms:
            xs, ldj, ldj_sign = transform(xs, t = t)
            if debug:
                F = transform.potential(orig_xs)
                all_xs.append(xs)
                all_ldjs.append(ldj)
                all_ldj_signs.append(ldj_sign)
                Fs.append(F)
            ldjs += ldj

        if not debug:
            return xs, ldjs
        else:
            all_xs = jnp.stack(all_xs)
            all_ldjs = jnp.stack(all_ldjs)
            all_ldj_signs = jnp.stack(all_ldj_signs)
            return all_xs, all_ldjs, all_ldj_signs, Fs, ldjs
