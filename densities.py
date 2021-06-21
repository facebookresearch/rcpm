# Copyright (c) Facebook, Inc. and its affiliates.

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import norm

import numpy as np

from functools import partial
import sys

from dataclasses import dataclass
from abc import ABC, abstractmethod
from manifolds import Manifold, Sphere

import utils
import pickle

from scipy.stats import gaussian_kde
from jax.numpy import newaxis

def get(manifold, name):
    if name == 'SphereBaseWrappedNormal':
        assert isinstance(manifold, Sphere)
        loc = manifold.zero()
        scale = jnp.full(manifold.D-1, .3)
        return WrappedNormal(manifold=manifold, loc=loc, scale=scale)
    elif name == 'LouSphereSingleMode':
        assert isinstance(manifold, Sphere)
        loc = manifold.projx(-jnp.ones(manifold.D))
        scale = jnp.full(manifold.D-1, .3)
        return WrappedNormal(manifold=manifold, loc=loc, scale=scale)
    elif 'Earth' in name:
            try:
                name, year = name.split('_')
                return getattr(sys.modules[__name__], name)(manifold=manifold, year = year)
            except:
                print(f"Error loading data class {name}")
                raise
    else:
        try:
            return getattr(sys.modules[__name__], name)(manifold=manifold)
        except:
            print(f"Error loading data class {name}")
            raise

def get_uniform(manifold):
    if isinstance(manifold, Sphere):
        return get(manifold, 'SphereUniform')
    else:
        assert False

@dataclass
class Density(ABC):
    manifold: Manifold

    @abstractmethod
    def log_prob(self, x):
        pass

    @abstractmethod
    def sample(self, key, n_samples):
        pass

    def __hash__(self): return 0 # For jitting


@dataclass
class Earth(Density):
    year: int
    def __post_init__(self):
        self.data = pickle.load(open('../../../data/earth_data_sphere_' + self.year + '.pkl','rb'))
        self.data = self.data[self.data[:,-1]>-0.8]
        self.data = jnp.array(self.data)[::5]
        self.data = jax.ops.index_update(self.data, jax.ops.index[:, -1], -self.data[:, -1])
        self.dens = dens = gaussian_kde(self.data.T, 0.1)
        L = jnp.linalg.cholesky(self.dens.covariance*2*jnp.pi)
        self.log_det = 2*jnp.log(jnp.diag(L)).sum()
        self.inv_cov = jnp.array(self.dens.inv_cov)
        self.weights = jnp.array(self.dens.weights)

        #Can't pickle kde object
        self.dens = 0.
    def log_prob(self,xs):
        def fun(point):
            diff = self.data.T - point
            tdiff = jnp.dot(self.inv_cov, diff)
            energy = jnp.sum(diff * tdiff, axis=0)
            log_to_sum = 2.0 * jnp.log(self.weights) - self.log_det - energy
            result = jax.scipy.special.logsumexp(0.5 * log_to_sum)
            return result
        fun_map = jax.vmap(fun)
        return fun_map(xs[:,:,newaxis])

    def sample(self, key, n_samples):
        key, k1 = jax.random.split(key, 2)
        indexes = jax.random.randint(k1,  [n_samples], 0, self.data.shape[0]-1)
        return self.data[indexes]




class SphereUniform(Density):
    def log_prob(self, xs):
        # TODO, support other spheres
        assert xs.ndim == 2
        n_batch, D = xs.shape
        assert D == self.manifold.D

        if self.manifold.D == 2:
            SA = 2.*jnp.pi
        elif self.manifold.D == 3:
            SA = 4.*jnp.pi
        else:
            raise NotImplementedError()

        return jnp.full([n_batch], jnp.log(1. / SA))

    def sample(self, key, n_samples):
        xs = random.normal(key, shape=[n_samples, self.manifold.D])
        return self.manifold.projx(xs)


@dataclass
class WrappedNormal(Density):
    loc: jnp.ndarray
    scale: jnp.ndarray

    def log_prob(self, z):
        u = self.manifold.log(self.loc, z)
        y = self.manifold.zero_like(self.loc)
        v = self.manifold.transp(self.loc, y, u)
        v = self.manifold.squeeze_tangent(v)
        n_logprob = norm.logpdf(v, scale=self.scale).sum(axis=-1)
        logdet = self.manifold.logdetexp(self.loc, u)
        assert n_logprob.shape == logdet.shape
        log_prob = n_logprob - logdet
        return log_prob

    def sample(self, key, n_samples):
        v = self.scale * random.normal(key, [n_samples, self.manifold.D-1])
        v = self.manifold.unsqueeze_tangent(v)
        x = self.manifold.zero_like(self.loc)
        u = self.manifold.transp(x, self.loc, v)
        z = self.manifold.exponential_map(self.loc, u)
        return z

    def __hash__(self): return 0 # For jitting

@dataclass
class SphereDemo(Density):
    def __post_init__(self):
        self.modes = []
        locs = [
            jnp.array([0.3, 1., 1.]),
            jnp.array([0.3, -1., 1.]),
            jnp.array([0.3, 1., -1.]),
            jnp.array([0.3, -1., -1.]),
        ]
        locs = [self.manifold.projx(loc) for loc in locs]
        scale = jnp.full(self.manifold.D-1, .3)
        self.dists = [
            WrappedNormal(manifold=self.manifold, loc=loc, scale=scale)
            for loc in locs
        ]

    def log_prob(self, z):
        raise NotImplementedError()

    def sample(self, key, n_samples):
        keys = random.split(key, len(self.dists))
        n = int(np.ceil(n_samples/len(self.dists)))
        samples = jnp.concatenate([
            d.sample(key, n) for key, d in zip(keys, self.dists)
        ], axis=0)
        samples = random.permutation(key, samples)
        return samples[:n_samples]

    def __hash__(self): return 0 # For jitting

@dataclass
class LouSphereFourModes(Density):
    def __post_init__(self):
        self.modes = []
        one = jnp.ones(3)
        oned = jnp.ones(3)
        oned = jax.ops.index_update(oned, jax.ops.index[2], -1.)
        locs = [one, -one, oned, -oned]
        locs = [self.manifold.projx(loc) for loc in locs]
        scale = jnp.full(self.manifold.D-1, .3)
        self.dists = [
            WrappedNormal(manifold=self.manifold, loc=loc, scale=scale)
            for loc in locs
        ]

    def log_prob(self, z):
        raise NotImplementedError()

    def sample(self, key, n_samples):
        keys = random.split(key, len(self.dists))
        n = int(np.ceil(n_samples/len(self.dists)))
        samples = jnp.concatenate([
            d.sample(key, n) for key, d in zip(keys, self.dists)
        ], axis=0)
        samples = random.permutation(key, samples)
        return samples[:n_samples]

    def __hash__(self): return 0 # For jitting


class RezendeSphereFourMode(Density):
    # https://github.com/katalinic/sdflows/blob/master/optimisation.py#L12
    target_mu = utils.spherical_to_euclidean(jnp.array([
        [1.5, 0.7 + jnp.pi / 2],
        [1., -1. + jnp.pi / 2],
        [5., 0.6 + jnp.pi / 2],
        [4., -0.7 + jnp.pi / 2]
    ]))

    def log_prob(self, x):
        # TODO: This is unnormalized
        assert x.ndim == 2
        return jnp.log(jnp.sum(jnp.exp(10. * x.dot(self.target_mu.T)), axis=1))

    def sample(self, key, n_samples):
        raise NotImplementedError()

class RezendeTorusUnimodal(Density):
    psi = [4.18, 5.96]

    def log_prob(self, x):
        assert x.ndim == 2

        theta1, theta2 = utils.S1euclideantospherical(x[:,:2]), utils.S1euclideantospherical(x[:,2:])

        return jnp.log(jnp.exp(jnp.cos(theta1-self.psi[0]) + jnp.cos(theta2-self.psi[1])))

    def sample(self, key, n_samples):
        raise NotImplementedError()

class RezendeCorrelated(Density):
    psi = 1.94

    def log_prob(self, x):
        assert x.ndim == 2

        theta1, theta2 = utils.S1euclideantospherical(x[:,:2]), utils.S1euclideantospherical(x[:,2:])

        return jnp.log(jnp.exp(jnp.cos(theta1 + theta2 - self.psi)))

    def sample(self, key, n_samples):
        raise NotImplementedError()

class SphereCheckerboard(Density):
    def log_prob(self, x):
        # TODO: Could be optimized
        # TODO: Assumes x is uniformly distributed

        lonlat = utils.euclidean_to_spherical(x)
        s = jnp.pi/2-.2 # long side length

        def in_board(z, s):
            # z is lonlat
            lon = z[0]
            lat = z[1]

            if np.pi <= lon < np.pi+s or np.pi-2*s <= lon < np.pi-s:
                v = np.pi/2 <= lat < np.pi/2+s/2 or \
                    np.pi/2-s <= lat < np.pi/2-s/2
            elif np.pi-2*s <= lon < np.pi+2*s:
                v = np.pi/2+s/2 <= lat < np.pi/2+s or \
                    np.pi/2-s/2 <= lat < np.pi/2
            else:
                v = 0.

            v = float(v)
            return v

        probs = []
        for i in range(lonlat.shape[0]):
            probs.append(in_board(lonlat[i,:], s))
        probs = jnp.stack(probs)
        probs /= jnp.sum(probs)
        probs = jnp.log(probs)
        return probs

    def sample(self, key, n_samples):
        s = jnp.pi/2.-.2 # long side length
        offsets = jnp.array([
            (0,0), (s, s/2), (s, -s/2), (0, -s), (-s, s/2),
            (-s, -s/2), (-2*s, 0), (-2*s, -s)])

        # (x,y) ~ uniform([pi,pi + s] times [pi/2, pi/2 + s/2])
        k1, k2, k3 = jax.random.split(key, 3)
        x1 = random.uniform(k1, [n_samples]) * s + jnp.pi
        x2 = random.uniform(k1, [n_samples]) * s + jnp.pi
        x2 = random.uniform(k2, [n_samples]) * s/2. + jnp.pi/2.

        samples = jnp.stack([x1, x2], axis=1)
        off = offsets[random.randint(
            k3, [n_samples], minval=0, maxval=len(offsets))]

        samples += off

        samples = utils.spherical_to_euclidean(samples)
        return samples


@dataclass
class ProductUniformComponents(Density):
    def __post_init__(self):
        self.base_dists = []
        for man in self.manifold.manifolds:
            self.base_dists.append(get_uniform(man))

    def log_prob(self, xs):
        #Note this is not necessarily uniform
        assert xs.ndim == 2
        n_batch = xs.shape[0]
        log_probas = jnp.zeros([n_batch])
        d = 0
        for i, base_dist in enumerate(self.base_dists):
            D = self.manifold.manifolds[i].D
            log_probas += base_dist.log_prob(xs[:, d:d+D])
            d = d + D
        return log_probas

    def sample(self, key, n_samples):
        #Note this is not necessarily uniform
        xs = []
        keys = jax.random.split(key, len(self.base_dists))
        for key, base_dist in zip(keys, self.base_dists):
            samples_man = base_dist.sample(key = key, n_samples = n_samples)
            xs.append(samples_man)
        xs = jnp.concatenate(xs, 1)
        return xs

    def __hash__(self): return 0 # For jitting
