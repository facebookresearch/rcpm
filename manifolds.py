# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import jax.numpy as jnp
from jax import random
import jax
from jax.scipy.linalg import block_diag

from spherical_kde import SphericalKDE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
import os
from dataclasses import dataclass
from abc import ABC, abstractmethod

import utils
import cartopy.crs as ccrs

import matplotlib

@dataclass
class Manifold(ABC):
    D: int # Dimension of the ambient Euclidean space

    @abstractmethod
    def exponential_map(self, x, v):
        pass

    @abstractmethod
    def tangent_projection(self, x, v):
        pass

    @abstractmethod
    def projx(self, x):
        pass

    # @abstractmethod
    # def dist(self, x, y):
    #     pass

    @abstractmethod
    def cost(self, x, y):
        pass

    @abstractmethod
    def tangent_orthonormal_basis(self, x, dF):
        pass


eps = 1e-5 # TODO: Other stabilization?
divsin = lambda x: x / jnp.sin(x)
sindiv = lambda x: jnp.sin(x) / (x + eps)
divsinh = lambda x: x / jnp.sinh(x)
sinhdiv = lambda x: jnp.sinh(x) / (x + eps)

def lorentz_cross(x, y):
    z = jnp.cross(x, y)
    z = z.at[...,0].set(-z[...,0])
    return z

@dataclass
class Sphere(Manifold):
    jitter: float = 1e-2

    NUM_POINTS = 100

    theta = jnp.linspace(0, 2 * np.pi, 2 * NUM_POINTS)
    phi = jnp.linspace(0, np.pi, NUM_POINTS)
    tp = jnp.array(np.meshgrid(theta, phi, indexing='ij'))
    tp = tp.transpose([1, 2, 0]).reshape(-1, 2)

    def exponential_map(self, x, v):
        v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
        return x * jnp.cos(v_norm) + v * sindiv(v_norm)

    def log(self, x, y):
        xy = (x * y).sum(axis=-1, keepdims=True)
        xy = jnp.clip(xy, a_min=-1 + 1e-6, a_max=1 - 1e-6)
        val = jnp.arccos(xy)
        return divsin(val) * (y - xy * x)

    def tangent_projection(self, x, u):
        proj_u = u - x*x.dot(u)
        return proj_u

    def tangent_orthonormal_basis(self, x, dF):
        assert x.ndim == 2

        if x.shape[1] == 2:
            E = x[:, jnp.array([1,0])] * jnp.array([-1., 1.])
            E = E.reshape(*E.shape, 1)
        elif x.shape[1] == 3:
            # The potential's Riemannian derivative dF is on the
            # tangent space, so on S2 we normalize this and
            # find the only remaining orthogonal direction.
            norm_v = dF / jnp.linalg.norm(dF, axis=-1, keepdims=True)
            E = jnp.dstack([norm_v, jnp.cross(x, norm_v)])
        else:
            raise NotImplementedError()

        return E

    def dist(self, x, y):
        inner = jnp.matmul(x, y)
        inner = inner/(1 + self.jitter)
        return jnp.arccos(inner)

    def cost(self, x, y):
        return self.dist(x, y)**2 / 2.

    def projx(self, x):
        x /= jnp.linalg.norm(x, axis=-1, keepdims=True)
        return x

    def transp(self, x, y, u):
        yu = jnp.sum(y * u, axis=-1, keepdims=True)
        xy = jnp.sum(x * y, axis=-1, keepdims=True)
        return u - yu/(1 + xy) * (x + y)

    def logdetexp(self, x, u):
        norm_u = jnp.linalg.norm(u, axis=-1)
        val = jnp.log(jnp.abs(sindiv(norm_u)))
        return (u.shape[-1]-2) * val


    def zero(self):
        y = jnp.zeros(self.D)
        y = y.at[...,0].set(-1.)
        return y

    def zero_like(self, x):
        y = jnp.zeros_like(x)
        y = y.at[...,0].set(-1.)
        return y

    def squeeze_tangent(self, x):
        return x[..., 1:]

    def unsqueeze_tangent(self, x):
        return jnp.concatenate((jnp.zeros_like(x[..., :1]), x), axis=-1)

    def plot_samples(self, model_samples, kde_factor=0.1, save='t.png'):
        spherical_samples = utils.euclidean_to_spherical(model_samples)
        kde = SphericalKDE(
            spherical_samples[:,0], spherical_samples[:,1], bandwidth=kde_factor)
        heatmap = np.exp(kde(self.tp[:,0], self.tp[:,1]).reshape(
            2 * self.NUM_POINTS, self.NUM_POINTS))
        self.plot_mollweide(heatmap, save=save)

    def plot_density(self, log_prob_fn, save='t.png'):
        density = log_prob_fn(utils.spherical_to_euclidean(self.tp))
        density = jnp.exp(density)
        heatmap = density.reshape(2 * self.NUM_POINTS, self.NUM_POINTS)
        self.plot_mollweide(heatmap, save=save)

    def plot_mollweide(self, heatmap, save):
        tt, pp = np.meshgrid(
            self.theta - np.pi, self.phi - np.pi / 2, indexing='ij')

        proj = ccrs.Mollweide()
        fig = plt.figure(figsize=(3,2), dpi=200)
        ax = fig.add_subplot(111, projection='mollweide')
        norm = matplotlib.colors.Normalize()
        ax.pcolormesh(tt, pp, heatmap, cmap='magma', norm = norm)
        ax.set_axis_off()
        plt.savefig(save)
        os.system(f"convert {save} -trim {save} &")
        plt.close(fig)



class Euclidean(Manifold):
    def exponential_map(self, x, v):
        return x + v

    def tangent_projection(self, x, u):
        return u

    def cost(self, x, y):
        return 0.5 * self.dist(x,y)**2

    def dist(self, x, y):
        return - jnp.matmul(x, y)

    def tangent_orthonormal_basis(self, x, dF):
        tang_vecs = [jnp.eye(x.shape[1]) for i in range(x.shape[0])]
        return jnp.stack(tang_vecs, 0)



def get(manifold):
    if manifold == 'S1':
        return Sphere(D = 2)
    elif manifold == 'S2':
        return Sphere(D = 3)
    elif manifolds == 'R':
        return Euclidean(D = 1)
    else:
        assert False

@dataclass
class Product(Manifold):
    manifolds_str: str = 'S1,S1'

    def __post_init__(self):
        self.manifolds = []
        for man in self.manifolds_str.split(','):
            self.manifolds.append(get(man))

    def exponential_map(self, x, v):
        exp_prod = []
        d = 0
        for man in self.manifolds:
            exp_man = man.exponential_map(x[d:d+man.D], v[d:d+man.D])
            exp_prod.append(exp_man)
            d = d + man.D
        exp_prod = jnp.concatenate(exp_prod)
        return exp_prod

    def tangent_projection(self, x, u):
        proj_prod = []
        d = 0
        for man in self.manifolds:
            proj_man = man.tangent_projection(x[d:d+man.D], u[d:d+man.D])
            proj_prod.append(proj_man)
            d = d + man.D
        proj_prod = jnp.concatenate(proj_prod)
        return proj_prod

    def cost(self, x, y):
        cost_prod = jnp.zeros([x.shape[0], y.T.shape[0]])
        d = 0
        for man in self.manifolds:
            cost_prod += man.cost(x[:,d:d+man.D], y[d:d+man.D,:])
            d = d + man.D
        return cost_prod

    def dist(self, x, y):
        pass

    def tangent_orthonormal_basis(self, x, dF):
        d = 0
        map_block_diag = jax.vmap(block_diag)
        blocks = []
        for man in self.manifolds:
            onb_man = man.tangent_orthonormal_basis(x[:,d:d+man.D], dF[:,d:d+man.D])
            blocks.append(onb_man)
            d = d + man.D
        onb = map_block_diag(*(blocks))
        return onb

    def projx(self, x):
        x_proj = []
        d = 0
        for man in self.manifolds:
            x_proj_man = man.projx(x[:,d:d+man.D])
            d = d + man.D
            x_proj.append(x_proj_man)
        x_proj = jnp.concatenate(x_proj, 1)
        return x_proj

    def plot_samples(self, model_samples, save='t.png'):
        pass

    def plot_density(self, log_prob_fn, save='t.png'):
        pass



@dataclass
class Torus(Product):
    manifolds: str = 'S1,S1'

    NUM_POINTS = 160

    theta = jnp.linspace(0, 2 * np.pi, 2 * NUM_POINTS)
    phi = jnp.linspace(0, 2 * np.pi, NUM_POINTS)
    tp = jnp.array(np.meshgrid(theta, phi, indexing='ij'))
    tp = tp.transpose([1, 2, 0]).reshape(-1, 2)

    def plot_samples(self, model_samples, save='t.png'):
        theta1 = utils.S1euclideantospherical(model_samples[:,:2])
        theta2 = utils.S1euclideantospherical(model_samples[:,2:])

        x, y, z = utils.productS1toTorus(theta1, theta2)
        data = jnp.stack((x, y, z), 1)
        estimated_density = gaussian_kde(
                data.T, 0.2)

        x_grid, y_grid, z_grid = utils.productS1toTorus(self.tp[:,0], self.tp[:,1])
        grid = jnp.stack((x_grid, y_grid, z_grid), 1)
        probas_grid = estimated_density(grid.T)

        fig = plt.figure()
        ax = Axes3D(fig)
        #TODO: fix this - I negate become the mode is at the bottom of the torus in unimodal density
        ax.scatter(-x_grid, -y_grid, -z_grid, alpha = 0.2, c = probas_grid)
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_zlim(-1,1)
        plt.axis('off')
        plt.savefig(save)


    def plot_density(self, log_prob_fn, save='t.png'):
        euc1 = jnp.stack((jnp.cos(self.tp[:,0]), jnp.sin(self.tp[:,0])),1)
        euc2 = jnp.stack((jnp.cos(self.tp[:,1]), jnp.sin(self.tp[:,1])),1)
        prod_euc = jnp.concatenate((euc1,euc2),1)

        density = log_prob_fn(prod_euc)
        density = jnp.exp(density)

        x_grid, y_grid, z_grid = utils.productS1toTorus(self.tp[:,0], self.tp[:,1])
        grid = jnp.stack((x_grid, y_grid, z_grid), 1)

        fig = plt.figure()
        plt.savefig(save)
        ax = Axes3D(fig)
        #TODO: fix this - I negate become the mode is at the bottom of the torus in unimodal density
        ax.scatter(-x_grid, -y_grid, -z_grid, alpha = 0.2, c = density)
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_zlim(-1,1)
        plt.axis('off')

        plt.savefig(save)


@dataclass
class InfCylinder(Product):
    manifolds: str = 'S1,R'
