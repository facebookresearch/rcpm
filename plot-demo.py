#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import jax.numpy as jnp
import numpy as np

import argparse
import os
import sys
import pickle as pkl
import shutil
from omegaconf import OmegaConf
from collections import namedtuple

from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt
plt.style.use('bmh')
from matplotlib import cm
from matplotlib.collections import LineCollection


import utils

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(
    mode='Verbose', color_scheme='Linux', call_pdb=1)

NUM_POINTS = 100

theta = jnp.linspace(0, 2 * jnp.pi, 2 * NUM_POINTS)
phi = jnp.linspace(0, jnp.pi, NUM_POINTS)
tp = jnp.array(np.meshgrid(theta, phi, indexing='ij'))
tp = tp.transpose([1, 2, 0]).reshape(-1, 2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_root', type=str)
    args = parser.parse_args()

    fname = f"{args.exp_root}/latest.pkl"
    with open(fname, 'rb') as f:
        W = pkl.load(f)

    nrows, ncols = 1, 1
    fig, ax = plt.subplots(
        nrows, ncols, figsize=(6*ncols, 4*nrows),
        subplot_kw={'projection': 'mollweide'}
    )

    all_xs, _, _, Fs, _ = W.flow.apply(
        W.optimizer.target, utils.spherical_to_euclidean(tp), debug=True)
    plot_heatmap(Fs[0].reshape(2*NUM_POINTS, NUM_POINTS), ax)

    fname = f"{args.exp_root}/potential.png"
    print(f'Saving to {fname}')
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(fname)
    os.system(f"convert {fname} -trim {fname}")


    nrows, ncols = 1, 1
    fig, ax = plt.subplots(
        nrows, ncols, figsize=(6*ncols, 4*nrows),
        subplot_kw={'projection': 'mollweide'}
    )

    def plot_grid(x,y, ax=None, **kwargs):
        ax = ax or plt.gca()
        segs1 = np.stack((x,y), axis=2)
        segs2 = segs1.transpose(1,0,2)
        ax.add_collection(LineCollection(segs1, **kwargs))
        ax.add_collection(LineCollection(segs2, **kwargs))

    b = 0.2
    lw = 0.5
    grid_x, grid_y = np.meshgrid(
        np.linspace(-np.pi+b, np.pi-b, 50),
        np.linspace(-np.pi+b, np.pi-b, 50))
    plot_grid(grid_x, grid_y, ax, color='lightgrey', lw=lw)

    grid_sphere = utils.spherical_to_euclidean(
        jnp.stack((grid_x+np.pi, (grid_y+np.pi)/2.)).reshape(2, -1).T
    )
    F_grid_sphere, _ = W.flow.apply(W.optimizer.target, grid_sphere)
    F_grid = utils.euclidean_to_spherical(F_grid_sphere)
    F_grid_x = F_grid[:,0].reshape(grid_x.shape) - np.pi
    F_grid_y = F_grid[:,1].reshape(grid_x.shape)*2. - np.pi
    plot_grid(F_grid_x, F_grid_y, color='C0', lw=lw)

    ax.set_axis_off()
    fname = f"{args.exp_root}/grid.png"
    print(f'Saving to {fname}')
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(fname)
    os.system(f"convert {fname} -trim {fname}")

def plot_density(xs, ax):
    estimated_density = gaussian_kde(
        utils.euclidean_to_spherical(xs).T, 0.2)
    heatmap = estimated_density(tp.T).reshape(2 * NUM_POINTS, NUM_POINTS)
    plot_heatmap(heatmap, ax)

def plot_heatmap(fs, ax):
    tt, pp = jnp.meshgrid(theta - jnp.pi, phi - jnp.pi / 2, indexing='ij')
    ax.pcolormesh(tt, pp, fs, cmap=plt.cm.magma)
    ax.set_axis_off()

if __name__ == '__main__':
    main()
