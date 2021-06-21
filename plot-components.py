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

import utils

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(
    mode='Verbose', color_scheme='Linux', call_pdb=1)

NUM_POINTS = 150

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

    n_transforms = W.cfg.flow.n_transforms
    nrows, ncols = n_transforms+1, 3
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(6*ncols, 4*nrows),
        subplot_kw={'projection': 'mollweide'}
    )
    if nrows == 1:
        axs = np.expand_dims(axs, 0)
    # if ncols == 1:
    #     axs = np.expand_dims(axs, -1)

    axs[0,0].set_title('Potential', fontsize=20)
    axs[0,1].set_title('LDJ', fontsize=20)
    axs[0,2].set_title('Distribution', fontsize=20)

    all_xs, all_ldjs, all_ldj_signs, Fs, ldjs = W.flow.apply(
        W.optimizer.target, utils.spherical_to_euclidean(tp), debug=True)
    all_ldjs = jnp.stack(all_ldjs)
    Fs = jnp.stack(Fs)
    ldj_bounds = (jnp.min(all_ldjs), jnp.max(all_ldjs))
    F_bounds = (jnp.min(Fs), jnp.max(Fs))

    for t in range(n_transforms):
        plot_heatmap(Fs[t].reshape(2*NUM_POINTS, NUM_POINTS), axs[t,0],
                     vbounds=F_bounds)
        plot_heatmap(all_ldjs[t].reshape(2*NUM_POINTS, NUM_POINTS),
                     axs[t,1], vbounds=ldj_bounds)
        plot_density(all_xs[t], axs[t,2])

    axs[-1,0].set_axis_off()
    axs[-1,2].set_axis_off()
    axs[-1,1].set_title('Cumulative LDJ', fontsize=20)
    plot_heatmap(ldjs.reshape(2*NUM_POINTS, NUM_POINTS), axs[-1,1])

    fname = f"{args.exp_root}/components.png"
    print(f'Saving to {fname}')
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(fname)
    os.system(f"convert {fname} -trim {fname}")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4),
        subplot_kw={'projection': 'mollweide'})

    plot_heatmap(ldjs.reshape(2*NUM_POINTS, NUM_POINTS), ax)
    fname = f"{args.exp_root}/ldj.png"
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


def plot_heatmap(fs, ax, cmap=plt.cm.magma, vbounds=None):
    tt, pp = jnp.meshgrid(theta - jnp.pi, phi - jnp.pi / 2, indexing='ij')
    vmin = vmax = None
    if vbounds is not None:
        vmin, vmax = vbounds
    ax.pcolormesh(tt, pp, fs, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_axis_off()

if __name__ == '__main__':
    main()
