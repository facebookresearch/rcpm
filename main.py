#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(
    mode='Plain', color_scheme='Neutral', call_pdb=1)

import numpy as np

import jax
import jax.numpy as jnp
from jax.config import config; config.update("jax_enable_x64", True)


import pickle as pkl

from flax import linen as nn
from flax import optim

import time

import hydra

import csv
import os

import functools

import flows
import utils
import densities

from setproctitle import setproctitle
setproctitle('iccnn')


def kl_ess(log_model_prob, log_target_prob):
    weights = jnp.exp(log_target_prob) / jnp.exp(log_model_prob)
    Z = jnp.mean(weights)
    KL = jnp.mean(log_model_prob - log_target_prob) + jnp.log(Z)
    ESS = jnp.sum(weights) ** 2 / jnp.sum(weights ** 2)
    return Z, KL, ESS


class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg

        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.manifold = hydra.utils.instantiate(self.cfg.manifold)
        self.base = densities.get(self.manifold, self.cfg.base)
        self.target = densities.get(self.manifold, self.cfg.target)

        self.key = jax.random.PRNGKey(self.cfg.seed)

        self.flow = hydra.utils.instantiate(
            self.cfg.flow, manifold=self.manifold)
        self.key, k1, k2, k3, k4, k5 = jax.random.split(self.key, 6)
        batch = self.base.sample(k1, self.cfg.batch_size)
        init_params = self.flow.init(k2, batch)

        self.base_samples = self.base.sample(k3, self.cfg.eval_samples)
        self.base_log_probs = self.base.log_prob(self.base_samples)
        if self.cfg.loss == 'likelihood':
            self.eval_target_samples = self.target.sample(
                k5, self.cfg.eval_samples)

        optimizer_def = hydra.utils.instantiate(self.cfg.optim)
        self.optimizer = optimizer_def.create(init_params)

        self.iter = 0

    def run(self):
        if self.cfg.loss == 'kl':
            self.train_kl()
        elif self.cfg.loss == 'likelihood':
            self.train_likelihood()
        else:
            assert False

    def train_kl(self):
        @jax.jit
        def loss(params, base_samples, base_log_probs):
            z, ldjs = self.flow.apply(params, base_samples)
            loss =  (base_log_probs - ldjs -
                     self.target.log_prob(z)).mean()
            return loss

        @jax.jit
        def update(optimizer, base_samples, base_log_probs):
            l, grads = jax.value_and_grad(loss)(
                optimizer.target, base_samples, base_log_probs)
            optimizer = optimizer.apply_gradient(grads)
            return l, optimizer

        logf, writer = self._init_logging()

        times = []
        if self.iter == 0:
            model_samples, ldjs = self.flow.apply(
                self.optimizer.target, self.base_samples)
            self.manifold.plot_samples(
                model_samples, save=f'{self.iter:06d}.png')

            self.manifold.plot_density(self.target.log_prob, 'target.png')

        while self.iter < self.cfg.iterations:
            start = time.time()
            self.key, subkey = jax.random.split(self.key)
            base_samples = self.base.sample(subkey, self.cfg.batch_size)
            base_log_probs = self.base.log_prob(base_samples)
            l, self.optimizer = update(
                self.optimizer, base_samples, base_log_probs)

            times.append(time.time() - start)
            self.iter += 1
            if self.iter % self.cfg.log_frequency == 0:
                l = loss(self.optimizer.target,
                         self.base_samples, self.base_log_probs)

                model_samples, ldjs = self.flow.apply(
                    self.optimizer.target, self.base_samples)
                self.manifold.plot_samples(
                    model_samples, save=f'{self.iter:06d}.png')
                if not self.cfg.disable_evol_plots:
                    for i, t in enumerate(jnp.linspace(0.1,1,11)):
                        model_samples, ldjs = self.flow.apply(
                        self.optimizer.target, self.base_samples, t = t)
                        self.manifold.plot_samples(
                            model_samples,
                            save=f'{self.iter:06d}_{i}.png')


                log_prob = self.base_log_probs - ldjs
                _,  kl, ess = kl_ess(
                    log_prob, self.target.log_prob(model_samples))
                ess = ess / self.cfg.eval_samples * 100
                msg = "Iter {} | Loss {:.3f} | KL {:.3f} | ESS {:.2f}% | {:.2e}s/it"
                print(msg.format(
                    self.iter, l, kl, ess, jnp.mean(jnp.array(times))))
                writer.writerow({
                    'iter': self.iter, 'loss': l, 'kl': kl, 'ess': ess
                })
                logf.flush()
                self.save('latest')

                times = []


    def train_likelihood(self):
        @jax.jit
        def logprob(params, target_samples, t = 1):
            zs, ldjs = self.flow.apply(params, target_samples, t = t)
            log_prob = ldjs + self.base.log_prob(zs)
            return log_prob

        @jax.jit
        def loss(params, target_samples):
            return -logprob(params, target_samples).mean()

        @jax.jit
        def update(optimizer, target_samples):
            l, grads = jax.value_and_grad(loss)(
                optimizer.target, target_samples)
            optimizer = optimizer.apply_gradient(grads)
            return l, optimizer

        target_sample_jit = jax.jit(self.target.sample, static_argnums=(1,))
        base_sample_jit = jax.jit(self.base.sample, static_argnums=(1,))

        logf, writer = self._init_logging()

        times = []

        if self.iter == 0 and not self.cfg.disable_init_plots:
            model_samples, ldjs = self.flow.apply(
                self.optimizer.target, self.eval_target_samples)
            try:
                self.manifold.plot_density(
                 self.target.log_prob, save=f'target_density.png')
            except:
                pass
            self.manifold.plot_samples(
                self.eval_target_samples, save=f'target_samples.png')
            self.manifold.plot_samples(
                base_sample_jit(self.key, self.cfg.eval_samples),
                save=f'base_samples.png')
            self.manifold.plot_density(
                self.base.log_prob, save=f'base_density.png')
            self.manifold.plot_samples(
                model_samples, save=f'samples_{self.iter:06d}.png')
            self.manifold.plot_density(
                functools.partial(logprob, self.optimizer.target),
                save=f'density_{self.iter:06d}.png')
            if not self.cfg.disable_evol_plots:
                for i, t in enumerate(jnp.linspace(0.1,1,11)):
                    self.manifold.plot_density(
                        functools.partial(logprob, self.optimizer.target, t = t),
                        save=f'density_{self.iter:06d}_{i}.png')



        while self.iter < self.cfg.iterations:
            start = time.time()
            self.key, subkey = jax.random.split(self.key)
            target_samples = target_sample_jit(subkey, self.cfg.batch_size)
            l, self.optimizer = update(self.optimizer, target_samples)

            times.append(time.time() - start)
            self.iter += 1
            if self.iter % self.cfg.log_frequency == 0:
                l = loss(self.optimizer.target, self.eval_target_samples)
                model_samples, ldjs = self.flow.apply(
                    self.optimizer.target, self.eval_target_samples)
                self.manifold.plot_samples(
                    model_samples, save=f'samples_{self.iter:06d}.png')
                self.manifold.plot_density(
                    functools.partial(logprob, self.optimizer.target),
                    save=f'density_{self.iter:06d}.png')
                if not self.cfg.disable_evol_plots:
                    for i, t in enumerate(jnp.linspace(0.1,1,10)):
                        self.manifold.plot_density(
                            functools.partial(logprob, self.optimizer.target, t = t),
                            save=f'density_{self.iter:06d}_{i}.png')



                msg = "Iter {} | Loss {:.3f} | {:.2e}s/it"
                print(msg.format(
                    self.iter, l, jnp.mean(jnp.array(times))))
                writer.writerow({
                    'iter': self.iter, 'loss': l,
                })
                logf.flush()
                self.save('latest')
                times = []



    def save(self, tag='latest'):
        path = os.path.join(self.work_dir, f'{tag}.pkl')
        with open(path, 'wb') as f:
            pkl.dump(self, f)


    def _init_logging(self):
        logf = open('log.csv', 'a')
        fieldnames = ['iter', 'loss', 'kl', 'ess']
        writer = csv.DictWriter(logf, fieldnames=fieldnames)
        if os.stat('log.csv').st_size == 0:
            writer.writeheader()
            logf.flush()
        return logf, writer


# Import like this for pickling
from main import Workspace as W

@hydra.main(config_name='config')
def main(cfg):
    fname = os.getcwd() + '/latest.pt'
    if os.path.exists(fname):
        print(f'Resuming fom {fname}')
        with open(fname, 'rb') as f:
            workspace = pkl.load(f)
    else:
        workspace = W(cfg)

    workspace.run()

if __name__ == '__main__':
    main()
