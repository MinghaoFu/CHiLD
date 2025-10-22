"""Temporal VAE with gaussian margial and laplacian transition prior"""

import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.distributions as D
from torch.nn import functional as F
from .components.beta import BetaVAE_MLP
from .components.transition import (NPChangeInstantaneousTransitionPrior, NPInstantaneousTransitionPrior)
from .components.mlp import MLPEncoder, MLPDecoder, Inference
from .metrics.correlation import compute_mcc
from .metrics.block import nonlinear_disentanglement, compute_r2
import matplotlib.pyplot as plt
import seaborn as sns
import ipdb as pdb

import wandb


class InstantaneousProcess(pl.LightningModule):
    def __init__(
        self, 
        input_dim,
        z_dim,
        z_dim_fix, 
        z_dim_change,
        lag,
        nclass,
        hidden_dim=128,
        embedding_dim=8,
        lr=1e-4,
        beta=0.0025,
        gamma=0.0075,
        theta=0.2,
        decoder_dist='gaussian',
        correlation='Pearson',
        n_mea=5,
        enable_flexible_sparsity=False,
        w_hist=None,
        w_inst=None,
        z_dim_list=[1, 2, 4],
        optimizer_type='adam'):
        '''Nonlinear ICA for time-varing causal processes with instantaneous causal effects'''
        super().__init__()
        self.z_dim = sum(z_dim_list)
        self.z_dim_fix = z_dim_fix
        self.z_dim_change = z_dim_change
        assert (self.z_dim == self.z_dim_fix + self.z_dim_change)
        self.lag = lag
        self.input_dim = input_dim
        self.lr = lr
        self.beta = beta
        self.gamma = gamma
        self.theta = theta
        self.correlation = correlation
        self.decoder_dist = decoder_dist
        self.optimizer_type = optimizer_type
        self.nclass = nclass
        self.n_mea = n_mea
        self.z_dim_list = z_dim_list
        self.hz_to_z = True # for compute r2, hard code
        self.best_mcc = 0
        
        # Domain embeddings (dynamics)
        self.embed_func = nn.Embedding(nclass, embedding_dim)
        # Recurrent/Factorized inference
        self.net = BetaVAE_MLP(input_dim=input_dim, 
                                hidden_dim=hidden_dim,
                                extra_obs=n_mea,
                                z_dim_list=z_dim_list)

        self.transition_prior_fix = NPInstantaneousTransitionPrior(lags=lag, 
                                                      latent_size=z_dim, 
                                                      num_layers=3, 
                                                      hidden_dim=hidden_dim,
                                                      z_dim_list=z_dim_list)
        self.transition_prior_change = NPChangeInstantaneousTransitionPrior(lags=lag, 
                                                        latent_size=z_dim,
                                                        embedding_dim=embedding_dim,
                                                        num_layers=3, 
                                                        hidden_dim=hidden_dim)
        
                                                            
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(self.z_dim))
        self.register_buffer('base_dist_var', torch.eye(self.z_dim))

    @property
    def base_dist(self):
        # Noise density function
        return D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)
    
    def reparameterize(self, mean, logvar, random_sampling=True):
        if random_sampling:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5*logvar)
            z = mean + eps*std
            return z
        else:
            return mean

    def reconstruction_loss(self, x, x_recon, distribution):
        batch_size = x.size(0)
        assert batch_size != 0

        if distribution == 'bernoulli':
            recon_loss = F.binary_cross_entropy_with_logits(
                x_recon, x, size_average=False).div(batch_size)

        elif distribution == 'gaussian':
            recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)

        elif distribution == 'sigmoid_gaussian':
            x_recon = F.sigmoid(x_recon)
            recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)

        return recon_loss

    def loss_function(self, x, x_recon, mus, logvars, zs, embeddings):
        '''
        VAE ELBO loss: recon_loss + kld_loss (past: N(0,1), future: N(0,1) after flow) + sparsity_loss
        '''
        batch_size, length, _ = x.shape

        # Sparsity loss
        sparsity_loss = 0
        # Recon loss
        recon_loss = self.reconstruction_loss(x[:,:self.lag], x_recon[:,:self.lag], self.decoder_dist) + \
        (self.reconstruction_loss(x[:,self.lag:], x_recon[:,self.lag:], self.decoder_dist))/(length-self.lag)
        
        std = torch.exp(torch.clamp(logvars, max=10.0, min=-10.0) / 2)
        q_dist = D.Normal(mus, std)
        log_qz = q_dist.log_prob(zs)

        # Past KLD
        p_dist = D.Normal(torch.zeros_like(mus[:,:self.lag]), torch.ones_like(logvars[:,:self.lag]))
        log_pz_normal = torch.sum(torch.sum(p_dist.log_prob(zs[:,:self.lag]),dim=-1),dim=-1)
        log_qz_normal = torch.sum(torch.sum(log_qz[:,:self.lag],dim=-1),dim=-1)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = kld_normal.mean()

        # Future KLD
        kld_future = []
        # fix
        if self.z_dim_fix>0:
            log_qz_laplace = log_qz[:,self.lag:,:self.z_dim_fix]
            residuals, logabsdet, hist_jac = self.transition_prior_fix.forward(zs[:,:,:self.z_dim_fix])
            log_pz_laplace = torch.sum(self.base_dist.log_prob(residuals), dim=1) + logabsdet
            # print(logabsdet.shape)
            # exit()
            kld_laplace = (torch.sum(torch.sum(log_qz_laplace,dim=-1),dim=-1) - log_pz_laplace) / (length-self.lag)
            kld_future.append(kld_laplace)
        kld_future = torch.cat(kld_future, dim=-1)
        # print(kld_future.shape)
        kld_future = kld_future.mean()

        # print(residuals.shape)
        # print(zs.shape)

        return sparsity_loss ,recon_loss, kld_normal, kld_future
    
    def forward(self, batch):
        # Prepare data
        if self.z_dim_change>0:
            x, y, c = batch['xt'], batch['yt'], batch['ct']
            c = torch.squeeze(c).to(torch.int64)
            embeddings = self.embed_func(c)
        else:
            x, y = batch['xt'], batch['yt']
            embeddings = None
        batch_size, length, dim = x.shape

        x_raw = x
        x = x.unfold(dimension = 1, size = self.n_mea, step = 1)
        x = x.reshape(batch_size, length - self.n_mea + 1, dim * self.n_mea)
        length = length - self.n_mea + 1 

        x_flat = x.view(-1, dim * self.n_mea)

        # Inference & Reshape to time-series format
        x_recon, mus, logvars, zs = self.net(x_flat)

        x_recon = x_recon.view(batch_size, length, self.input_dim)
        
        mus = mus.reshape(batch_size, length, self.z_dim)
        logvars  = logvars.reshape(batch_size, length, self.z_dim)
        zs = zs.reshape(batch_size, length, self.z_dim)

        return x_raw[:, (self.n_mea - 1) // 2 : - (self.n_mea - 1) // 2], x_recon, mus, logvars, zs, embeddings

    def training_step(self, batch, batch_idx):
        x, x_recon, mus, logvars, zs, embeddings = self.forward(batch)
        sparsity_loss, recon_loss, kld_normal, kld_future = self.loss_function(x, x_recon, mus, logvars, zs, embeddings)

        # VAE training
        loss = 0.1 * recon_loss + self.beta * kld_normal + self.gamma * kld_future # + self.theta * sparsity_loss
        self.log("train_elbo_loss", loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_kld_normal", self.beta * kld_normal)
        self.log("train_kld_future", self.gamma * kld_future)
        # self.log("train_sparsity_loss", self.theta * sparsity_loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        # for r2
        if self.z_dim_change>0:
            x, y, c = batch['xt'], batch['yt'], batch['ct']
            c = torch.squeeze(c).to(torch.int64)
            embeddings = self.embed_func(c)
        else:
            x, y = batch['xt'], batch['yt']
            embeddings = None
        batch_size, length, dim = x.shape
        
        x, x_recon, mus, logvars, zs, embeddings = self.forward(batch)
        
        sparsity_loss, recon_loss, kld_normal, kld_future = self.loss_function(x, x_recon, mus, logvars, zs, embeddings)

        # Compute Mean Correlation Coefficient (MCC)
        zt_recon = mus.reshape(-1, self.z_dim).T.detach().cpu().numpy()
        assert (self.n_mea - 1) % 2 == 0
        zt_true = batch["yt"][:, (self.n_mea - 1) // 2 : - (self.n_mea - 1) // 2].reshape(-1, self.z_dim).T.detach().cpu().numpy()
        bottom_mcc = compute_mcc(zt_recon[:, 1:], zt_true[:, 1:], self.correlation)       
        mcc = compute_mcc(zt_recon, zt_true, self.correlation)
        if mcc >= self.best_mcc:
            self.best_mcc = mcc
            self.best_bottom_mcc = bottom_mcc
            
        
        print(f"\n val_mcc: {mcc}")
        print(f"bottom_mcc: {bottom_mcc}")
        # best mcc by the validation history
        
        # if self.hz_to_z is False:
        #     r2 = compute_r2(mus[:, 1:].reshape(-1, self.z_dim-1), y[:, 1:].reshape(-1, self.z_dim-1))
        # else:
        #     r2 = compute_r2(y[:, 1:].reshape(-1, self.z_dim-1), mus[:, 1:].reshape(-1, self.z_dim-1))
            
        corr_matrix = np.zeros((self.z_dim, self.z_dim))

        for i in range(0, self.z_dim):
            for j in range(0, self.z_dim):
                # np.corrcoef returns a 2x2 matrix; [0,1] is correlation between the two arrays
                corr_matrix[i, j] = np.corrcoef(zt_true[i], zt_recon[j])[0, 1]  
        # --- Plot the correlation matrix ---
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            corr_matrix, 
            vmin=-1, vmax=1, 
            cmap="RdBu", 
            square=True, 
            annot=True, 
            fmt=".2f", 
            ax=ax
        )
        ax.set_title("Correlation Matrix (z_true vs z_recon)")
        ax.set_xlabel("Reconstructed Dimension")
        ax.set_ylabel("True Dimension")

        # --- Log the figure to wandb ---
        # wandb.init() 
        # wandb.log({"correlation_matrix": wandb.Image(fig)})
        logger = getattr(self, "logger", None)
        experiment = getattr(logger, "experiment", None)
        if experiment is not None:
            experiment.log({'correlation_matrix': wandb.Image(fig)})

        # It's good practice to close the figure after logging to avoid memory leaks
        plt.close(fig)


        # Validation loss
        loss = recon_loss + self.beta * kld_normal + self.gamma * kld_future # + self.theta * sparsity_loss
        self.log("val_mcc", mcc) 
        self.log("val_bottom_mcc", bottom_mcc)  
        self.log("best_mcc", self.best_mcc)
        self.log("best_bottom_mcc", self.best_bottom_mcc)
        self.log("val_elbo_loss", loss)
        self.log("val_recon_loss", recon_loss)
        self.log("val_kld_normal", self.beta * kld_normal)
        self.log("val_kld_future", self.gamma * kld_future)
        # self.log("val_sparsity_loss", self.theta * sparsity_loss)

        return loss

    def configure_optimizers(self):
        if self.optimizer_type == 'adam':
            opt_v = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0001)
            return [opt_v], []
        elif self.optimizer_type == 'annealing':
            opt_v = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0001)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_v, T_max=100, eta_min=1e-6)
            return [opt_v], [{"scheduler": scheduler, "interval": "epoch"}]
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")
