import torch
import torch.nn as nn

import torch.distributions as D

from layers import Normalize, NPHierarchicalTransitionPrior, Base_Net, Embedding_Net


class SparsityMatrix(nn.Module):
    def __init__(self, z_dim):
        super(SparsityMatrix, self).__init__()
        prior_matrix = torch.zeros((z_dim, z_dim)).cuda()
        for i in range(z_dim - 1):
            prior_matrix[i + 1, i] = 1
            pass
        prior_matrix = torch.tensor(prior_matrix)
        self.trainable_parameters = nn.Parameter(prior_matrix)

    def forward(self):
        return self.trainable_parameters


class Model(nn.Module):

    def __init__(self, configs):

        super(Model, self).__init__()

        self.configs = configs
        self.seq_len = configs.seq_len
        self.c_type = configs.c_type
        self.layer_dim = configs.layer
        assert self.layer_dim != [], 'layer must be non-empty'
        assert all(self.layer_dim[i] <= self.layer_dim[i + 1] for i in
                   range(len(self.layer_dim) - 1)), 'layer_dim must be non-decreasing'
        self.z_dim = sum(self.layer_dim)

        self.normalize = Normalize(num_features=configs.enc_in, affine=True, non_norm=not configs.is_norm)
        self.emb_net = Embedding_Net(configs.patch_size, configs.seq_len, configs.seq_len, configs.emb_dim)
        self.encoder = Base_Net(self.configs.seq_len, configs.seq_len, configs.enc_in * self.configs.n_concat,
                                self.z_dim, c_type=self.c_type,
                                layer_norm=configs.is_ln, activation=configs.activation,
                                drop_out=configs.dropout, layer_nums=self.configs.layer_nums)
        self.decoder = Base_Net(configs.seq_len, configs.seq_len, self.z_dim, configs.enc_in,
                                c_type=self.c_type, layer_norm=configs.is_ln, activation=configs.activation,
                                drop_out=configs.dropout, layer_nums=self.configs.layer_nums, is_mean_std=False)

        self.decoder_dist = 'gaussian'
        self.alphas_fix = SparsityMatrix(self.z_dim)
        self.alphas_fix().requires_grad = False
        self.lag = self.configs.lags
        self.transition_prior_fix = NPHierarchicalTransitionPrior(lags=self.lag,
                                                                  layer_dim=self.layer_dim,
                                                                  num_layers=1,
                                                                  hidden_dim=8,
                                                                  gpu=configs.gpu)
        self.register_buffer('base_dist_mean', torch.zeros(self.z_dim))
        self.register_buffer('base_dist_var', torch.eye(self.z_dim))

    @property
    def base_dist(self):
        # Noise density function
        return D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)

    def loss_function(self, x, mus, logvars, zs):
        batch_size, length, _ = x.shape

        unmasked_alphas_fix = self.alphas_fix()
        mask_fix = (unmasked_alphas_fix > 0.1).float()
        alphas_fix = unmasked_alphas_fix * mask_fix

        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(zs)

        # Past KLD
        p_dist = D.Normal(torch.zeros_like(mus[:, :self.lag]), torch.ones_like(logvars[:, :self.lag]))
        log_pz_normal = torch.sum(torch.sum(p_dist.log_prob(zs[:, :self.lag]), dim=-1), dim=-1)
        log_qz_normal = torch.sum(torch.sum(log_qz[:, :self.lag], dim=-1), dim=-1)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = kld_normal.mean()

        # Future KLD
        kld_future = []

        log_qz_laplace = log_qz[:, self.lag:, :]
        residuals, logabsdet, hist_jac = self.transition_prior_fix.forward(zs, alphas_fix)
        log_pz_laplace = torch.sum(self.base_dist.log_prob(residuals), dim=1) + logabsdet
        kld_laplace = (torch.sum(torch.sum(log_qz_laplace, dim=-1), dim=-1) - log_pz_laplace) / (length - self.lag)
        kld_future.append(kld_laplace)

        kld_future = torch.cat(kld_future, dim=-1)
        kld_future = kld_future.mean()

        return kld_normal, kld_future

    def forward(self, x_enc, is_train=True):
        std_x = self.normalize(x_enc, "norm")
        emb = self.emb_net(std_x)
        emb = self.n_concat(emb, self.configs.n_concat)
        z_mean, z_std = self.encoder(emb)

        if is_train:
            z = self.reparametrize(z_mean, z_std)
            x_recon = self.decoder(z)
            x_recon = self.normalize(x_recon, "denorm")
            kld_normal, kld_future = self.loss_function(x_recon, z_mean, z_std, z)  #
            other_loss = self.configs.kld_weight * kld_normal + self.configs.kld_weight * kld_future
            return x_recon, other_loss
        else:
            z = z_mean
            x_recon = self.decoder(z)

            x_recon = self.normalize(x_recon, "denorm")
            return x_recon

    @staticmethod
    def reparametrize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    @staticmethod
    def n_concat(x, window):
        assert window % 2 != 0
        batch_size, length, _ = x.shape
        final_lenth = abs(length - window + 1)
        pad_len = abs(int((final_lenth - length) / 2))

        if pad_len != 0:
            pad_x = torch.cat([x[:, :pad_len, :], x, x[:, -pad_len:, :]], dim=1)
        else:
            pad_x = torch.cat([x], dim=1)
        window_x = pad_x.unfold(dimension=1, size=window, step=1)
        output_x = window_x.reshape(batch_size, length, -1)
        return output_x
