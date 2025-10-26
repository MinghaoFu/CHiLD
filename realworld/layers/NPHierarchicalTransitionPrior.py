import torch.nn as nn
import torch
from typing import List
from layers.MLP import MLP2
from torch.func import vmap


class NPHierarchicalTransitionPrior(nn.Module):
    def __init__(
            self,
            lags,
            layer_dim,
            num_layers=3,
            hidden_dim=64,
            gpu=0
    ):
        super().__init__()
        self.L = lags
        self.lags = lags
        self.layer_dim: List[int] = layer_dim
        self.gs_lst = []
        self.device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
        for i, dim in enumerate(layer_dim):
            if i == 0:
                # Pa_d(z^l_{t, i})
                net_lst = [MLP2(input_dim=lags * dim + 1,
                                output_dim=1,
                                num_layers=num_layers,
                                hidden_dim=hidden_dim).to(self.device) for _ in range(dim)]

            else:
                # Pa_d(z^l_{t, i}), Pa_h(z^l_{t, i})
                net_lst = [MLP2(input_dim=lags * dim + layer_dim[i - 1] + 1,
                                output_dim=1,
                                num_layers=num_layers,
                                hidden_dim=hidden_dim).to(self.device) for _ in range(dim)]
            self.gs_lst.append(nn.ModuleList(net_lst))

    def forward(self, x, alphas):
        # x: [BS, T, D] -> [BS, T- L - 1, L+1, D]
        batch_size, length, input_dim = x.shape
        # prepare data
        x = x.unfold(dimension=1, size=self.L + 1, step=1)
        x = torch.swapaxes(x, 2, 3)
        x = x.reshape(-1, self.L + 1, input_dim)
        xx, yy = x[:, -1:], x[:, :-1]
        yy = yy.reshape(-1, self.L * input_dim)
        # get residuals and |J|
        residuals = []
        hist_jac = []

        sum_log_abs_det_jacobian = 0
        # index range of time-delay parents for layer l latent variables
        delay_left, delay_right = 0, sum(self.layer_dim)  
        # index range of hierarchical parents for layer l latent variables, only when l != 0
        hiera_left, hiera_right = 0, 0  
        for (l, dim), gs in zip(enumerate(self.layer_dim), self.gs_lst):
            delay_left = delay_right - dim
            Pa_d = yy[:, delay_left: delay_right]
            for i in range(dim):
                if l == 0:  # Level = L
                    # Pa_d(z^l_{t - 1}) + Pa_d(z^l_{t, i})
                    inputs = torch.cat([Pa_d] + [xx[:, :, delay_left + i]], dim=-1)
                else:  # Level = l \neq L
                    assert hiera_right - hiera_left == self.layer_dim[l - 1]
                    Pa_h = xx[:, :, hiera_left: hiera_right].reshape(-1, self.L * self.layer_dim[l - 1])
                    # Pa_d(z^l_{t - 1}) +  Pa_h(z^l_{t}) + Pa_d(z^l_{t, i})
                    inputs = torch.cat([Pa_d] + [Pa_h] + [xx[:, :, i]], dim=-1)
                residual = gs[i](inputs)
                with torch.enable_grad():
                    pdd = vmap(torch.func.jacfwd(gs[i]))(inputs)
                # Determinant: product of diagonal entries, sum of last entry
                logabsdet = torch.log(torch.abs(pdd[:, 0, -1]))
                hist_jac.append(torch.unsqueeze(pdd[:, 0, :-1], dim=1))
                sum_log_abs_det_jacobian += logabsdet
                residuals.append(residual)
            # update index of Pa_d, Pa_h
            hiera_left, hiera_right = delay_left, delay_right
            delay_right -= dim

        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, -1, input_dim)
        sum_log_abs_det_jacobian = torch.sum(sum_log_abs_det_jacobian.reshape(batch_size, length - self.L), dim=1)
        return residuals, sum_log_abs_det_jacobian, hist_jac
