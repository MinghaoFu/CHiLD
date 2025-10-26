import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, layer_nums, in_dim, hid_dim=None, out_dim=None, activation="gelu", layer_norm=True):
        super().__init__()
        if activation == "gelu":
            a_f = nn.GELU()
        elif activation == "relu":
            a_f = nn.ReLU()
        elif activation == "tanh":
            a_f = nn.Tanh()
        else:
            a_f = nn.Identity()
        if out_dim is None:
            out_dim = in_dim
        if layer_nums == 1:
            net = [nn.Linear(in_dim, out_dim)]
        else:

            net = [nn.Linear(in_dim, hid_dim), a_f, nn.LayerNorm(hid_dim)] if layer_norm else [
                nn.Linear(in_dim, hid_dim), a_f]
            for i in range(layer_nums - 2):
                net.append(nn.Linear(hid_dim, hid_dim))
                net.append(a_f)
            net.append(nn.Linear(hid_dim, out_dim))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class MLP2(nn.Module):
    """A simple MLP with ReLU activations"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, leaky_relu_slope=0.2):
        super().__init__()
        layers = []
        for l in range(num_layers):
            if l == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.LeakyReLU(leaky_relu_slope))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.LeakyReLU(leaky_relu_slope))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Base_Net(nn.Module):
    def __init__(self, input_len, out_len, input_dim, out_dim, is_mean_std=True, activation="gelu",
                 layer_norm=True, c_type="None", drop_out=0, layer_nums=2) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.out_len = out_len
        self.c_type = c_type
        self.out_dim = out_dim

        self.radio = 2 if is_mean_std else 1

        if self.c_type == "None":
            self.net = MLP(layer_nums, in_dim=input_len, out_dim=out_len * self.radio, hid_dim=out_len * 2,
                           activation=activation,
                           layer_norm=layer_norm)
        elif self.c_type == "type1":
            self.net = MLP(layer_nums, in_dim=self.input_dim, hid_dim=self.out_dim * 2,
                           out_dim=self.out_dim * self.radio,
                           layer_norm=layer_norm, activation=activation)
        elif self.c_type == "type2":
            self.net = MLP(layer_nums, in_dim=self.input_dim * input_len, hid_dim=self.out_dim * 2 * out_len,
                           activation=activation,
                           out_dim=self.out_dim * out_len * self.radio, layer_norm=layer_norm)

        self.dropout_net = nn.Dropout(drop_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.c_type == "None":
            x = self.net(x.permute(0, 2, 1)).permute(0, 2, 1)
        elif self.c_type == "type1":
            x = self.net(x)
        elif self.c_type == "type2":
            x = self.net(x.reshape(x.shape[0], -1)).reshape(x.shape[0], -1, self.out_dim * self.radio)

        x = self.dropout_net(x)
        if self.radio == 2:
            dim = 2 if self.c_type == "type1" or self.c_type == "type2" else 1
            x1, x2 = torch.chunk(x, dim=dim, chunks=2)
            x2 = self.sigmoid(x2)
            return x1, x2
        else:
            return x
