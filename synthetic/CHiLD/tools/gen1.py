import os
import glob
import tqdm
import torch
import scipy
import random
import numpy as np
from torch import nn
from torch.nn import init
from collections import deque
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.stats import ortho_group
from sklearn.preprocessing import scale
# from CHiLD.tools.utils import create_sparse_transitions, controlable_sparse_transitions
# ^ Uncomment or remove these imports based on your actual code usage.

root_dir = '../datasets/dataset3'

###############################
# Activation and initialization
###############################

def leaky_ReLU_1d(x, neg_slope):
    """
    A simple 1D leaky ReLU for scalar x.
    """
    return x if x > 0 else neg_slope * x

leaky1d = np.vectorize(leaky_ReLU_1d)

def leaky_ReLU(X, neg_slope):
    """
    Vectorized leaky ReLU for array X.
    """
    assert neg_slope > 0
    return leaky1d(X, neg_slope)

def weight_init(m):
    """
    Example PyTorch weight initialization.
    You can keep or remove this depending on usage.
    """
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

def sigmoidAct(x):
    """
    Simple sigmoid function.
    """
    return 1.0 / (1.0 + np.exp(-x))

###############################
# Matrix generation helpers
###############################

def generate_uniform_mat(n_comp, cond_thresh):
    """
    Generate a random (n_comp x n_comp) matrix by sampling each element
    uniformly in [-1, 1], normalizing each column, and ensuring the
    condition number is below cond_thresh. This can loop until it finds
    a valid matrix.
    """
    A = np.random.uniform(0, 2, (n_comp, n_comp)) - 1.0
    # Normalize each column
    for col in range(n_comp):
        A[:, col] /= np.sqrt((A[:, col] ** 2).sum())
    
    # Keep trying until condition is below threshold
    while np.linalg.cond(A) > cond_thresh:
        A = np.random.uniform(0, 2, (n_comp, n_comp)) - 1.0
        for col in range(n_comp):
            A[:, col] /= np.sqrt((A[:, col] ** 2).sum())
    
    return A

###############################
# Main data generation
###############################

def generate_hierarchical_ts(z_dim=5, hist=0, inst=False, name='data'):
    """
    Generate hierarchical time-series data with a 'top' latent dimension (z^2)
    and 'bottom' latent dimensions (z^1), plus observed x_t.

    Args:
        z_dim (int): total latent dimension = 1 (top) + (z_dim-1) (bottom).
        hist (int): determines how transitions are constructed (0,1,2,3).
        inst (bool): whether to treat the noise injection as instantaneous or not.
        name (str): folder name for saving data.

    Returns:
        None. Saves (y_t, x_t) and transition matrices in the specified folder.
    """

    # Basic hyperparameters
    lags = 1
    n_layers = 2
    seq_length = 5       # How many transitions to generate beyond initial lags
    neg_slope = 0.2
    noise_scale = 0.1
    batch_size = 100_000
    n_iter_cond = int(1e4)

    path = os.path.join(root_dir, name)
    os.makedirs(path, exist_ok=True)

    #---------------------------------------
    # Precompute a condition threshold
    #---------------------------------------
    cond_list = []
    for _ in range(n_iter_cond):
        A = np.random.uniform(1, 2, (z_dim, z_dim))
        # Normalize columns
        for col in range(z_dim):
            A[:, col] /= np.sqrt((A[:, col] ** 2).sum())
        cond_list.append(np.linalg.cond(A))

    cond_thresh = np.percentile(cond_list, 25)  
    # Only accept matrices whose cond # is below the 25th percentile

    #---------------------------------------
    # Create transition matrices
    #---------------------------------------
    transitions = []
    for _lag in range(lags):
        if hist == 0:
            # fully random with condition check
            B = generate_uniform_mat(z_dim, cond_thresh)
            transitions.append(B)

        elif hist == 1:
            # fixed example for z_dim=3
            assert z_dim == 3, "hist=1 only coded for z_dim=3 example."
            transitions.append(np.array([
                [0.4, 0.6, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32))

        elif hist == 2:
            # identity matrix
            transitions.append(np.eye(z_dim, dtype=np.float32))

        elif hist == 3:
            # an example case for z_dim=5
            assert z_dim == 5, "hist=3 only coded for z_dim=5 example."
            trans_mat = np.eye(5, dtype=np.float32)
            # Quick modifications to illustrate non-trivial transitions
            trans_mat[1][3] += 1.0
            trans_mat[2][4] += 1.0
            transitions.append(trans_mat)

    # Note: The original code reversed transitions; keep or remove depending on usage
    transitions.reverse()

    #---------------------------------------
    # Create mixing layers for "bottom" dimension -> x
    #---------------------------------------
    mixing_list = []
    # We do n_layers - 1 mixing layers
    for _ in range(n_layers - 1):
        # Generate orthonormal matrix for the bottom dimension
        # but note we're effectively mixing z_dim-1 dimensions
        # since the top dimension is separate
        A_ortho = ortho_group.rvs(z_dim - 1)
        mixing_list.append(A_ortho)

    #---------------------------------------
    # Initialize the latent states for "lags" timesteps
    # shape: (batch_size, lags, z_dim)
    # y[..., 0] = top dimension, y[..., 1:] = bottom dimension(s)
    #---------------------------------------
    y_init = np.random.normal(0, 1, (batch_size, lags, z_dim))
    # Standardize across each dimension for the initial lags
    y_init = (y_init - np.mean(y_init, axis=0, keepdims=True)) / \
             np.std(y_init, axis=0, keepdims=True)

    # Keep track of these latents in lists
    y_list = [y_init[:, i, :] for i in range(lags)]

    #---------------------------------------
    # Create x by mixing bottom dimensions (z^1)
    # x has shape (batch_size, lags, z_dim - 1)
    #---------------------------------------
    x_init = np.copy(y_init[:, :, 1:])  # drop the top dimension
    for layer_mat in mixing_list:
        x_init = leaky_ReLU(x_init, neg_slope)
        x_init = np.dot(x_init, layer_mat)
    x_list = [x_init[:, i, :] for i in range(lags)]

    #---------------------------------------
    # Generate the next seq_length steps
    #---------------------------------------
    for tstep in range(seq_length):
        # ~~~~~ TRANSITION IN LATENT SPACE ~~~~~
        # sum over the lagged timesteps
        y_t_new = np.zeros((batch_size, z_dim), dtype=np.float32)
        for lag_idx in range(lags):
            y_t_new += leaky_ReLU(np.dot(y_list[lag_idx], transitions[lag_idx]),
                                  neg_slope)
        y_t_new = leaky_ReLU(y_t_new, neg_slope)

        # Add noise
        y_noise = np.random.normal(0, noise_scale, (batch_size, z_dim)) * 2
        # For top dimension (index=0) specifically:
        y_t_new[:, 0] = y_t_new[:, 0] * y_noise[:, 0] + y_noise[:, 0]

        # For bottom dimensions (1..z_dim-1)
        p_hist = 0.2
        for d in range(1, z_dim):
            if inst:
                # instantaneous version
                y_t_new[:, d] = y_t_new[:, d] * y_noise[:, d] + y_noise[:, d]
            else:
                # partially depends on top dimension
                y_t_new[:, d] = (
                    p_hist * y_t_new[:, d] +
                    (1 - p_hist) * y_t_new[:, 0]
                ) * y_noise[:, d] + y_noise[:, d]

        # ~~~~~ MIXING TO GET x ~~~~~
        # Only bottom dims feed into x
        x_bottom = np.copy(y_t_new[:, 1:])
        for layer_mat in mixing_list:
            x_bottom = leaky_ReLU(x_bottom, neg_slope)
            x_bottom = np.dot(x_bottom, layer_mat)

        #---------------------------------------
        # Update arrays and shift lags
        #---------------------------------------
        y_list.append(y_t_new)
        x_list.append(x_bottom)
        # Slide the window for y_list to keep the last 'lags' states
        # In your code lags=1, so effectively you just keep the new one.
        y_list = y_list[-lags:]  # keep the last `lags` items

    #---------------------------------------
    # Convert to shape (batch_size, total_time, dim)
    # total_time = lags + seq_length
    #---------------------------------------
    Y_out = np.stack(y_list, axis=1) if lags > 1 else y_list[0][:, np.newaxis, :]
    X_out = np.stack(x_list, axis=1) if lags > 1 else x_list[0][:, np.newaxis, :]

    # For clarity, if lags=1, Y_out is shape (batch_size, seq_length+1, z_dim).
    # If lags>1, it’s shape (batch_size, lags+seq_length, z_dim).
    # But you might want them all in the same shape. Let's unify them:
    Y_out = np.array(y_list).transpose(1, 0, 2)  # shape (batch, time, z_dim)
    X_out = np.array(x_list).transpose(1, 0, 2)  # shape (batch, time, z_dim-1)

    #---------------------------------------
    # Save the data
    #---------------------------------------
    np.savez(os.path.join(path, "data"), yt=Y_out, xt=X_out)
    print("Saved Y_out shape:", Y_out.shape)
    print("Saved X_out shape:", X_out.shape)

    # Save transition matrices
    for idx, B_mat in enumerate(transitions):
        np.save(os.path.join(path, f"W{lags - idx}"), B_mat)

###############################
# Example usage
###############################
if __name__ == "__main__":
    # Generate data for different configurations
    generate_hierarchical_ts(z_dim=5, hist=2, inst=False, name='A')
    # generate_hierarchical_ts(z_dim=5, hist=2, inst=False, name='B')
    # generate_hierarchical_ts(z_dim=8, hist=2, inst=False, name='C')
    # generate_hierarchical_ts(z_dim=8, hist=0, inst=False, name='D')
    # generate_hierarchical_ts(z_dim=8, hist=0, inst=True,  name='E')
    # generate_hierarchical_ts(z_dim=16, hist=2, inst=True, name='F')