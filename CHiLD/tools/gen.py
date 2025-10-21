import os
import sys
import glob
import tqdm
import torch
import scipy
import random
import ipdb as pdb
import numpy as np
from torch import nn
from torch.nn import init
from collections import deque
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.stats import ortho_group
from sklearn.preprocessing import scale
# Ensure the project root is on sys.path so local imports resolve when run as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from CHiLD.tools.utils import create_sparse_transitions, controlable_sparse_transitions
from itertools import product, permutations
from scipy.linalg import block_diag
import argparse

VALIDATION_RATIO = 0.2
root_dir = '../datasets/dataset3'
standard_scaler = preprocessing.StandardScaler()

def random_permutation_matrix(n):
    P = np.eye(n)
    np.random.shuffle(P)
    return P

def block_diagonal_permutation_matrix(z_dim_list=[1, 2, 4]):
    'generate transition, simple version'
    blocks = [random_permutation_matrix(block_size) for block_size in z_dim_list]
    return block_diag(*blocks)

def leaky_ReLU_1d(d, negSlope):
    if d > 0:
        return d
    else:
        return d * negSlope

leaky1d = np.vectorize(leaky_ReLU_1d)

# def leaky_ReLU(D, negSlope):
#     assert negSlope > 0
#     return leaky1d(D, negSlope)

def leaky_ReLU(D, negSlope):
    return D

def tanh_activation(D):
    return np.tanh(D)

def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data,0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0,0.01)
        m.bias.data.zero_()

def sigmoidAct(x):
    return 1. / (1 + np.exp(-1 * x))

def generateUniformMat(Ncomp, condT):
    """
    generate a random matrix by sampling each element uniformly at random
    check condition number versus a condition threshold
    """
    A = np.random.uniform(0, 2, (Ncomp, Ncomp)) - 1
    for i in range(Ncomp):
        A[:, i] /= np.sqrt((A[:, i] ** 2).sum())

    while np.linalg.cond(A) > condT:
        # generate a new A matrix!
        A = np.random.uniform(0, 2, (Ncomp, Ncomp)) - 1
        for i in range(Ncomp):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())

    return A


def stationary_z_link_temporal(obs_dim = 4, hist = 0, inst = False, name = 'data', matrix=None, seed=None, z_dim_list=[1, 2, 4]):
    """
    Generate temporal stationary data where top layer (dim=1) at time t-1 
    contributes to down layer (dim=2) at time t through linear additive model
    """
    lags = 1
    Nlayer = 2
    length = 6
    condList = []
    negSlope = 0.2
    latent_size = sum(z_dim_list)
    transitions = []
    noise_scale = 0.1
    batch_size = 100000
    Niter4condThresh = 1e4
    
    n_layer = len(z_dim_list)   
    
    if seed is not None:
        np.random.seed(seed)

    path = os.path.join(root_dir, name)
    os.makedirs(path, exist_ok=True)

    for i in range(int(Niter4condThresh)):
        A = np.random.uniform(1, 2, (latent_size, latent_size))
        for i in range(latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))

    condThresh = np.percentile(condList, 25)
    for l in range(lags):
        if hist == 0:
            B = generateUniformMat(latent_size, condThresh)
            transitions.append(B)
        elif hist == 1:
            assert(latent_size == 3)
            transitions.append(np.array([[0.4, 0.6, 0],
                                         [0, 1, 0],
                                         [0, 0, 1]], dtype=np.float32))
        elif hist == 2:
            if matrix is None:
                transitions.append(block_diagonal_permutation_matrix(z_dim_list))
            else:
                transitions.append(matrix)
        elif hist == 3:
            assert(latent_size == 5)
            trans_mat = np.eye(5, dtype=np.float32)
            trans_mat[1][3]+=1.0
            trans_mat[2][4]+=1.0
            transitions.append(trans_mat)
        
    transitions.reverse()

    all_mixingList = []
    for _ in range(n_layer):
        mixingList = []
        for l in range(Nlayer - 1):
            A = ortho_group.rvs(obs_dim)
            mixingList.append(A)
        all_mixingList.append(mixingList)

    y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
    y_l = (y_l - np.mean(y_l, axis=0, keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)

    yt = []; xt = []
    for i in range(lags):
        yt.append(y_l[:,i,:])
    mixedDat = np.copy(y_l[:,:,-z_dim_list[-1]:])
        
    all_mixedDat = []
    for i in range(1):        
        mixedDat = np.copy(y_l[:,:,-z_dim_list[-1]:])
        for l in range(Nlayer - 1):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            mixedDat = np.dot(mixedDat, all_mixingList[i][l])
        
        all_mixedDat.append(mixedDat)
    all_mixedDat = np.concatenate(all_mixedDat, axis=-1)
    x_l = np.copy(all_mixedDat)
    for i in range(lags):
        xt.append(x_l[:,i,:])
    
    # Create temporal connection matrices for cross-links
    # t-1 dim=1 -> t dim=2
    temporal_connection_1_to_2 = np.random.normal(0, 0.4, (z_dim_list[0], z_dim_list[1]))  # 1x2 matrix
    # t-1 dim=2 -> t dim=4  
    temporal_connection_2_to_4 = np.random.normal(0, 0.4, (z_dim_list[1], z_dim_list[2]))  # 2x4 matrix
    
    # Layer mixing matrices
    layer_mixing_list = []
    for idx, z_dim in enumerate(z_dim_list):
        if idx == len(z_dim_list) - 1:
            layer_mixing_list.append(np.ones((z_dim, obs_dim)))
        else:
            layer_mixing_list.append(np.ones((z_dim, z_dim_list[idx+1])))
        
    # Time evolution with temporal connections
    for i in range(length):
        # Transition function
        y_t_noise = np.random.normal(0, noise_scale, (batch_size, latent_size)) * 2
        x_t_noise = np.random.normal(0, noise_scale, (batch_size, latent_size))
        
        y_t = 0
        # Standard transition
        for l in range(lags):
            y_t += leaky_ReLU(np.dot(y_l[:,l,:], transitions[l]), negSlope)
        y_t = leaky_ReLU(y_t, negSlope)

        p_hist = 0.2
        
        # First layer (top layer, dim=1)
        y_t[:,:z_dim_list[0]] = y_t[:,:z_dim_list[0]] + 1 * y_t_noise[:,:z_dim_list[0]]
        
        # Cross-link temporal connections
        if i > 0:  # Only apply temporal connections after first timestep
            prev_y = yt[-1]  # Previous timestep values
            
            # Connection 1: t-1 dim=1 -> t dim=2
            prev_layer_0 = prev_y[:, :z_dim_list[0]]  # Previous layer 0 (dim=1)
            temporal_contrib_1_to_2 = 0.5 * prev_layer_0 @ temporal_connection_1_to_2  # 1x2 -> batch x 2
            
            # Add to layer 1 (dim=2) at current time t
            layer1_start = z_dim_list[0]
            layer1_end = z_dim_list[0] + z_dim_list[1]
            y_t[:, layer1_start:layer1_end] += temporal_contrib_1_to_2
            
            # Connection 2: t-1 dim=2 -> t dim=4
            prev_layer_1 = prev_y[:, z_dim_list[0]:z_dim_list[0] + z_dim_list[1]]  # Previous layer 1 (dim=2)
            temporal_contrib_2_to_4 = 0.4 * prev_layer_1 @ temporal_connection_2_to_4  # 2x4 -> batch x 4
            
            # Add to layer 2 (dim=4) at current time t
            layer2_start = z_dim_list[0] + z_dim_list[1]
            layer2_end = z_dim_list[0] + z_dim_list[1] + z_dim_list[2]
            y_t[:, layer2_start:layer2_end] += temporal_contrib_2_to_4
        
        # Process remaining layers with standard hierarchical connections
        for layer in range(1, n_layer):
            for j in range(z_dim_list[layer]):
                if inst:
                    y_t[:,j] = y_t[:,j] * y_t_noise[:,j] + y_t_noise[:,j]
                else:
                    layer_start = sum(z_dim_list[:layer])
                    prev_layer_start = sum(z_dim_list[:layer-1])
                    prev_layer_end = sum(z_dim_list[:layer])
                    
                    # Standard hierarchical connection from previous layer
                    standard_contribution = (1-p_hist) * j * y_t[:, prev_layer_start:prev_layer_end] @ layer_mixing_list[layer-1][:, j]
                    
                    y_t[:, layer_start + j] = (
                        p_hist * y_t[:, layer_start + j] +
                        standard_contribution +
                        1 * y_t_noise[:,j]
                    )

        yt.append(y_t)
        
        # Mixing function
        all_mixedDat = []   
        for k in range(1):
            mixedDat = np.copy(y_t[:, -z_dim_list[-1]:])
            for l in range(Nlayer - 1):
                mixedDat = leaky_ReLU(mixedDat, negSlope)
                mixedDat = np.dot(mixedDat, all_mixingList[k][l])
            all_mixedDat.append(mixedDat)
        all_mixedDat = np.concatenate(all_mixedDat, axis=-1)    
        x_t = np.copy(all_mixedDat)
        xt.append(x_t)
        y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]

    yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2)
    
    np.savez(os.path.join(path, "data"), 
            yt = yt, 
            xt = xt)
    print(f"Cross-link temporal dataset {name} generated: yt.shape = {yt.shape}, xt.shape = {xt.shape}")

    for l in range(lags):
        B = transitions[l]
        np.save(os.path.join(path, "W%d"%(lags-l)), B)    

def stationary_z_link_hierarchical(obs_dim = 4, hist = 0, inst = False, name = 'data', matrix=None, seed=None, z_dim_list=[1, 2, 4]):
    """
    Generate hierarchical stationary data where highest layer (dim=1) can contribute to lowest layer (dim=4)
    through linear additive connections
    """
    lags = 1
    Nlayer = 2
    length = 6
    condList = []
    negSlope = 0.2
    latent_size = sum(z_dim_list)
    transitions = []
    noise_scale = 0.1
    batch_size = 100000
    Niter4condThresh = 1e4
    
    n_layer = len(z_dim_list)   
    
    if seed is not None:
        np.random.seed(seed)

    path = os.path.join(root_dir, name)
    os.makedirs(path, exist_ok=True)

    for i in range(int(Niter4condThresh)):
        A = np.random.uniform(1, 2, (latent_size, latent_size))
        for i in range(latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))

    condThresh = np.percentile(condList, 25)
    for l in range(lags):
        if hist == 0:
            B = generateUniformMat(latent_size, condThresh)
            transitions.append(B)
        elif hist == 1:
            assert(latent_size == 3)
            transitions.append(np.array([[0.4, 0.6, 0],
                                         [0, 1, 0],
                                         [0, 0, 1]], dtype=np.float32))
        elif hist == 2:
            if matrix is None:
                transitions.append(block_diagonal_permutation_matrix(z_dim_list))
            else:
                transitions.append(matrix)
        elif hist == 3:
            assert(latent_size == 5)
            trans_mat = np.eye(5, dtype=np.float32)
            trans_mat[1][3]+=1.0
            trans_mat[2][4]+=1.0
            transitions.append(trans_mat)
        
    transitions.reverse()

    all_mixingList = []
    for _ in range(n_layer):
        mixingList = []
        for l in range(Nlayer - 1):
            A = ortho_group.rvs(obs_dim)
            mixingList.append(A)
        all_mixingList.append(mixingList)

    y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
    y_l = (y_l - np.mean(y_l, axis=0, keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)

    yt = []; xt = []
    for i in range(lags):
        yt.append(y_l[:,i,:])
    mixedDat = np.copy(y_l[:,:,-z_dim_list[-1]:])
        
    all_mixedDat = []
    for i in range(1):        
        mixedDat = np.copy(y_l[:,:,-z_dim_list[-1]:])
        for l in range(Nlayer - 1):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            mixedDat = np.dot(mixedDat, all_mixingList[i][l])
        
        all_mixedDat.append(mixedDat)
    all_mixedDat = np.concatenate(all_mixedDat, axis=-1)
    x_l = np.copy(all_mixedDat)
    for i in range(lags):
        xt.append(x_l[:,i,:])
    
    # Create hierarchical mixing matrices with top-to-bottom connections
    layer_mixing_list = []
    # Create connection matrix from layer 0 (dim=1) to layer 2 (dim=4)
    top_to_bottom_connection = np.random.normal(0, 0.5, (z_dim_list[0], z_dim_list[-1]))  # 1x4 matrix
    
    for idx, z_dim in enumerate(z_dim_list):
        if idx == len(z_dim_list) - 1:
            layer_mixing_list.append(np.ones((z_dim, obs_dim)))
        else:
            layer_mixing_list.append(np.ones((z_dim, z_dim_list[idx+1])))
        
    # Mixing function with hierarchical connections
    for i in range(length):
        # Transition function
        y_t_noise = np.random.normal(0, noise_scale, (batch_size, latent_size)) * 2
        x_t_noise = np.random.normal(0, noise_scale, (batch_size, latent_size))
        
        y_t = 0
        # transition
        for l in range(lags):
            y_t += leaky_ReLU(np.dot(y_l[:,l,:], transitions[l]), negSlope)
        y_t = leaky_ReLU(y_t, negSlope)

        p_hist = 0.2 # the weight of history data
        
        # First layer (highest, dim=1)
        y_t[:,:z_dim_list[0]] = y_t[:,:z_dim_list[0]] + 1 * y_t_noise[:,:z_dim_list[0]]
        
        # Middle and bottom layers with hierarchical connections
        for layer in range(1, n_layer):
            for i in range(z_dim_list[layer]):
                if inst:
                    y_t[:,i] = y_t[:,i] * y_t_noise[:,i] + y_t_noise[:,i]
                else:
                    layer_start = sum(z_dim_list[:layer])
                    prev_layer_start = sum(z_dim_list[:layer-1])
                    prev_layer_end = sum(z_dim_list[:layer])
                    
                    # Standard hierarchical connection from previous layer
                    standard_contribution = (1-p_hist) * i * y_t[:, prev_layer_start:prev_layer_end] @ layer_mixing_list[layer-1][:, i]
                    
                    # Special: Add direct connection from top layer (layer 0) to bottom layer (layer 2)
                    hierarchical_contribution = 0
                    if layer == n_layer - 1:  # This is the bottom layer (dim=4)
                        # Direct contribution from top layer (dim=1) to bottom layer (dim=4)
                        top_layer_values = y_t[:, :z_dim_list[0]]  # Get top layer values
                        hierarchical_contribution = 0.3 * top_layer_values @ top_to_bottom_connection[:, i:i+1].flatten()
                    
                    y_t[:, layer_start + i] = (
                        p_hist * y_t[:, layer_start + i] +
                        standard_contribution +
                        hierarchical_contribution +
                        1 * y_t_noise[:,i]
                    )

        yt.append(y_t)
        
        # Mixing function
        all_mixedDat = []   
        for k in range(1):
            mixedDat = np.copy(y_t[:, -z_dim_list[-1]:])
            for l in range(Nlayer - 1):
                mixedDat = leaky_ReLU(mixedDat, negSlope)
                mixedDat = np.dot(mixedDat, all_mixingList[k][l])
            all_mixedDat.append(mixedDat)
        all_mixedDat = np.concatenate(all_mixedDat, axis=-1)    
        x_t = np.copy(all_mixedDat)
        xt.append(x_t)
        y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]

    yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2)
    
    np.savez(os.path.join(path, "data"), 
            yt = yt, 
            xt = xt)
    print(f"Hierarchical dataset {name} generated: yt.shape = {yt.shape}, xt.shape = {xt.shape}")

    for l in range(lags):
        B = transitions[l]
        np.save(os.path.join(path, "W%d"%(lags-l)), B)    

def stationary_z_link_4layer_mlp(obs_dim = 4, hist = 0, inst = False, name = 'data', matrix=None, seed=None, z_dim_list=[1, 2, 4]):
    """
    Generate stationary data with 4-layer MLP for mixing procedure (z->x)
    z_dim_list stays [1, 2, 4] but mixing uses 4 layers
    """
    lags = 1
    Nlayer = 4  # 4-layer MLP for mixing
    length = 6
    condList = []
    negSlope = 0.2
    latent_size = sum(z_dim_list)
    transitions = []
    noise_scale = 0.1
    batch_size = 100000
    Niter4condThresh = 1e4
    
    n_layer = len(z_dim_list)   
    
    if seed is not None:
        np.random.seed(seed)

    path = os.path.join(root_dir, name)
    os.makedirs(path, exist_ok=True)

    for i in range(int(Niter4condThresh)):
        A = np.random.uniform(1, 2, (latent_size, latent_size))
        for i in range(latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))

    condThresh = np.percentile(condList, 25)
    for l in range(lags):
        if hist == 0:
            B = generateUniformMat(latent_size, condThresh)
            transitions.append(B)
        elif hist == 1:
            assert(latent_size == 3)
            transitions.append(np.array([[0.4, 0.6, 0],
                                         [0, 1, 0],
                                         [0, 0, 1]], dtype=np.float32))
        elif hist == 2:
            if matrix is None:
                transitions.append(block_diagonal_permutation_matrix(z_dim_list))
            else:
                transitions.append(matrix)
        elif hist == 3:
            assert(latent_size == 5)
            trans_mat = np.eye(5, dtype=np.float32)
            trans_mat[1][3]+=1.0
            trans_mat[2][4]+=1.0
            transitions.append(trans_mat)
        
    transitions.reverse()

    # Create 4-layer MLP mixing networks
    all_mixingList = []
    for _ in range(1):  # Only create one mixing network
        mixingList = []
        # Layer dimensions for 4-layer MLP: z_dim_list[-1] -> hidden -> hidden -> hidden -> obs_dim
        layer_dims = [z_dim_list[-1], 64, 32, 16, obs_dim]  # 4 layers: input->hidden1->hidden2->hidden3->output
        
        for l in range(Nlayer - 1):  # 3 weight matrices for 4 layers
            input_dim = layer_dims[l]
            output_dim = layer_dims[l + 1]
            A = ortho_group.rvs(max(input_dim, output_dim))[:input_dim, :output_dim]
            mixingList.append(A)
        all_mixingList.append(mixingList)

    y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
    y_l = (y_l - np.mean(y_l, axis=0, keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)

    yt = []; xt = []
    for i in range(lags):
        yt.append(y_l[:,i,:])
    
    # Initial mixing using 4-layer MLP
    mixedDat = np.copy(y_l[:,:,-z_dim_list[-1]:])  # Take last layer for mixing
    for l in range(Nlayer - 1):
        mixedDat = leaky_ReLU(mixedDat, negSlope)
        mixedDat = np.dot(mixedDat, all_mixingList[0][l])
    
    x_l = np.copy(mixedDat)
    for i in range(lags):
        xt.append(x_l[:,i,:])
    
    # Layer mixing matrices (for hierarchical structure in latent space)
    layer_mixing_list = []
    for idx, z_dim in enumerate(z_dim_list):
        if idx == len(z_dim_list) - 1:
            layer_mixing_list.append(np.ones((z_dim, obs_dim)))
        else:
            layer_mixing_list.append(np.ones((z_dim, z_dim_list[idx+1])))
        
    # Time evolution with 4-layer mixing
    for i in range(length):
        # Transition function (same as original)
        y_t_noise = np.random.normal(0, noise_scale, (batch_size, latent_size)) * 2
        y_t = 0
        
        for l in range(lags):
            y_t += leaky_ReLU(np.dot(y_l[:,l,:], transitions[l]), negSlope)
        y_t = leaky_ReLU(y_t, negSlope)

        p_hist = 0.2
        y_t[:,:z_dim_list[0]] = y_t[:,:z_dim_list[0]] + 1 * y_t_noise[:,:z_dim_list[0]]
        
        for layer in range(1, n_layer):
            for j in range(z_dim_list[layer]):
                if inst:
                    y_t[:,j] = y_t[:,j] * y_t_noise[:,j] + y_t_noise[:,j]
                else:
                    y_t[:, sum(z_dim_list[:layer]) + j] = \
                    p_hist * y_t[:, sum(z_dim_list[:layer]) + j]+ \
                        (1-p_hist) * j * y_t[:, sum(z_dim_list[:layer - 1]) : sum(z_dim_list[:layer])] @ layer_mixing_list[layer-1][:, j] + 1 * y_t_noise[:,j]

        yt.append(y_t)
        
        # 4-layer MLP mixing function
        mixedDat = np.copy(y_t[:, -z_dim_list[-1]:])  # Take last layer values
        for l in range(Nlayer - 1):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            mixedDat = np.dot(mixedDat, all_mixingList[0][l])
        
        x_t = np.copy(mixedDat)
        xt.append(x_t)
        y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]

    yt = np.array(yt).transpose(1,0,2)
    xt = np.array(xt).transpose(1,0,2)
    
    np.savez(os.path.join(path, "data"), 
            yt = yt, 
            xt = xt)
    print(f"4-layer MLP dataset {name} generated: yt.shape = {yt.shape}, xt.shape = {xt.shape}")

    for l in range(lags):
        B = transitions[l]
        np.save(os.path.join(path, "W%d"%(lags-l)), B)    

def stationary_z_link_tanh_mlp(obs_dim = 4, hist = 0, inst = False, name = 'data', matrix=None, seed=None, z_dim_list=[1, 2, 4]):
    """
    Generate stationary data with tanh activation in MLP mixing procedure (z->x)
    z_dim_list: [1, 2, 4] (3 layers) but mixing uses tanh instead of leaky ReLU
    """
    lags = 1
    Nlayer = 2  # Standard 2-layer MLP but with tanh activation
    length = 6
    condList = []
    negSlope = 0.2
    latent_size = sum(z_dim_list)
    transitions = []
    noise_scale = 0.1
    batch_size = 100000
    Niter4condThresh = 1e4
    
    n_layer = len(z_dim_list)   
    
    if seed is not None:
        np.random.seed(seed)

    # Use CHiLD internal path for dataset O
    if name == 'O':
        internal_root = 'CHiLD/datasets/dataset3'
        path = os.path.join(internal_root, name)
    else:
        path = os.path.join(root_dir, name)
    os.makedirs(path, exist_ok=True)

    for i in range(int(Niter4condThresh)):
        A = np.random.uniform(1, 2, (latent_size, latent_size))
        for i in range(latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))

    condThresh = np.percentile(condList, 25)
    for l in range(lags):
        if hist == 0:
            B = generateUniformMat(latent_size, condThresh)
            transitions.append(B)
        elif hist == 1:
            assert(latent_size == 3)
            transitions.append(np.array([[0.4, 0.6, 0],
                                         [0, 1, 0],
                                         [0, 0, 1]], dtype=np.float32))
        elif hist == 2:
            if matrix is None:
                transitions.append(block_diagonal_permutation_matrix(z_dim_list))
            else:
                transitions.append(matrix)
        elif hist == 3:
            assert(latent_size == 5)
            trans_mat = np.eye(5, dtype=np.float32)
            trans_mat[1][3]+=1.0
            trans_mat[2][4]+=1.0
            transitions.append(trans_mat)
        
    transitions.reverse()

    # Create MLP mixing networks with tanh activation
    all_mixingList = []
    for _ in range(1):
        mixingList = []
        for l in range(Nlayer - 1):
            A = ortho_group.rvs(obs_dim)
            mixingList.append(A)
        all_mixingList.append(mixingList)

    y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
    y_l = (y_l - np.mean(y_l, axis=0, keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)

    yt = []; xt = []
    for i in range(lags):
        yt.append(y_l[:,i,:])
    
    # Initial mixing using tanh activation
    mixedDat = np.copy(y_l[:,:,-z_dim_list[-1]:])
    for l in range(Nlayer - 1):
        mixedDat = tanh_activation(mixedDat)  # Use tanh instead of leaky ReLU
        mixedDat = np.dot(mixedDat, all_mixingList[0][l])
    
    x_l = np.copy(mixedDat)
    for i in range(lags):
        xt.append(x_l[:,i,:])
    
    # Layer mixing matrices (for hierarchical structure in latent space)
    layer_mixing_list = []
    for idx, z_dim in enumerate(z_dim_list):
        if idx == len(z_dim_list) - 1:
            layer_mixing_list.append(np.ones((z_dim, obs_dim)))
        else:
            layer_mixing_list.append(np.ones((z_dim, z_dim_list[idx+1])))
        
    # Time evolution with tanh mixing
    for i in range(length):
        # Transition function (same as original)
        y_t_noise = np.random.normal(0, noise_scale, (batch_size, latent_size)) * 2
        y_t = 0
        
        for l in range(lags):
            y_t += leaky_ReLU(np.dot(y_l[:,l,:], transitions[l]), negSlope)
        y_t = leaky_ReLU(y_t, negSlope)

        p_hist = 0.2
        y_t[:,:z_dim_list[0]] = y_t[:,:z_dim_list[0]] + 1 * y_t_noise[:,:z_dim_list[0]]
        
        for layer in range(1, n_layer):
            for j in range(z_dim_list[layer]):
                if inst:
                    y_t[:,j] = y_t[:,j] * y_t_noise[:,j] + y_t_noise[:,j]
                else:
                    y_t[:, sum(z_dim_list[:layer]) + j] = \
                    p_hist * y_t[:, sum(z_dim_list[:layer]) + j]+ \
                        (1-p_hist) * j * y_t[:, sum(z_dim_list[:layer - 1]) : sum(z_dim_list[:layer])] @ layer_mixing_list[layer-1][:, j] + 1 * y_t_noise[:,j]

        yt.append(y_t)
        
        # MLP mixing function with tanh activation
        mixedDat = np.copy(y_t[:, -z_dim_list[-1]:])  # Take last layer values
        for l in range(Nlayer - 1):
            mixedDat = tanh_activation(mixedDat)  # Use tanh instead of leaky ReLU
            mixedDat = np.dot(mixedDat, all_mixingList[0][l])
        
        x_t = np.copy(mixedDat)
        xt.append(x_t)
        y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]

    yt = np.array(yt).transpose(1,0,2)
    xt = np.array(xt).transpose(1,0,2)
    
    np.savez(os.path.join(path, "data"), 
            yt = yt, 
            xt = xt)
    print(f"Tanh MLP dataset {name} generated: yt.shape = {yt.shape}, xt.shape = {xt.shape}")

    for l in range(lags):
        B = transitions[l]
        np.save(os.path.join(path, "W%d"%(lags-l)), B)    

def stationary_z_link(obs_dim = 4, hist = 0, inst = False, name = 'data', matrix=None, seed=None, z_dim_list=[1, 2, 4]):
    lags = 1
    Nlayer = 2
    length = 6
    condList = []
    negSlope = 0.2
    latent_size = sum(z_dim_list)
    transitions = []
    noise_scale = 0.1
    batch_size = 100000
    Niter4condThresh = 1e4
    
    n_layer = len(z_dim_list)   
    
    if seed is not None:
        np.random.seed(seed)

    path = os.path.join(root_dir, name)
    os.makedirs(path, exist_ok=True)

    for i in range(int(Niter4condThresh)):
        # A = np.random.uniform(0,1, (Ncomp, Ncomp))
        A = np.random.uniform(1, 2, (latent_size, latent_size))  # - 1
        for i in range(latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))

    condThresh = np.percentile(condList, 25)  # only accept those below 25% percentile
    for l in range(lags):

        if hist == 0:
            B = generateUniformMat(latent_size, condThresh)
            transitions.append(B)
        elif hist == 1:
            assert(latent_size == 3)
            transitions.append(np.array([[0.4, 0.6, 0],
                                         [0, 1, 0],
                                         [0, 0, 1]], dtype=np.float32))
        elif hist == 2:
            # transition_matrix = np.eye(latent_size, dtype=np.float32)
            # # transition_matrix = transition_matrix[0] * 0.1
            # transitions.append(transition_matrix)
            if matrix is None:
                transitions.append(block_diagonal_permutation_matrix(z_dim_list))
            else:
                transitions.append(matrix)
        elif hist == 3:
            assert(latent_size == 5)
            trans_mat = np.eye(5, dtype=np.float32)
            trans_mat[1][3]+=1.0
            trans_mat[2][4]+=1.0
            transitions.append(trans_mat)
        
    transitions.reverse()

    all_mixingList = []
    for _ in range(n_layer):
        mixingList = []
        for l in range(Nlayer - 1):
            # generate causal matrix first:
            # A = ortho_group.rvs(latent_size)  # generateUniformMat(Ncomp, condThresh)
            A = ortho_group.rvs(obs_dim)  # generateUniformMat(Ncomp, condThresh)
            mixingList.append(A)
        all_mixingList.append(mixingList)

    y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
    y_l = (y_l - np.mean(y_l, axis=0, keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)

    yt = []; xt = []
    for i in range(lags):
        yt.append(y_l[:,i,:])
    # mixedDat = np.copy(y_l)
    mixedDat = np.copy(y_l[:,:,-z_dim_list[-1]:])
    # print(mixedDat.shape)
    # exit()
        
    # also generate with all mixingList, then concat them
    all_mixedDat = []
    for i in range(1):        
        mixedDat = np.copy(y_l[:,:,-z_dim_list[-1]:])
        for l in range(Nlayer - 1):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            mixedDat = np.dot(mixedDat, all_mixingList[i][l])
        
        all_mixedDat.append(mixedDat)
    all_mixedDat = np.concatenate(all_mixedDat, axis=-1)
    x_l = np.copy(all_mixedDat)
    for i in range(lags):
        xt.append(x_l[:,i,:])
    
    # randomly generate a permutation matrix
    # 1->2->4
    layer_mixing_list = []
    for idx, z_dim in enumerate(z_dim_list):
        if idx == len(z_dim_list) - 1:
            # make it is ones but not gaussian
            layer_mixing_list.append(np.ones((z_dim, obs_dim)))
            #layer_mixing_list[-1][np.random.rand(*layer_mixing_list[-1].shape) < 0.7] = 0
        else:
            # make it is ones but not gaussian
            layer_mixing_list.append(np.ones((z_dim, z_dim_list[idx+1])))
            # layer_mixing_list[-1][np.random.rand(*layer_mixing_list[-1].shape) < 0.7] = 0
        
    # Mixing function
    for i in range(length):
        # Transition function
        y_t_noise = np.random.normal(0, noise_scale, (batch_size, latent_size)) * 2
        x_t_noise = np.random.normal(0, noise_scale, (batch_size, latent_size))
        # Modulate the noise scale with averaged history
        # y_t_noise = y_t_noise * np.mean(y_l, axis=1)
        y_t = 0
        # transition
        for l in range(lags):
            y_t += leaky_ReLU(np.dot(y_l[:,l,:], transitions[l]), negSlope)
        y_t = leaky_ReLU(y_t, negSlope)


        p_hist = 0.2 # the weight of history data
        y_t[:,:z_dim_list[0]] = y_t[:,:z_dim_list[0]] + 1 * y_t_noise[:,:z_dim_list[0]]  # first layer variable is related to the 
        # y_t[:,0] = y_t[:,0] + 1 * y_t_noise[:,0]  # first layer variable is related to the 
        
        for layer in range(1, n_layer):
            for i in range(z_dim_list[layer]):
                if inst:
                    y_t[:,i] = y_t[:,i] * y_t_noise[:,i] + y_t_noise[:,i]
                else:
                    # y_t[:,i] = (p_hist * y_t[:,i] + (1-p_hist) * y_t[:,i-1]) * y_t_noise[:,i] + y_t_noise[:,i]
                    y_t[:, sum(z_dim_list[:layer]) + i] = \
                    p_hist * y_t[:, sum(z_dim_list[:layer]) + i]+ \
                        (1-p_hist) * i * y_t[:, sum(z_dim_list[:layer - 1]) : sum(z_dim_list[:layer])] @ layer_mixing_list[layer-1][:, i] + 1 * y_t_noise[:,i]

        yt.append(y_t)
        # Mixing function
        all_mixedDat = []   
        for k in range(1):
            mixedDat = np.copy(y_t[:, -z_dim_list[-1]:])
            for l in range(Nlayer - 1):
                mixedDat = leaky_ReLU(mixedDat, negSlope) # + x_t_noise * 0.05
                mixedDat = np.dot(mixedDat, all_mixingList[k][l])
            all_mixedDat.append(mixedDat)
        all_mixedDat = np.concatenate(all_mixedDat, axis=-1)    
        x_t = np.copy(all_mixedDat)
        xt.append(x_t)
        y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]

    yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2)
    
    np.savez(os.path.join(path, "data"), 
            yt = yt, 
            xt = xt)
    print(yt.shape)
    print(xt.shape)

    for l in range(lags):
        B = transitions[l]
        np.save(os.path.join(path, "W%d"%(lags-l)), B)    

def generate_5x5_matrices():
    """
    生成所有满足以下条件的 5x5 二进制矩阵（numpy.ndarray）：
      1. 第 0 行全为 1
      2. 第 0 列全为 1
      3. 对于其余行和列，每行/列只包含 2 个 1
         （其中一个在第 0 列/行，另一个在子矩阵 4x4 内）
    返回：含有所有有效矩阵的列表
    """
    results = []
    # 遍历所有长度为 4 的排列
    for perm in permutations(range(4)):
        # 创建一个 5x5 的零矩阵
        mat = np.zeros((5, 5), dtype=int)
        
        # 第 0 行全部置为 1
        mat[0, :] = 0
        # 第 0 列全部置为 1
        mat[:, 0] = 0
        mat[0,0] = 1
        
        # 在 mat[1..4, 1..4] 区域内放置置换矩阵：
        # perm[i] 表示第 i 行 (在子矩阵中的第 i 行) 的 1 放在第 perm[i] 列
        for i in range(4):
            mat[i + 1, perm[i] + 1] = 1
        
        results.append(mat)
        
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=773)
    args = parser.parse_args()
    stationary_z_link(4, 2, False, 'A', seed=772, z_dim_list=[1, 2, 4])
    stationary_z_link(8, 2, False, 'G', seed=772, z_dim_list=[8, 8, 8])
    stationary_z_link(4, 2, False, 'H', seed=772, z_dim_list=[4, 4, 4])
    stationary_z_link(10, 2, False, f'I_{args.seed}', seed=args.seed, z_dim_list=[10, 10, 10])
    stationary_z_link(4, 2, False, 'L', seed=772, z_dim_list=[1, 2, 4]) # 3-layer MLP
    stationary_z_link_hierarchical(4, 2, False, 'M', seed=772, z_dim_list=[1, 2, 4]) # hierarchical with top->bottom connection
    stationary_z_link_4layer_mlp(4, 2, False, 'N', seed=772, z_dim_list=[1, 2, 4]) # 4-layer MLP for mixing
    stationary_z_link_tanh_mlp(4, 2, False, 'O', seed=772, z_dim_list=[1, 2, 4]) # MLP with tanh activation
    stationary_z_link_temporal(4, 2, False, 'P', seed=772, z_dim_list=[1, 2, 4]) # temporal connection: top layer t-1 -> down layer t
    # stationary_z_link(5, 2, False, 'B')
    # stationary_z_link(8, 2, False, 'C')
    # stationary_z_link(8, 0, False, 'D')
    # stationary_z_link(8, 0, True, 'E')
    # stationary_z_link(16, 2, True, 'F')
