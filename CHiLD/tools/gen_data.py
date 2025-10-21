import os
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
from CHiLD.tools.utils import create_sparse_transitions, controlable_sparse_transitions

VALIDATION_RATIO = 0.2
root_dir = '../datasets/dataset'
standard_scaler = preprocessing.StandardScaler()

def leaky_ReLU_1d(d, negSlope):
    if d > 0:
        return d
    else:
        return d * negSlope

leaky1d = np.vectorize(leaky_ReLU_1d)

def leaky_ReLU(D, negSlope):
    assert negSlope > 0
    return leaky1d(D, negSlope)

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


def stationary_z_link(z_dim = 5):
    lags = 1
    Nlayer = 2
    length = 5
    condList = []
    negSlope = 0.2
    latent_size = z_dim
    transitions = []
    noise_scale = 0.1
    batch_size = 100000
    Niter4condThresh = 1e4

    path = os.path.join(root_dir, f"d{z_dim}")
    os.makedirs(path, exist_ok=True)

    for i in range(int(Niter4condThresh)):
        # A = np.random.uniform(0,1, (Ncomp, Ncomp))
        A = np.random.uniform(1, 2, (latent_size, latent_size))  # - 1
        for i in range(latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))

    condThresh = np.percentile(condList, 25)  # only accept those below 25% percentile
    for l in range(lags):
        # full
        B = generateUniformMat(latent_size, condThresh)
        transitions.append(B)

        # transitions.append(np.array([[0.4, 0.6, 0],
        #                              [0, 1, 0],
        #                              [0, 0, 1]], dtype=np.float32))
        
        # extreme sparse
        # transitions.append(np.eye(latent_size, dtype=np.float32))
        
        # half half dense
        # eye_mat = np.eye(latent_size//2, dtype=np.float32)
        # dense_mat = generateUniformMat(latent_size//2, condThresh)
        # B = np.zeros((latent_size, latent_size), dtype=np.float32)
        # B[:latent_size//2, :latent_size//2] = eye_mat
        # B[latent_size//2:, latent_size//2:] = dense_mat
        # transitions.append(B)
        
        # half half
        eye_mat = np.eye(latent_size//2, dtype=np.float32)
        dense_mat = generateUniformMat(latent_size//2, condThresh)
        dense_mat = (dense_mat - dense_mat.min()) / (dense_mat.max() - dense_mat.min())
        dense_mat = dense_mat / dense_mat.sum() * 4
        B = np.zeros((latent_size, latent_size), dtype=np.float32)
        B[:latent_size//2, :latent_size//2] = eye_mat
        B[latent_size//2:, latent_size//2:] = dense_mat
        transitions.append(B)

        # import pdb
        # pdb.set_trace()

        # # eye+
        # transitions.append(np.array([[1, 0, 0, 0],
        #                                 [0, 1, 0, 0],
        #                                 [0, 0, 1, 0],
        #                                 [0, 0, 1, 1]], dtype=np.float32))
        # trans_mat = np.eye(5, dtype=np.float32)
        # trans_mat[1][3]+=1.0
        # trans_mat[2][4]+=1.0
        # transitions.append(trans_mat)
        
    transitions.reverse()

    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        A = ortho_group.rvs(latent_size)  # generateUniformMat(Ncomp, condThresh)
        mixingList.append(A)

    y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
    y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)

    yt = []; xt = []
    for i in range(lags):
        yt.append(y_l[:,i,:])
    mixedDat = np.copy(y_l)
    for l in range(Nlayer - 1):
        mixedDat = leaky_ReLU(mixedDat, negSlope)
        mixedDat = np.dot(mixedDat, mixingList[l])
    x_l = np.copy(mixedDat)
    for i in range(lags):
        xt.append(x_l[:,i,:])
        
    # Mixing function
    for i in range(length):
        # Transition function
        y_t_noise = np.random.normal(0, noise_scale, (batch_size, latent_size)) * 2
        # Modulate the noise scale with averaged history
        # y_t_noise = y_t_noise * np.mean(y_l, axis=1)
        y_t = 0
        # transition
        for l in range(lags):
            y_t += leaky_ReLU(np.dot(y_l[:,l,:], transitions[l]), negSlope)
        y_t = leaky_ReLU(y_t, negSlope)

        # add your causal graph here
        # p_hist = 0.2 # the weight of history data
        # y_t[:,0] = y_t[:,0] * y_t_noise[:,0] + y_t_noise[:,0]
        # for i in range(1, latent_size):
        #     y_t[:,i] = (p_hist * y_t[:,i] + (1-p_hist) * y_t[:,i-1]) * y_t_noise[:,i] + y_t_noise[:,i]

        p_hist = 0.2 # the weight of history data
        y_t[:,0] = y_t[:,0] * y_t_noise[:,0] + y_t_noise[:,0]
        for i in range(1, latent_size//2):
            y_t[:,i] = (p_hist * y_t[:,i] + (1-p_hist) * y_t[:,i-1]) * y_t_noise[:,i] + y_t_noise[:,i]


        yt.append(y_t)
        # Mixing function
        mixedDat = np.copy(y_t)
        for l in range(Nlayer - 1):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            mixedDat = np.dot(mixedDat, mixingList[l])
        x_t = np.copy(mixedDat)
        xt.append(x_t)
        y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]

    yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2)

    np.savez(os.path.join(path, "data"), 
            yt = yt, 
            xt = xt)

    for l in range(lags):
        B = transitions[l]
        np.save(os.path.join(path, "W%d"%(lags-l)), B)    

if __name__ == "__main__":
    stationary_z_link(8)
    # stationary_z_link(10)
    # stationary_z_link(15)
