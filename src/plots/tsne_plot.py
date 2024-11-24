from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE_Multicore
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import cv2
import glob
import numpy as np
import pandas as pd
import psutil
import os
from tqdm import tqdm


def plot_2d_scatterplot(results, ground_truth, num_classes=2, alpha=0.3, display=False, save_plt_path="./test.png"):
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x=results[:,0], y=results[:,1],
        hue=ground_truth,
        palette=sns.color_palette("hls", num_classes),
        data=pd.DataFrame(results[:,:2]),
        legend="full",
        alpha=alpha
    )
    if display:
        plt.show()
    if not os.path.exists(save_plt_path[:-len(save_plt_path.split('/')[-1])]):
        os.makedirs(save_plt_path[:-len(save_plt_path.split('/')[-1])])
    plt.savefig(save_plt_path)
    return

def plot_3d_scatterplot(results, ground_truth, reduction_algo="pca", display=False, save_plt_path="./test.png"):
    ax = plt.figure(figsize=(16,10)).add_subplot(1,1,1,projection='3d')
    ax.scatter(
        xs=results[:,0], 
        ys=results[:,1], 
        zs=results[:,2], 
        c=ground_truth, 
        cmap='tab10',
    )
    ax.set_xlabel(reduction_algo + '-one')
    ax.set_ylabel(reduction_algo + '-two')
    ax.set_zlabel(reduction_algo + '-three')
    ax.legend()
    if display:
        plt.show()
    plt.savefig(save_plt_path)
    return

def compute_tsne(features, n_components=2, verbose=1, perplexity=5, n_iter=5000, n_jobs=4):
    # tsne = TSNE(n_components=n_components, verbose=verbose, perplexity=perplexity, n_iter=n_iter)
    tsne = TSNE_Multicore(n_jobs=n_jobs, n_components=n_components, verbose=verbose, perplexity=perplexity, n_iter=n_iter)
    tsne_results = tsne.fit_transform(features)
    return tsne_results

def compute_PCA(features, n_components=2):
    pca = PCA(n_components=n_components)
    pca_results = pca.fit_transform(features)
    return pca_results
