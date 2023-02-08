import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

import gmm

def get_marker_size():
    return 20

def get_marker_edge_width():
    return 2

def get_line_width():
    return 2

def get_font_size():
    return 24

def get_tick_font_size():
    return 18

def save_mnist_kmeans_centers(C):
    figures_directory = 'figures'

    os.makedirs(figures_directory, exist_ok=True)
    
    K = C.shape[0]
    for k in range(K):
        plt.imshow(C[k].reshape((28,28)))
        plt.savefig('{}/kmeans_K_{}_cluster_{}'.format(figures_directory, K, k))
        plt.clf()

def plot_gmm(X, mus, Sigmas, title=None, colors=None,
        new_figure=True, show_figure=True, save_filename=None):
    """
    Plots the points X and the contours of all 2_D Gaussians

    X: numpy ndarray of size (N, 2) where each row is a datapoint
    mus: list of K numpy ndarray of size (2,) representing the mean for each cluster
    Sigmas: list of K numpy nd array of size (2, 2) representing the covariance matrix 
        each cluster
    title (default=None): Title of plot if not None
    colors (default=None): Color of contour lines. None will use default cmap.
    new_figure (default=True): If true, calls plt.figure(), which create a 
        figure. If false, it will modify an existing figure (if one exists).
    show_figure (default=True): If true, calls plt.show(), which will open
        a new window and block program execution until that window is closed
    save_filename (defalut=None): If not None, save figure to save_filename 
    """
    if new_figure:
        plt.figure(figsize=(8,8))

    plt.plot(X[:, 0], X[:, 1], 'o', fillstyle='none')
    plt.axis('equal')

    for mu, Sigma in zip(mus, Sigmas):
        plot_gaussian_contours(mu, Sigma,
                x1_min=None, x1_max=None, x2_min=None, x2_max=None,  
                new_figure=False, show_figure=False, save_filename=None)
 
    if new_figure:
        plt.tick_params(labelsize=get_tick_font_size())

        ax = plt.gca()
        ax.axhline(0, color='lightgray')
        plt.axvline(0, color='lightgray')
        ax.set_axisbelow(True)

    if title is not None:
        plt.title(title)

    if save_filename is not None:
        plt.savefig(save_filename)

    if show_figure:
        plt.show()

def plot_gaussian_contours(mu, Sigma,
        x1_min=-8, x1_max=8, x2_min=-8, x2_max=8, title=None, colors=None,
        new_figure=True, show_figure=True, save_filename=None):
    """
    Plots the contours of a 2_D Gaussian pdf with parameters mu and Sigma

    mu: numpy ndarray of size (2,) representing the mean
    Sigma: numpy nd array of size (2, 2) representing the covariance matrix
    x1_min (default=None): Minimum of axes[0] range
    x1_max (default=None): Maximum of axes[0] range
    x2_min (default=None): Minimum of axes[1] range
    x2_max (default=None): Maximum of axes[2] range
    title (default=None): Title of plot if not None
    colors (default=None): Color of contour lines. None will use default cmap.
    new_figure (default=True): If true, calls plt.figure(), which create a 
        figure. If false, it will modify an existing figure (if one exists).
    show_figure (default=True): If true, calls plt.show(), which will open
        a new window and block program execution until that window is closed
    save_filename (defalut=None): If not None, save figure to save_filename 
    """
    if x1_min is None or x1_max is None and not new_figure:
        x1_min, x1_max = plt.xlim()
    if x2_min is None or x2_max is None and not new_figure:
        x2_min, x2_max = plt.ylim()

    N = 101
    
    x1 = np.linspace(x1_min, x1_max, N)
    x2 = np.linspace(x2_min, x2_max, N)
    X1, X2 = np.meshgrid(x1, x2)
    
    Y = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            x = np.array([X1[i,j], X2[i,j]])
            Y[i, j] = gmm.gaussian_pdf(x, mu, Sigma)
    
    # Ploting contour 

    if new_figure:
        plt.figure(figsize=(8,8))

    ax = plt.gca()
    contour_plot = ax.contour(X1, X2, Y, colors=colors)

    if new_figure:
        plt.tick_params(labelsize=get_tick_font_size())

        ax.axhline(0, color='lightgray')
        plt.axvline(0, color='lightgray')
        ax.set_axisbelow(True)

    if title is not None:
        plt.title(title)

    if save_filename is not None:
        plt.savefig(save_filename)

    if show_figure:
        plt.show()

