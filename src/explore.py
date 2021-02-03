"""
Exploring trained generator model
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

# flint imports

# src imports
import utils


################################################################################
### Interrogate latent space
################################################################################


def interpolate_points(p1, p2, n_steps=10, method='linear'):
    """Uniform interpolation between two points
    Returns array of shape (n_steps, <p1/p2 shape>)
    """
    if method.startswith('line'):
        func = linear_interp
    elif method.startswith('spher'):
        func = slerp
    else:
        raise ValueError("Argument 'method' must be 'linear' or 'spherical'")

    ratios = np.linspace(0, 1, num=n_steps)
    vectors = [func(r, p1, p2) for r in ratios]
    return np.asarray(vectors)


def append_avg_vector(vectors):
    """Returns a new array of vectors with the average appended
    """
    avg_vector = np.mean(vectors, axis=0)
    return np.vstack((vectors, np.expand_dims(avg_vector, axis=0)))


def plot_representative_img_set(img_set, label=None, annot_avg=True):
    """Plots selected representative images - assumes average is present
    """
    fig = utils.plot_images(img_set, grid_shape=(1, len(img_set)), rescale=True, figsize=(12, 3))
    axes = fig.axes
    if annot_avg:
        axes[-1].text(0.5, 0, 'avg', transform=axes[-1].transAxes, ha='center', va='top')
    if label is not None:
        axes[0].text(0, 0.5, label, transform=axes[0].transAxes, ha='right', va='center', rotation=90)
    return fig


################################################################################
### Helper utils
################################################################################


def linear_interp(val, lo, hi):
    """Linear interpolation between lo and hi values
    """
    return (1 - val) * lo + val * hi


def slerp(val, lo, hi):
    """Spherical linear interpolation between lo and hi values
    """
    lo_norm = lo / np.linalg.norm(lo)
    hi_norm = hi / np.linalg.norm(hi)
    omega = np.arccos(np.clip(np.dot(lo_norm, hi_norm), -1, 1)) # angle between lo and hi
    so = np.sin(omega)
    if so == 0:
        return linear_interp(val, lo, hi) # L'Hopital's rule/LERP
    return (np.sin((1 - val) * omega) / so) * lo + (np.sin(val * omega) / so) * hi

