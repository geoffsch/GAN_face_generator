"""
Basic helper utils
"""

import os, sys
from warnings import warn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


################################################################################
### Global variables
################################################################################


# Directories for intermediate images and model files created during training
INTER_IMG_DIR = os.path.join(os.path.dirname(__file__), "../images/intermed_training/")
INTER_MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/")
HISTORY_DIR = os.path.join(os.path.dirname(__file__), "../data/")


################################################################################
### Plotting
################################################################################


def plot_images(images, grid_shape=(10, 10), rescale=False, figsize='default'):
    """Displays images in specified grid shape
    If rescale, assumes image pixels range [-1,1] and rescales to [0,1]
    """
    if isinstance(grid_shape, (int, float)):
        grid_shape = (grid_shape, grid_shape)
    if figsize == 'default':
        figsize = (np.clip(2*grid_shape[1], 2, 16), np.clip(2*grid_shape[0], 2, 16))
    n_images = grid_shape[0] * grid_shape[1]

    if len(images) < n_images:
        warn(f"Insufficient number of images for {grid_shape[0]} x {grid_shape[1]} plot - will plot all {len(images)} provided images")
        img_sample = images.copy()
    else:
        img_sample = images[:n_images].copy()

    if rescale:
        img_sample = (img_sample + 1) / 2

    fig = plt.figure(figsize=figsize)
    for i, img in enumerate(img_sample):
        ax = fig.add_subplot(*grid_shape, 1+i) # new subplot axis
        ax.imshow(img) # plot raw pixel data
        ax.axis('off') # turn off axes
    return fig


def disp_intermed_img(epoch, figsize=(16, 16)):
    """Displays image saved out for specified epoch
    """
    img_file = name_inter_img_file(epoch)
    img = mpimg.imread(img_file)
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


################################################################################
### File I/O
################################################################################


def retrieve_prep_images(file, scale=False):
    """Load and optionally scale images from file
    (as output by prep_data.load_celeba_faces)
    If scale, rescales pixel values from 0,255 to -1,1

    This is the function called by model to retrieve data for training
    """
    data = np.load(file)
    X = data['arr_0']

    if scale:
        X = X.astype('float32')
        X = (X - 127.5) / 127.5 # scale from [0,255] to [-1,1]

    return X


def name_history_file(suffix=None, savedir=HISTORY_DIR):
    """Create filename for saved training history
    """
    os.makedirs(savedir, exist_ok=True)
    fname = savedir.rstrip('/') + '/' + 'training_history.csv'
    if suffix is not None:
        suffix = '_' + str(suffix).lstrip('_')
        fname = fname.replace('.csv', f'{suffix}.csv')
    return fname


def name_inter_img_file(epoch, savedir=INTER_IMG_DIR):
    """Create filename for intermediate saved image
    """
    if not isinstance(epoch, str):
        epoch = f"e{epoch:03d}"
    os.makedirs(savedir, exist_ok=True)
    return savedir.rstrip('/') + '/' + f"generated_plot_{epoch}.png"


def name_inter_model_file(epoch, savedir=INTER_MODEL_DIR):
    """Create filename for intermediate saved model
    """
    if not isinstance(epoch, str):
        epoch = f"e{epoch:03d}"
    os.makedirs(savedir, exist_ok=True)
    return savedir.rstrip('/') + '/' + f"generator_model_{epoch}.h5"

