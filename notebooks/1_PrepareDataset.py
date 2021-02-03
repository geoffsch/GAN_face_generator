
# coding: utf-8

# # Prepare Dataset
# 
# - Load and preprocess images
# - Inspect data
# - Save to numpy array

# ## SETUP

# In[1]:


### Environment: conda_gan-tutorial

# Extensions & config
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '1')

# General package imports
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Flint imports
from flint import style

# src imports
src_dir = os.path.join(os.getcwd(), "../src")
sys.path.append(src_dir)
get_ipython().run_line_magic('aimport', 'utils, prep_data')


# In[2]:


### Variables

directory = "../data/celeba_dataset/img_align_celeba/img_align_celeba/" # run scripts/get_celeba_data.sh first if necessary
n_images = 50_000
#n_images = 500


# In[3]:


### Set Masternaut default style settings

style.set_mpl_style(background='light') # Optional args, background='dark', orientation='horizontal'

# For horizontal plots use orientation argument in set_mpl_style, or context manager e.g.,
# with style.mpl_horizontal():


# ## LOAD AND INSPECT IMAGES FROM FILE

# In[4]:


# Load and display some faces, to 4D numpy array (n_img, height, width, channels)

faces = prep_data.load_images_from_dir(directory, n_images)
print(f"Loaded {n_images} images from {directory}\nResults array shape (n_img x H x W x channels):", faces.shape)


# In[5]:


# Display them

fig = utils.plot_images(faces)
plt.show()


# ## LOAD IMAGES, DETECT AND EXTRACT FACES, SAVE TO FILE

# In[6]:


# Save detected faces to numpy compressed file

file_out = "../data/celeba_dataset/img_align_celeba.npz" # set to None to get back faces array instead
#file_out = None

faces = prep_data.load_celeba_faces(n_images, file_out)


# ## RECOVER IMAGE ARRAYS FROM FILE

# In[9]:


# Recover image arrays from compressed numpy file

if file_out is not None:
    faces = utils.retrieve_prep_images(file_out, scale=False)
    print(f"Loaded {len(faces)} faces from file {file_out}, data shape:", faces.shape)


# In[10]:


# Display them

fig = utils.plot_images(faces)
plt.show()


# ## OLD / BACKUP CELLS

# In[ ]:


raise RuntimeError('Stop here, dont run old cells')

