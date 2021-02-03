#!/usr/bin/env python
# coding: utf-8

# # Train GAN
# 
# - Define and compile discriminator network
# - Define and compile generator network
# - Combine into GAN and train
# - Save images, logs, model artefact

# ## SETUP

# In[1]:


### Environment: conda_gan-tutorial

# Extensions & config
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '1')

# General package imports
import os, sys
from warnings import warn
from multiprocessing import cpu_count
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

# Flint imports
from flint import style

# src imports
src_dir = os.path.join(os.getcwd(), "../src")
sys.path.append(src_dir)
get_ipython().run_line_magic('aimport', 'utils, model')


# In[2]:


### Variables

prepped_data_file = "../data/celeba_dataset/img_align_celeba.npz" # produced by notebook 1

latent_dim = model.LATENT_DIM_DEF


# In[3]:


### Set Masternaut default style settings

style.set_mpl_style(background='light') # Optional args, background='dark', orientation='horizontal'

# For horizontal plots use orientation argument in set_mpl_style, or context manager e.g.,
# with style.mpl_horizontal():


# In[4]:


### Check hardware

gpus = tf.config.experimental.list_logical_devices('GPU')

if len(gpus) > 1:
    print(f"Detected {len(gpus)} GPUs - may need to set up a strategy to actually use these?")
elif len(gpus) == 1:
    print("Running on single GPU - this should happen automatically")
else:
    warn("No GPU detected, training will be slow")


# ## DEFINE MODELS
# Separate generator and discriminator networks, with combined GAN

# In[5]:


# Define stand-alone discriminator network

d_model = model.define_discriminator()
d_model.summary()


# In[6]:


# Define the standalone generator model

g_model = model.define_generator(latent_dim)
g_model.summary()


# In[7]:


# Define the combined generator and discriminator model, for updating the generator

gan_model = model.define_gan(g_model, d_model)
gan_model.summary()


# ## TRAIN MODEL

# In[11]:


# Retrieve set of real images produced by notebook 1

max_images = 10_000

real_img_dataset = utils.retrieve_prep_images(prepped_data_file, scale=True)

if len(real_img_dataset) > max_images:
    real_img_dataset = real_img_dataset[:int(max_images)]

print(f"Loaded {len(real_img_dataset)} images from file {prepped_data_file}, data shape:", real_img_dataset.shape)


# In[12]:


# Example - plot random selection of real images

grid_shape = (3, 3)
n_samples = grid_shape[0] * grid_shape[1]

X_real, y_real = model.generate_real_samples(real_img_dataset, n_samples, seed=42)
print("Data shape:", X_real.shape)
print("Label shape:", y_real.shape)

fig = utils.plot_images(X_real, grid_shape=grid_shape, rescale=True, figsize=(6, 6))
plt.show()


# In[13]:


# Example - generate fake images with generator

grid_shape = (3, 3)
n_samples = grid_shape[0] * grid_shape[1]

X_fake, y_fake = model.generate_fake_samples(g_model, n_samples=n_samples, latent_dim=latent_dim, seed=42)
print("Data shape:", X_fake.shape)
print("Label shape:", y_fake.shape)

fig = utils.plot_images(X_fake, grid_shape=grid_shape, rescale=True, figsize=(6, 6))
plt.show()


# In[ ]:


# Train model

loss_data, acc_data = model.train(g_model, d_model, gan_model,
                              dataset=real_img_dataset, n_epochs=50, latent_dim=latent_dim)


# ---
# 
# ### Notes on training time/memory requirements:
# - On t3a.small:
#     - 50 images (16 per batch), 10 epochs: **5 min 11 sec, ~1.3GB RAM**
# - On g4dn.2xlarge (prev-gen GPU):
#     - 50 images (16 per batch), 10 epochs: **51 sec, ~1.3GB RAM**
#     - 500 images (def 128 per batch), 10 epochs: **7 min 58 sec, ~1.6GB RAM**
#     - 10k images (def 128 per batch), 50 epochs: **12 hr 33 min 7 sec**
#     
# - Attempts to get GPU working properly on g4dn.2xlarge, tests with 50 images (16/batch) & 10 epochs:
#     - Run blindly after installing tensorflow-gpu and cudatoolkit: **1 min 6 sec**
#         - tf.config.list_physical_devices() does NOT show a GPU
#     - Run blindly after installing ubuntu cuda drivers: **1 min 9 sec**
#         - (but now tf.config.list_physical_devices() does show a GPU)
#     - Run after the below commands: **1 min 11 sec**
#         - config = tf.compat.v1.ConfigProto(device_count={'GPU': 1, 'CPU': 8})
#         - sess = tf.compat.v1.Session(config=config)
#         - tf.compat.v1.keras.backend.set_session(sess)
#     - Run on new AWS DLAMI instance on conda env copied from built-in tensorflow2_latest_p37: **11 sec**
#          - no additional config, session, or setup commands (as above)
#          - GPU shown in both tf.config.list_physical_devices() and tf.config.experimental.list_logical_devices()
# 
# ---

# In[ ]:


# Save final models - may be unnecessary as they've been saved throughout

final_fname = utils.name_inter_model_file(epoch='final')

g_model.save(final_fname)
d_model.save(final_fname.replace('generator', 'discriminator'))


# ## EXPLORE TRAINING RESULTS

# In[18]:


# Plot training history - loss and accuracy

loss_df = pd.read_csv(utils.name_history_file('loss'))
acc_df = pd.read_csv(utils.name_history_file('_acc'))

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 8))

loss_df['fract_epoch'] = loss_df.epoch + (loss_df.batch - 1)/loss_df.batch.max()  - 1
loss_df.set_index('fract_epoch').drop(columns=['epoch', 'batch']).plot(ax=axes[0], secondary_y='g_loss')
axes[0].set_ylim(bottom=0)
axes[0].right_ax.set_ylim(bottom=0)
axes[0].set_ylabel("Discriminator Loss (Batch-Level)")
axes[0].right_ax.set_ylabel("Generator Loss (Batch-Level)")
axes[0].grid('x')
axes[0].right_ax.grid(False)

acc_df.set_index('epoch').plot(ax=axes[1], marker='.')
axes[1].set_ylim(0, 1)
axes[1].set_ylabel("Discriminator Accuracy")
axes[1].set_xlabel("Epoch")

fig.tight_layout()
fig.suptitle("Training History")
plt.show()


# In[15]:


# Look at generated images

utils.disp_intermed_img(epoch=50)


# ## OLD / BACKUP CELLS

# In[14]:


raise RuntimeError('Stop here, dont run old cells')

