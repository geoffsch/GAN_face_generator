"""
GAN definition and training
"""

import os, sys
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout

# flint imports

# src imports
import utils


################################################################################
### Global variables
################################################################################


LATENT_DIM_DEF = 100


################################################################################
### Model definitions
################################################################################


def define_discriminator(in_shape=(80,80,3), n_kernels=128, k_size=5, adam_kwds={}):
    """Define and compile stand-alone discriminator network
    
    Takes an image as input and produces a binary prediction (1=real, 0=fake)
    Downsampling done with stride 2 (reduces dimensions by half each time)
    
    Arguments:
        in_shape = shape of input images (RGB channels assumed to be last dimension)
        n_kernels = int, number of kernels per convolutional layer
        k_size = int, number of pixels per side of kernel
        adam_kwds = keyword args for Adam optimiser
        
    Returns:
        model = compiled keras model
    """
    n_kernels = int(n_kernels)
    k_size = int(k_size)
    
    model = Sequential(name='discriminator')

    # First layer output, same size as input (80x80 pixels by default)
    model.add(Conv2D(n_kernels, (k_size,k_size), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(n_kernels, (k_size,k_size), strides=(2,2), padding='same')) # downsample to 40x40 (or half of in_shape)
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(n_kernels, (k_size,k_size), strides=(2,2), padding='same')) # downsample to 20x20
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(n_kernels, (k_size,k_size), strides=(2,2), padding='same')) # downsample to 10x10
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(n_kernels, (k_size,k_size), strides=(2,2), padding='same')) # downsample to 5x5
    model.add(LeakyReLU(alpha=0.2))
    
    # classifier
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid')) # binary prediction
    
    # compile model
    opt_kwds = {'lr': 0.0002, 'beta_1': 0.5, **adam_kwds}
    opt = Adam(**opt_kwds)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model


def define_generator(latent_dim=LATENT_DIM_DEF, n_kernels=128, k_size=5):
    """Define stand-alone generator network
    
    Takes a point in latent space as input and produces an 80x80 colour image
    Upsampling done with stride 2 and transpose layers (doubles dimension each time)
    Transpose conv layers done with filter size double stride

    Stand-alone generator is NOT compiled, as it is trained as part of combined GAN
    
    Arguments:
        latent_dim = int, dimensionality of latent space
        n_kernels = int, number of kernels per convolutional layer
        k_size = int, number of pixels per side of kernel
        
    Returns:
        model = defined (not compiled) keras model
    """
    latent_dim = int(latent_dim)
    n_kernels = int(n_kernels)
    k_size = int(k_size)
    
    n_nodes = n_kernels * k_size * k_size # foundation for k_size x k_size feature maps
    
    model = Sequential(name='generator')
    
    # Reshape latent space point to initial geometry specified by k_size and n_kernels 
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((k_size, k_size, n_kernels)))
    
    model.add(Conv2DTranspose(n_kernels, (4,4), strides=(2,2), padding='same')) # upsample to 10x10 (or double k_size)
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(n_kernels, (4,4), strides=(2,2), padding='same')) # 20x20
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(n_kernels, (4,4), strides=(2,2), padding='same')) # 40x40
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(n_kernels, (4,4), strides=(2,2), padding='same')) # 80x80
    model.add(LeakyReLU(alpha=0.2))
    
    # Output layer 80x80x3
    model.add(Conv2D(3, (k_size,k_size), activation='tanh', padding='same'))
    
    return model


def define_gan(g_model, d_model, adam_kwds={}):
    """Define and compile combined GAN from standalone generator and discriminator
    
    Takes as input a point in latent space, uses generator to create an image,
    which is classified as real or fake by discriminator.
    
    Will be used to train generator parameters, using output/error calculated by discriminator;
    discriminator parameters are fixed (trained separately).
    
    Arguments:
        g_model = generator model, output by define_generator
        d_model = discriminator model, output by define_discriminator
        adam_kwds = keyword args for Adam optimiser
        
    Returns:
        model = compiled keras model
    """
    d_model.trainable = False # fix weights in discriminator
    
    # Connected model
    model = Sequential(name='GAN')
    model.add(g_model)
    model.add(d_model)
    
    # Compile model
    opt_kwds = {'lr': 0.0002, 'beta_1': 0.5, **adam_kwds}
    opt = Adam(**opt_kwds)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


################################################################################
### Model training
################################################################################


def train(g_model, d_model, gan_model, dataset,
    latent_dim=LATENT_DIM_DEF, n_epochs=100, n_batch=128):
    """Train the generator and discriminator in alternating fashion
    
    For each batch in each epoch:
        - update discriminator for half batch of real samples
        - update discriminator for half batch of fake samples
        - update generator via combined GAN model, with fake samples with inverted labels
    
    Arguments:
        g_model = generator model, output by define_generator
        d_model = discriminator model, output by define_discriminator
        gan_model = gan model, output by define_gan
        dataset = input data array, (n_samples, height_pixels, width_pixels, n_channels)
        latent_dim = int, dimensionality of latent space
        n_epochs = int, number of training epochs
        n_batch = int, batch size
        
    Returns (loss_records, acc_records), training history, tuple of record lists
        - loss at every batch trained, accuracy after each epoch
    """
    start = dt.datetime.now()
    log_print("BEGIN GAN TRAINING", highlight=True)
    
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)

    bat_digs = int(np.log10(bat_per_epo)) + 1
    
    loss_records = []
    acc_records = []
    for epoch in range(1, n_epochs+1): # loop over epochs
        for batch in range(1, bat_per_epo+1): # loop over batches
            # Randomly select 'real' samples and update discriminator weights
            X_real, y_real = generate_real_samples(dataset, n_samples=half_batch)
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            
            # Generate 'fake' samples and update discriminator weights
            X_fake, y_fake = generate_fake_samples(g_model,
                n_samples=half_batch, latent_dim=latent_dim)
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            
            # Prepare latent space points with inverted labels to train generator via GAN
            X_gan = generate_latent_points(n_samples=n_batch, latent_dim=latent_dim)
            y_gan = np.ones((n_batch, 1)) # target for generator is prediction of 1 (real)
            g_loss = gan_model.train_on_batch(X_gan, y_gan)

            # Append results to history record
            loss_records.append({'epoch': epoch, 'batch': batch,
                'd_loss_real': d_loss1, 'd_loss_fake': d_loss1, 'g_loss': g_loss})
            
            # Summarise batch
            s = f"Epoch {epoch:03d}, batch {batch:0{bat_digs}d}/{bat_per_epo:0{bat_digs}d}"
            s += f" - loss: D_real={d_loss1:.3f}, D_fake={d_loss2:.3f}, G={g_loss:.3f}"
            print(s)
        
        # Record model artefact and images periodically
        save_files = ((epoch % 10) == 0)
        d_acc_real, d_acc_fake = summarise_performance(epoch, g_model, d_model, dataset,
            latent_dim=latent_dim, save_files=save_files)
        acc_records.append({'epoch': epoch, 'd_acc_real': d_acc_real, 'd_acc_fake': d_acc_fake})
        save_history({'loss': loss_records, 'acc': acc_records})

    log_print(f"FINISHED GAN TRAINING - duration {dt.datetime.now() - start}", highlight=True)

    return (loss_records, acc_records)


def summarise_performance(epoch, g_model, d_model, dataset,
    latent_dim=LATENT_DIM_DEF, n_samples=100, save_files=False):
    """Summarise current model performance and save results
    
    - Logs accuracy of disriminator on real and fake samples
    - Saves image file of generated images
    - Saves the generator model as h5
    
    Arguments:
        g_model = generator model, in training
        d_model = discriminator model, in training
        dataset = input data array, (n_samples, height_pixels, width_pixels, n_channels)
        latent_dim = int, dimensionality of latent space
        n_epochs = int, number of training epochs
        n_batch = int, batch size
        save_files = bool, whether to print results and save files, otherwise just returns accuracy
        
    Returns (d_acc_real, d_acc_fake), tuple of discriminator accuracy on real and fake images
    """
    # Evaluate discriminator on real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    
    # Evaluate discriminator on fake samples
    x_fake, y_fake = generate_fake_samples(g_model, n_samples=n_samples, latent_dim=latent_dim)
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)

    if save_files:
        # Save fake images and generator model
        save_plot(x_fake, epoch)
        model_fname = utils.name_inter_model_file(epoch)
        g_model.save(model_fname)
        d_model.save(model_fname.replace('generator', 'discriminator'))
        
        # Log discriminator performance
        print("")
        log_print(f"Epoch {epoch} discriminator accuracy: real={acc_real:.1%}, fake={acc_fake:.1%}")
        print(f"    - sample generated images and model artefacts saved to file")
        print("")

    return (acc_real, acc_fake)


def save_history(data, **kwargs):
    """Save training history to file
    """
    to_csv_kwds = {'index': False, **kwargs}
    if isinstance(data, pd.DataFrame):
        fname = utils.name_history_file()
        data.to_csv(fname, **to_csv_kwds)

    elif isinstance(data, dict):
        for suffix, sub_data in data.items():
            fname = utils.name_history_file(suffix=suffix)
            sub_df = pd.DataFrame(sub_data)
            sub_df.to_csv(fname, **to_csv_kwds)

    else:
        fname = utils.name_history_file()
        data_df = pd.DataFrame(data)
        data_df.to_csv(fname, **to_csv_kwds)


################################################################################
### Helper utils
################################################################################


def generate_real_samples(dataset, n_samples, seed=None):
    """Randomly select 'real' sample images from dataset
    Returns pixel data X and labels y (namely 1, because all are real images) 
    """
    # Select samples from dataset at random
    if seed is not None:
        np.random.seed(int(seed))
    ix = np.random.randint(low=0, high=dataset.shape[0], size=n_samples)
    X = dataset[ix]
    
    y = np.ones((n_samples, 1)) # create 'real' class labels (1)
    
    return X, y


def generate_fake_samples(g_model, n_samples=1, latent_dim=LATENT_DIM_DEF, seed=None):
    """Generate 'fake' images using generator
    Returns pixel data X and labels y (namely 0, because all are fake images) 
    """
    x_input = generate_latent_points(n_samples=n_samples,
        latent_dim=latent_dim, seed=seed) # sample latent space
    X = g_model.predict(x_input) # generator predictions from points in latent space 
    
    y = np.zeros((n_samples, 1)) # create 'real' class labels (1)
    
    return X, y


def generate_latent_points(n_samples=1, latent_dim=LATENT_DIM_DEF, seed=None):
    """Randomly generate points in latent space
    Returns (n_samples, latent_dim) array of numbers drawn from standard normal distribution
    """
    if seed is not None:
        np.random.seed(int(seed))
    return np.random.randn(n_samples, latent_dim)


def save_plot(examples, epoch, n=10):
    """Save sample images to file
    """
    fig = utils.plot_images(examples, grid_shape=(n,n), rescale=True) # generate plot
    
    # Save plot to file
    img_fname = utils.name_inter_img_file(epoch)
    plt.savefig(img_fname)
    plt.close()


def log_print(s, date_fmt='%c', highlight=False):
    """Prints string with timestamp
    """
    if highlight:
        print("\n-------------------------------------------------")
    print(f"{dt.datetime.now():{date_fmt}}:", s)
    if highlight:
        print("-------------------------------------------------\n")

