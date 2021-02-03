GAN_face_generator
==============================

Author: Geoff Chambers

Active Date: August 2020 

Tutorial for training a GAN to generate faces, and exploring the model latent space.

Based on:
- [ML Mastery blog post](https://machinelearningmastery.com/how-to-interpolate-and-perform-vector-arithmetic-with-faces-using-a-generative-adversarial-network/)
- [Radford et al, 2016, Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)


### GPU setup tips:

Note, may have to get Nvidia cuda drivers for ubuntu to use GPU, in my case:
```
wget https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda_11.0.3_450.51.06_linux.run
sudo sh cuda_11.0.3_450.51.06_linux.run
# then follow instructions to update environment variables
```

Based on [this blog](https://davidsanwald.github.io/2016/11/13/building-tensorflow-with-gpu-support.html), this Nvidia setup may be incomplete ... the above link above maybe for Cuda toolkit only, may also need to download and install Nvidia drivers first.

A more reliable approach to set up the environment may be to use the AWS Deep Learning AMI and adapt one
of the provided conda environments as required.

Finally, create conda environment from YML file:
```
conda env create --file ./environments/proj-home-ins.yml`
```


### Key files:
- notebooks/3_ExploreResults.ipynb - explore latent space with trained model
- notebooks/2_TrainModel.ipynb - model set-up and training

Project Organization
------------

    ├── README.md        <- The top-level README for developers using this project.
    │
    ├── data/            <- data files used for this project.
        ├── img_align_celeba/    <- faces dataset (https://www.kaggle.com/jessicali9530/celeba-dataset/data)   
    │   
    ├── environments/    <- YAML files definining Conda environment(s) used for this project.
        ├── gan-tutorial.yml     <- pip install kaggle, mtcnn, tensorflow, tensorflow-gpu; conda install cudatoolkit
    │
    ├── images/          <- image files
        ├── intermed_training/   <- examples of images produced during training for QC
    │
    ├── models/          <- model artefacts
    │
    ├── notebooks/       <- Jupyter notebooks
        ├── 1_PrepareDataset     <- prepares celeba dataset
        ├── 2_TrainModel         <- define and train GAN
        ├── 3_ExploreResults     <- explore latent space of trained model
    │
    ├── scripts/         <- Python/Bash scripts.
        ├── get_celeba_data.sh   <- download celeba dataset from kaggle (req kaggle set up with api key)
    │
    ├── src/             <- Source code for use in this project.
        ├── utils.py             <- basic utils used in multiple settings
        ├── prep_data.py         <- ingest and prepare dataset for training
        ├── model.py             <- GAN defininition and training functions
        ├── explore.py           <- explore results of trained model
    │

--------

