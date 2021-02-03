"""
Data ingestion and preparation functions
"""

import os, sys
from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN # face detection model


################################################################################
### Global variables
################################################################################


CELEBA_DIR = os.path.join(os.path.dirname(__file__),
    "../data/celeba_dataset/img_align_celeba/img_align_celeba/")


################################################################################
### Top-level ingestion pipeline, and data retrieval
################################################################################


def load_celeba_faces(n_faces=None, file_out=None, verbose=True):
    """Load celeba faces from file, detect face and resize image
    If n_faces is None, loads all files from CELEBA_DIR
    If file_out is None, returns face array,
        otherwise saves numpy compressed array file (.npz)
    """
    detection_model = MTCNN()
    faces = []
    for file in os.listdir(CELEBA_DIR):
        pixels = load_image(CELEBA_DIR + '/' + file)
        face = extract_face(pixels, model=detection_model)
        if face is not None:
            faces.append(face)
        if (n_faces is not None) and (len(faces) >= n_faces):
            break
    faces = np.asarray(faces)

    if verbose:
        print(f"Loaded and extracted {len(faces)} faces from celeba image directory")

    if file_out is None:
        return faces

    else:
        np.savez_compressed(file_out, faces)
        if verbose:
            print(f"Saved to numpy compressed file: {file_out}")
        return


################################################################################
### Helper utils
################################################################################


def extract_face(img_array, model=None, required_size=(80, 80)):
    """Extract face from image pixel array using MTCNN (or other model) and resize
    """
    if model is None:
        model = MTCNN()

    # Detect face in the image, return None if detection fails
    faces = model.detect_faces(img_array)
    if len(faces) == 0:
        return None

    # extract details of the face
    x1, y1, width, height = faces[0]['box']
    # force detected pixel values to be positive (bug fix)
    x1, y1 = abs(x1), abs(y1)
    # convert into coordinates
    x2, y2 = x1 + width, y1 + height
    # retrieve face pixels
    face_pixels = img_array[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face_pixels)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array


def load_images_from_dir(directory, n_images=None):
    """Load (optionally a subset) of images from specified directory
    Returns array of image pixel arrays
    """
    # Select files to load
    files = os.listdir(directory)
    if (n_images is not None) and (len(files) > n_images):
        files = files[:n_images]

    images = []
    for file in files:
        pixels = load_image(directory + file)
        images.append(pixels)

    return np.asarray(images)


def load_image(filename):
    """Load single image file to numpy array of pixel values (height x width x channels)
    """
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = np.asarray(image)
    return pixels

