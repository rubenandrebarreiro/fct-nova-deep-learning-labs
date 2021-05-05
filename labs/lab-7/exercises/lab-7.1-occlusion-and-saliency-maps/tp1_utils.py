#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lab 7.1 - Occlusion Maps

Author:
- Rodrigo Jorge Ribeiro (rj.ribeiro@campus.fct.unl.pt)
- Ruben Andre Barreiro (r.barreiro@campus.fct.unl.pt)

Adapted from:
1) Deep Learning - Assignment #1:
   - Pokemon Image Classification and Segmentation with Deep Neural Networks

"""

# Import Python's Modules, Libraries and Packages

# Import the Image Save
# from the SciKit-Learn.Input/Output Python's Module
from skimage.io import imsave

# Import the NumPy Python's Library, with the np alias
import numpy as np

# Import the Pandas Python's Library, with the pd alias
import pandas as pd

# Import the NumPy Python's Library, with the np alias
from tensorflow.keras.utils import to_categorical

# Import the Util
# from the SciKit-Learn-Input/Output.Core Python's Module
from imageio.core import util as image_io_util


# Function to silence the SciKit-Learn-Input/Output Warnings
# noinspection PyUnusedLocal
def silence_imageio_warning(*args, **kwargs):

    # Pass/Ignore it
    pass


# Silence the SciKit-Learn-Input/Output Warnings
image_io_util._precision_warn = silence_imageio_warning


def images_to_pic(f_name, images, width=20):
    """
    Saves a single image file with the images provided in the images array.
    Example:
        data = load_data()
        images_to_pic('test_set.png',data['test_X'])
    """
    rows = - (len(images)//-width)

    image_hei = images[0].shape[0]
    image_wid = images[0].shape[1]

    picture = np.zeros((rows*image_hei, width*image_wid, 3))
    row = col = 0

    for img in images:
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]

        picture[row*image_hei:(row+1)*image_hei, col*image_wid:(col+1)*image_wid] = img
        col += 1

        if col >= width:
            col = 0
            row += 1

    imsave(f_name, picture)


def compare_masks(f_name, masks1, masks2, width=20):
    """
    Creates a single image file comparing the two sets of masks
    Red pixels are pixels in mask1 but not in mask2
    Green pixels are in mask2 but not in mask2
    White pixels are in both and black in neither.
    Example:
        predicts = model.predict(data['test_X'])
        compare_masks('test_compare.png',data['test_masks'],predicts)
    """
    imgs = []

    for m1, m2 in zip(masks1, masks2):

        img = np.zeros((m1.shape[0], m1.shape[1], 3))

        img[m1[:, :, 0] > 0.5, 0] = 1
        img[m2[:, :, 0] > 0.5, 1] = 1

        img[np.logical_and(m1[:, :, 0] > 0.5, m2[:, :, 0] > 0.5), 2] = 1
        imgs.append(img)

    images_to_pic(f_name, imgs, width)


def overlay_masks(f_name, images, masks, width=20):
    """
    Creates a single image file overlaying the masks and the images
    Region outside the masks are turned red and darker
    Example:
        predicts = model.predict(data['test_X'])
        overlay_masks('test_overlay.png',data['test_X'],predicts)
    """
    imgs = []
    for im, m in zip(images, masks):
        img = np.copy(im)

        mask = m[:, :, 0] < 0.5

        img[mask, 1:] = img[mask, 1:]/10
        img[mask, 0] = (1+img[mask, 0])/2

        imgs.append(img)

    images_to_pic(f_name, imgs, width)

    
def load_data():
    """
    Returns dictionary with data set, in the following keys:
        'train_X':         training images, RGB, shape (4000,64,64,3)
        'test_X':          test images, RGB, shape (500,64,64,3)
        'train_masks':     training masks, shape (4000,64,64,1)
        'test_masks':      test masks, shape (500,64,64,1)
        'train_classes':   classes for training, shape (4000,10), one-hot encoded
        'train_labels':    labels for training, shape (4000,10) with 1 on main and secondary types
        'test_classes':    test classes, shape (500,10), one-hot encoded
        'test_labels':     test labels, shape (500,10) with 1 on main and secondary types
    """
    data = np.load('dataset/data.npz')        
    table = pd.read_csv('dataset/labels.csv')
    classes = to_categorical(table['primary_type_index'].values)
    labels = np.copy(classes)
    
    mask = ~table['secondary_type_index'].isnull()
    rows = np.arange(len(mask))[mask]
    secondary_ixs = table['secondary_type_index'][mask].values.astype(int)    
    labels[tuple((rows, secondary_ixs))] = 1
    
    return {'train_X': data['images'][:4000],
            'test_X': data['images'][4000:],
            'train_masks': data['masks'][:4000],
            'test_masks': data['masks'][4000:],
            'train_classes': classes[:4000],
            'train_labels': labels[:4000],
            'test_classes': classes[4000:],
            'test_labels': labels[4000:], }


if __name__ == '__main__':

    ds = load_data()

    for k in ds.keys():
        print(k, ds[k].shape)

    images_to_pic('tp1_sample.png', ds['train_X'][:100])
    images_to_pic('tp1_sample_masks.png', ds['train_masks'][:100])
