import os
import cv2 as cv
import numpy as np

from remove_background import get_foreground_mask
from get_images import getTriplet
from mesh import generate_mesh

# camera_parameters = get_camera_parameters()

mask = [None, None, None]
triplet = getTriplet(1, 0)  # In final version we'll loop over these
for i, im in enumerate(triplet):
    # Make Foreground mask
    mask[i] = get_foreground_mask(im)
    # compensated for Non-linear lens deformation
    im = im
    # stereo rectification to facilitate the dense stereo matching
    im = im
    # global colour normalization to make sure that the so-called ‘Constant Brightness Assumption’ holds true.
    # A normalization could be applied with respect to mean and standard deviation of the colour channels.
    im = im
    # Save the image to the triplet
    triplet[i] = im

# Generate two meshes
mesh_LM = generate_mesh(im[0], im[1], mask[0], mask[1])
mesh_MR = generate_mesh(im[1], im[2], mask[1], mask[2])

# Merge the meshes using the ICP algorithm (iterated closest points)
