import os
import time
import nibabel as nib
import pandas as pd
import numpy as np
import scipy
from scipy import ndimage
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import skimage
from skimage import feature

from scipy import spatial
from helpers import utils
#from loaders import data_augmentation

def heart_metrics(seg_3Dmap, voxel_size, classes=[3, 1, 2]):
    """
    Compute the volumes of each classes
    """
    # Loop on each classes of the input images
    volumes = []
    for c in classes:
        # Copy the gt image to not alterate the input
        seg_3Dmap_copy = np.copy(seg_3Dmap)
        seg_3Dmap_copy[seg_3Dmap_copy != c] = 0

        # Clip the value to compute the volumes
        seg_3Dmap_copy = np.clip(seg_3Dmap_copy, 0, 1)

        # Compute volume
        # volume = seg_3Dmap_copy.sum() * np.prod(voxel_size) / 1000.
        volume = seg_3Dmap_copy.sum() * np.prod(voxel_size)
        volumes += [volume]
    return volumes

def ejection_fraction(ed_vol, es_vol):
    """
    Calculate ejection fraction
    """
    stroke_vol = ed_vol - es_vol
    return float(stroke_vol) / float(ed_vol)

def myocardialmass(myocardvol):
    """
    Specific gravity of heart muscle (1.05 g/ml)
    """ 
    return myocardvol*1.05

def bsa(height, weight):
    """
    Body surface Area
    """
    return np.sqrt((height*weight)/3600)

def myocardial_thickness(label_path):
    import nibabel as nib
    import numpy as np
    from skimage import feature
    from scipy import ndimage, spatial

    myo_label = 2
    label_obj = nib.load(label_path)
    myocardial_mask = (label_obj.get_fdata() == myo_label)

    # Define slices to skip (avoid edge noise)
    slices_to_skip = (1, 1)

    # Pixel spacing in X and Y
    pixel_spacing = label_obj.header.get_zooms()[:2]
    assert pixel_spacing[0] == pixel_spacing[1]

    holes_filles = np.zeros(myocardial_mask.shape)
    interior_circle = np.zeros(myocardial_mask.shape)
    cinterior_circle_edge = np.zeros(myocardial_mask.shape)
    cexterior_circle_edge = np.zeros(myocardial_mask.shape)

    overall_avg_thickness = []
    overall_std_thickness = []

    for i in range(slices_to_skip[0], myocardial_mask.shape[2] - slices_to_skip[1]):
        holes_filles[:, :, i] = ndimage.binary_fill_holes(myocardial_mask[:, :, i])
        interior_circle[:, :, i] = holes_filles[:, :, i] - myocardial_mask[:, :, i]
        cinterior_circle_edge[:, :, i] = feature.canny(interior_circle[:, :, i])
        cexterior_circle_edge[:, :, i] = feature.canny(holes_filles[:, :, i])

        x_in, y_in = np.where(cinterior_circle_edge[:, :, i] != 0)
        x_ex, y_ex = np.where(cexterior_circle_edge[:, :, i] != 0)

        if len(x_ex) > 0 and len(x_in) > 0:
            total_distance_in_slice = []
            for z in range(len(x_in)):
                a = np.array([x_in[z], y_in[z]])
                distances = [spatial.distance.euclidean(a, [x_ex[k], y_ex[k]]) for k in range(len(x_ex))]
                min_dist = np.min(distances)
                total_distance_in_slice.append(min_dist)

            avg_thickness = np.mean(total_distance_in_slice) * pixel_spacing[0]
            std_thickness = np.std(total_distance_in_slice) * pixel_spacing[0]

            overall_avg_thickness.append(avg_thickness)
            overall_std_thickness.append(std_thickness)

    return overall_avg_thickness, overall_std_thickness
