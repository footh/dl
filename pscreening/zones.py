import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from skimage.filters import gaussian, threshold_li, threshold_mean, threshold_otsu
from skimage.morphology import convex_hull_image, reconstruction, closing, opening, disk
import util
from setup_data import shuffled_files, label_dict
from PIL import Image, ImageDraw, ImageFont
import os

_DEBUG_ = True

GAUSSIAN_FILTER_SIGMA = 3

def gaussian_filter(img, sigma=GAUSSIAN_FILTER_SIGMA):
    """
        Return gaussian filter of image
    """
    return gaussian(img, sigma=sigma)

def fun_filter(img):
    from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                     denoise_wavelet, estimate_sigma)
    from skimage import data, img_as_float, color
    
    imgf = img_as_float(img, force_copy=True)
       
    return denoise_tv_chambolle(imgf, weight=0.1)
    


def threshold(img, method='li'):
    """
        Return binary image based on threshold method
    """
    from skimage.filters import try_all_threshold

#     if _DEBUG_:
#         fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
#         plt.show()
    
    thresh = None
    if method == 'otsu':
        thresh = threshold_otsu(img)
    elif method == 'mean':
        thresh = threshold_mean(img)
    else:
        thresh = threshold_li(img)
        
    binary = img > thresh
    return binary

def convex_hull(img):
    """
        Return the convex hull of image
    """
    return convex_hull_image(img)

def fill_holes(img):
    """
        Fill holes in the image
    """ 
    seed = np.copy(img)
    seed[1:-1, 1:-10] = img.max()
    mask = img
    filled = reconstruction(seed, mask, method='erosion')
        
    return filled

def close(img, rad=10, times=1):
    selem = disk(rad)
    close_img = img
    for i in range(times):
        close_img = closing(close_img, selem)
        
    return close_img

def open(img, rad=10, times=1):
    selem = disk(rad)
    open_img = img
    for i in range(times):
        open_img = opening(open_img, selem)
        
    return open_img

def binary_peaks(img, peak_count, hits=20, min_size=10, min_distance=5):
    """
        Find first occurrance of exactly 'peak_count' peaks in binary file row. Peak must be found 'hits' 
        times in a row with 'min_size' and 'min_distance between peaks. Searching from bottom up. Returns
        found row.
    """    
    rows, columns = img.shape
    print(img.shape)
    
    # This implementation will count peaks at beginning and end of row which is OK since these rows
    # should have plenty of space in between peaks
    peak_hits = 0
    peak_hit_row = 0
    for i in range(rows-1, -1, -1):
        # Gets indices of all peak values
        peak_indices = np.where(img[i])[0]
        peaks = 0
        cur_peak_size = 1
        for j in range(len(peak_indices)-1):
            cur_index = peak_indices[j]
            next_index = peak_indices[j+1]
            if next_index > cur_index + 1:
                if next_index - cur_index > min_distance:
                    if cur_peak_size >= min_size: peaks += 1
                    cur_peak_size = 1
            else:
                cur_peak_size += 1
                
        if cur_peak_size >= min_size: peaks += 1
        
        if peaks == peak_count:
            if peak_hits == 0: peak_hit_row = i 
            peak_hits += 1
        else:
            peak_hits = 0
            peak_hit_row = 0
            
        if peak_hits == hits:
            return peak_hit_row

def bounding_columns(img, row, span=20, outliers=5):
    """
        Find bounding columns of 'row' by looking up and down 'span' rows, throwing out top and bottom
        'outliers' values and averaging the rest.
    """
    row_arr, index_arr = np.where(img[row-span:row+span])
    begins = []
    ends = []
    for i in range(span*2):
        indices = index_arr[row_arr==i]
        begins.append(indices[0])
        ends.append(indices[-1])
        
    begins.sort()
    ends.sort()
    begin = np.mean(begins[outliers:-outliers])
    end = np.mean(ends[outliers:-outliers])
    
    return int(begin), int(end) + 1

def critical_points(img):
    """
        Takes the (front or back) image array and returns the critical points needed to create zones:
    """
    #gauss = gaussian_filter(img)
    gauss = fun_filter(img)
    if _DEBUG_: util.plot_compare(img, gauss)
    
    #holes_filled = fill_holes(gauss)
    #if _DEBUG_: util.plot_compare(gauss, holes_filled)
    
    thresh = threshold(gauss)
    if _DEBUG_: util.plot_compare(gauss, thresh)

    closed = close(thresh, rad=15)
    if _DEBUG_: util.plot_compare(thresh, closed)
    
    opened = open(closed, rad=4)
    if _DEBUG_: util.plot_compare(closed, opened)
    
    TORSO_MARGIN = 100
    HEAD_MARGIN = 100
    
    #torso_begin_row = min(binary_peaks(thresh[:-100], 1), binary_peaks(closed[:-100], 1), binary_peaks(opened[:-100], 1))
    torso_begin_row = binary_peaks(opened[:-TORSO_MARGIN], 1)
    head_begin_row = binary_peaks(opened[:torso_begin_row-HEAD_MARGIN], 3)

    torso_begin_column, torso_end_column = bounding_columns(opened, torso_begin_row)

    return opened, torso_begin_row, torso_begin_column, torso_end_column, head_begin_row

def create_zones(file=None, slices=[], src_dir='train', save_file='zone_test'):
    # Read first file from shuffled list
    if file is None:
        file = shuffled_files(src_dir)[0]
        print(file)

    file_data = util.read_data(file)
    
    img = np.flipud(file_data[:,:,0].transpose())
    img = misc.toimage(img, channel_axis=2)
    imga = np.asarray(img)
    rows, columns = imga.shape        
    
    arr, torso_begin_row, torso_begin_column, torso_end_column, head_begin_row =  critical_points(imga)
    #torso_begin_row = critical_points(imga)
    #print(torso_begin_row)
    
    draw = ImageDraw.Draw(img)
    draw.line([(0, torso_begin_row), (columns-1, torso_begin_row)], fill='white')
    draw.line([(0, head_begin_row), (columns-1, head_begin_row)], fill='white')
    draw.line([(torso_begin_column, torso_begin_row), (torso_begin_column, head_begin_row)], fill='white')
    draw.line([(torso_end_column, torso_begin_row), (torso_end_column, head_begin_row)], fill='white')
    
    del draw

    if save_file is not None:
        img.save(os.path.join('zones', save_file + str(0) + '.png'))
        
    return arr, imga

