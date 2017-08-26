import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from skimage.filters import gaussian, threshold_li, threshold_mean, threshold_otsu
from skimage.morphology import convex_hull_image, reconstruction, closing, opening, disk
import util
from setup_data import shuffled_files, label_dict
from PIL import Image, ImageDraw, ImageFont
import os
from collections import OrderedDict

_DEBUG_ = False

def gaussian_filter(img, sigma=3):
    """
        Return gaussian filter of image
    """
    return gaussian(img, sigma=sigma)

def fun_filter(img):
    from skimage.filters import threshold_local
           
    thresh =  threshold_local(img, 11, offset=5)
    return img > thresh

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

def binary_peaks(img, peak_count, hits=20, min_size=10, min_distance=5, down=True):
    """
        Find first occurrence of exactly 'peak_count' peaks in binary file row. Peak must be found 'hits' 
        times in a row with 'min_size' and 'min_distance between peaks. Searching from bottom up. Returns
        found row.
    """    
    rows, columns = img.shape
    #print(f"binary_peaks, img.shape: {img.shape}")
    
    # This implementation will count peaks at beginning and end of row which is OK since these rows
    # should have plenty of space in between peaks
    peak_hits = 0
    peak_hit_row = 0
    
    row_span = range(rows)
    if not down:
        row_span = range(rows-1, -1, -1)
    
    for i in row_span:
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

def bounding_columns(img, row, up=10, down=10, outliers=5):
    """
        Find bounding columns of 'row' by looking 'up' and 'down' rows, throwing out top and bottom
        'outliers' values and averaging the rest.
    """
    row_arr, index_arr = np.where(img[row-down:row+up])
    begins = []
    ends = []
    for i in range(up+down):
        indices = index_arr[row_arr==i]
        begins.append(indices[0])
        ends.append(indices[-1])
        
    begins.sort()
    ends.sort()
    begin = np.mean(begins[outliers:-outliers])
    end = np.mean(ends[outliers:-outliers])
    
    return int(begin), int(end) + 1

def run_transforms(imga, transforms):
    """
        Runs a series of transforms as defined by 'transforms' on the 'img'
    """
    transformed_image = np.copy(imga)
    for method, kwargs in transforms.items():
        imgt = method(transformed_image, **kwargs)
        if _DEBUG_: util.plot_compare(transformed_image, imgt)
        transformed_image = imgt
    
    return transformed_image

def nub_head_begin(imga):
    """
        Returns head begin row by taking the minimum of top of head - estimated head size
        and first spot where torso ends
    """
    EST_HEAD_SIZE = 75
    MID_COL_SPAN = 7
    MID_ROW_SPAN = 4
    
    rows, columns = imga.shape
    mid_left = (columns // 2) - MID_COL_SPAN
    mid_right = (columns // 2) + MID_COL_SPAN
    
    head_begin1 = headbegin2 = 0
    
    head_top = 0
    for i in range(rows):
        if np.sum(imga[i-MID_ROW_SPAN:i+MID_ROW_SPAN, mid_left:mid_right]) > 0:
            head_top = i
            break
    head_begin1 = head_top + EST_HEAD_SIZE
    
    head_top = 0
    for i in range(imga.shape[0]-1, -1, -1):
        if np.sum(imga[i-MID_ROW_SPAN:i+MID_ROW_SPAN, mid_left:mid_right]) == 0:
            head_top = i
            break
        
    head_begin2 = head_top + EST_HEAD_SIZE
    #print(f"head_top: {head_top}")
    #print(f"head_begin1, head_begin2: {head_begin1}, {head_begin2}")
    return head_begin1, head_begin2

def nub_torso_begin(imga):
    """
        Get the torso points of the nub image left after the transform
    """
    torso_begin_row = binary_peaks(imga, 2, hits=10)
    
    torso_begin_column, torso_end_column = bounding_columns(imga, torso_begin_row, up=20, down=5)
        
    return torso_begin_row, torso_begin_column, torso_end_column
    
def critical_points(imga_dict):
    """
        Takes the (front or back) image array and returns the critical points needed to create zones:
    """
    point_dict = {}
    for slice, imga in imga_dict.items():    
        rows, columns = imga.shape
        midrow = rows // 2
        
        torso_begin_rows = []
        torso_begin_columns = []
        torso_end_columns = []
        
        transforms = OrderedDict()
        transforms[gaussian_filter] = {}
        transforms[threshold] = {}
        transforms[open] = {}
        
        sigmas = [4, 6]
        rads = [10, 15]
        imgt = imga[midrow:]
        for i in range(len(sigmas)):
            transforms[gaussian_filter] = {'sigma': sigmas[i]}
            transforms[open] = {'rad': rads[i]}
            
            transformed_img = run_transforms(imgt, transforms)
            
            torso_begin_row, torso_begin_column, torso_end_column = nub_torso_begin(transformed_img)
            torso_begin_rows.append(torso_begin_row + midrow)
            torso_begin_columns.append(torso_begin_column)
            torso_end_columns.append(torso_end_column)
            
        print(f"torso_begin_rows: {torso_begin_rows}")
        torso_begin_row = min(torso_begin_rows)
        torso_begin_column = max(torso_begin_columns)
        torso_end_column = min(torso_end_columns)
        
        head_begin_rows = []

        sigmas = [4, 6]
        rads = [10, 15]
        imgt = imga[:midrow, torso_begin_column:torso_end_column]    
        for i in range(len(sigmas)):
            transforms[gaussian_filter] = {'sigma': sigmas[i]}
            transforms[open] = {'rad': rads[i]}
            
            transformed_img = run_transforms(imgt, transforms)
            
            head_begin_rows.extend(nub_head_begin(transformed_img))   
        
        print(f"head_begin_rows: {head_begin_rows}")
        # Drop high and low and take max of rest
        head_begin_rows.sort()
        head_begin_row = max(head_begin_rows[1:-1])
        
        point_dict[slice] = [torso_begin_row, torso_begin_column, torso_end_column, head_begin_row]
    
    return point_dict

def create_zones(file=None, slices=[], src_dir='train', save_file='zone_test'):
    # Read first file from shuffled list
    if file is None:
        file = shuffled_files(src_dir)[0]
        print(file)

    file_data = util.read_data(file)
    
    crit_point_slices = [0]
    crit_point_imga_dict = {}
    for slice in range(len(crit_point_slices)):
        img = np.flipud(file_data[:,:,slice].transpose())
        img = misc.toimage(img, channel_axis=2)
        crit_point_imga_dict[slice] = np.asarray(img)
    
    crit_point_dict = critical_points(crit_point_imga_dict)

    imga = crit_point_imga_dict[0]
    torso_begin_row, torso_begin_column, torso_end_column, head_begin_row = crit_point_dict[0]
    
    rows, columns = imga.shape

    slice = 0
    torso_size = head_begin_row - torso_begin_row
    torso_unit = torso_size // 15
    zone_5_endrow = head_begin_row - 4 * torso_unit
    zone_67_endrow = head_begin_row - 10 * torso_unit
    torso_width = torso_end_column - torso_begin_column
    zone_67_column = (torso_width // 2 + torso_begin_column) - int(slice * torso_width * 0.1)
    
    draw = ImageDraw.Draw(img)
    #draw.line([(0, torso_begin_row), (columns-1, torso_begin_row)], fill='white')
    draw.line([(0, head_begin_row), (columns-1, head_begin_row)], fill='white')
    draw.line([(torso_begin_column, torso_begin_row), (torso_begin_column, head_begin_row)], fill='white')
    draw.line([(torso_end_column, torso_begin_row), (torso_end_column, head_begin_row)], fill='white')
    
    #zone 5 bottom
    draw.line([(torso_begin_column, zone_5_endrow), (torso_end_column, zone_5_endrow)], fill='white')
    #zone 6/7 split
    draw.line([(zone_67_column, zone_5_endrow), (zone_67_column, zone_67_endrow)], fill='white')
    #zone 6/7 end
    draw.line([(torso_begin_column, zone_67_endrow), (torso_end_column, zone_67_endrow)], fill='white')
    
    del draw

    if save_file is not None:
        img.save(os.path.join('zones', save_file + str(0) + '.png'))
        
    #return imga

