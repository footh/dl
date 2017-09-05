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
import zones_config
import math

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

def gaussian_fit(x, h, mu, sig):
    """
        Return values from gaussian curve based on parameters. Takes scalar or numpy array.
    """
    return h*np.exp(-np.power((x - mu)/sig, 2.)/2)

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
        'outliers' values and averaging the rest. (Note that 'up' is subtracting rows because image starts from top.)
    """
    row_arr, index_arr = np.where(img[row-up:row+down])
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
            
        print(f"torso_begin_rows({slice}): {torso_begin_rows}")
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
        
        print(f"head_begin_rows({slice}): {head_begin_rows}")
        # Drop high and low and take max of rest
        head_begin_rows.sort()
        head_begin_row = max(head_begin_rows[1:-1])
        
        point_dict[slice] = [torso_begin_row, torso_begin_column, torso_end_column, head_begin_row]
    
    return point_dict

def critical_points_quarters(imga_dict, torso_begin_row):
    """
        Takes quarter images and gets the torso begin and end columns and the center point based on the torso begin row
    """
    SIDE_MARGIN = 100
    
    point_dict = {}
    for slice, imga in imga_dict.items():    
        rows, columns = imga.shape
        midrow = rows // 2
        
        torso_begin_columns = []
        torso_end_columns = []
        
        transforms = OrderedDict()
        transforms[gaussian_filter] = {}
        transforms[threshold] = {}
        transforms[open] = {}
        #transforms[convex_hull] = {}
        
        sigmas = [4, 6]
        rads = [10, 15]
        imgt = imga[midrow:,SIDE_MARGIN:-SIDE_MARGIN]
        for i in range(len(sigmas)):
            transforms[gaussian_filter] = {'sigma': sigmas[i]}
            transforms[open] = {'rad': rads[i]}
            
            transformed_img = run_transforms(imgt, transforms)
            
            mod_tbr = torso_begin_row - midrow
            torso_begin_column, torso_end_column = bounding_columns(transformed_img, mod_tbr, up=mod_tbr, down=5)
            
            # This is a cleaner center that won't be skewed by unusual girth
            center_tbc, center_tec = bounding_columns(transformed_img, mod_tbr, up=10, down=5, outliers=3)
            center = (center_tbc + center_tec) // 2
            
            torso_begin_columns.append(torso_begin_column + SIDE_MARGIN)
            torso_end_columns.append(torso_end_column + SIDE_MARGIN)
            
        print(f"torso_begin_columns({slice}): {torso_begin_columns}")
        print(f"torso_end_columns({slice}): {torso_end_columns}")
        torso_begin_column = max(torso_begin_columns)
        torso_end_column = min(torso_end_columns)

        point_dict[slice] = [torso_begin_column, torso_end_column, center]
        
    return point_dict

def torso_rects(tbr, tbc, tec, hbr, rows, slice_cursor=0, slice=0, x_rot_adj=0, z_rot_adj=0):
    #TODO: pass in an optional config for constants so I don't have to constantly stop python when I want to
    # tweak a parameter during fine-tuning
    
    
    # Rotation to account for standing off center on the x and z axes. This adjustment applied to the 
    # bounding column values only (from which interior columns are derived).
    #TODO: determine if x adjustment is worth doing (from testing, looks very small)
    #Z_ROT_SLICE_ADJ = 0.25
    Z_HEIGHT = 0.25
    Z_SIGMA = 5.0
    z_rot_slice_adj = gaussian_fit(slice, Z_HEIGHT, 8, Z_SIGMA)
    rot_adj = int(slice_cursor * x_rot_adj * 0) + \
              int(slice_cursor * z_rot_adj * z_rot_slice_adj)            

    tbc_adj = tbc + rot_adj
    tec_adj = tec + rot_adj
    #print(f"rot_adj: {rot_adj}")

    torso_width = tec - tbc
    
    # Upper torso column movement is a linear adjustment
    # TODO: this may need a gaussian too?
    UPPER_TORSO_SLICE_ADJ = 0.1
    slice_col_adj1 = int(slice_cursor * torso_width * UPPER_TORSO_SLICE_ADJ)

    # Lower torso column movement is determined by a gaussian with these parameters (to account from scanner
    # accelerating then decelerating as it starts and stops).
    G_HEIGHT = 0.12
    G_SIGMA = 2.2
    lower_torso_slice_adj = gaussian_fit(slice, G_HEIGHT, 8, G_SIGMA) 
    slice_col_adj2 = int(round(slice_cursor * torso_width * lower_torso_slice_adj))
    #print(f"slice, adj: {slice}, {lower_torso_slice_adj}")
    #print(f"slice_cursor: {slice_cursor}")
    #print(f"slice_col_adj1: {slice_col_adj1}")
    #print(f"slice_col_adj2: {slice_col_adj2}")    
    
    TORSO_PORTIONS = 15
    UPPER_TORSO_PORTION = 4
    LOWER_TORSO_PORTION = 10
    torso_height = tbr - hbr
    torso_unit = torso_height // TORSO_PORTIONS

    torso_split_row = hbr + UPPER_TORSO_PORTION * torso_unit
    waist_split_row = hbr + LOWER_TORSO_PORTION * torso_unit
    lower_torso_split_column = (torso_width // 2 + tbc_adj) + slice_col_adj1                          

    WAIST_PORTIONS = 15
    SIDE_WAIST_PORTION = 4    
    side_waist_size = torso_width // WAIST_PORTIONS * SIDE_WAIST_PORTION
    waist_split_column1 = side_waist_size + tbc_adj + slice_col_adj2
    waist_split_column2 = torso_width - side_waist_size + tbc_adj + slice_col_adj2
#     if slice == 0 or slice == 8:
#         print(f"--------------slice {slice}")
#         print(f"torso_width: {torso_width}")
#         print(f"waist portions: {torso_width // WAIST_PORTIONS}")
#         print(f"left waist size: {side_waist_size}")
#         print(f"mid waist size: {torso_width - 2 * side_waist_size}")
#         print(f"right waist size: {torso_width - side_waist_size}")
    
    #TODO: build rects but have 'valid_rect' method that makes sure rect doesn't have (-) values o/w returns (0,0,0,0),
    # but if padding zone an 'invalid' rect may become valid?
    
    # Note: left/right are as the observer of the image
    # These next two are for the upper chest/upper back adjustment from side body
    #TODO: fine tune this
    utl_adj = slice_col_adj1 if slice_cursor > 0 else 0
    utr_adj = slice_col_adj1 if slice_cursor < 0 else 0
    upper_torso_rect = [tbc_adj + utl_adj, hbr, tec_adj + utr_adj, torso_split_row]
    left_torso_rect = [tbc_adj, torso_split_row, lower_torso_split_column, waist_split_row]
    right_torso_rect = [lower_torso_split_column, torso_split_row, tec_adj, waist_split_row]
    
    #TODO: better choice than +35
    leg_size = rows - tbr
    leg_portion = leg_size // 7
    print(f"leg_portion: {leg_portion}")
    left_waist_rect = [tbc_adj, waist_split_row, waist_split_column1, tbr+leg_portion]
    mid_waist_rect = [waist_split_column1, waist_split_row, waist_split_column2, tbr+leg_portion]
    right_waist_rect = [waist_split_column2, waist_split_row, tec_adj, tbr+leg_portion]
    
    return upper_torso_rect, left_torso_rect, right_torso_rect, left_waist_rect, mid_waist_rect, right_waist_rect

def torso_rects_side(tbr, tbc, tec, hbr):
    TORSO_PORTIONS = 15
    TORSO_PORTION = 10
    torso_height = tbr - hbr
    torso_unit = torso_height // TORSO_PORTIONS
    
    waist_split_row = hbr + TORSO_PORTION * torso_unit
        
    torso_rect = [tbc, hbr+20, tec, waist_split_row] #TODO: better then +20
    waist_rect = [tbc, waist_split_row, tec, tbr+0]
    
    return torso_rect, waist_rect

def adjust_torso_bounds(tbc1, tec1, tbc2, tec2):
    """
        Will adjust the torso bounding columns by apportioning up the smaller width, (larger width stays the same)
    """
    ADJUST_PCT = 0.8
    w1 = tec1 - tbc1
    w2 = tec2 - tbc2
    
    diff_adj = abs(w1-w2) * ADJUST_PCT // 2
    if w1 > w2:
        return tbc1, tec1, tbc2-diff_adj, tec2+diff_adj
    else:
        return tbc1-diff_adj, tec1+diff_adj, tbc2, tec2
    

def create_zones16(file_images):
    """
        Takes the 16 slice file images and returns the zone rectangles by slice
    """
    crit_point_slices = [0, 8]
    crit_point_imga_dict = {}
    for slice in crit_point_slices:
        crit_point_imga_dict[slice] = np.asarray(file_images[slice])
    
    crit_point_dict = critical_points(crit_point_imga_dict)
 
    c_tbr0, c_tbc0, c_tec0, c_hbr0 = crit_point_dict[0]
    c_tbr8, c_tbc8, c_tec8, c_hbr8 = crit_point_dict[8]
    
    c_tbc0a, c_tec0a, c_tbc8a, c_tec8a = adjust_torso_bounds(c_tbc0, c_tec0, c_tbc8, c_tec8)
    
    crit_point_slices_q = [4, 12]
    crit_point_imga_dict_q = {}
    for slice in crit_point_slices_q:
        crit_point_imga_dict_q[slice] = np.asarray(file_images[slice])
    
    # Difference of calculated midpoint from actual midpoint (do I even need this)?
    rows, columns = 660, 512 # TODO: parameterize
    x_rot_adj = (columns // 2) - ((c_tbc0 + c_tec0) // 2)
    print(f"x_rot_adj: {x_rot_adj}")
    print(f"tbc diff: {c_tbc8-c_tbc0}")
    print(f"tec diff: {c_tec8-c_tec0}")

    crit_point_dict_q = critical_points_quarters(crit_point_imga_dict_q, c_tbr0)

    c_tbc4, c_tec4, c_ctr4 = crit_point_dict_q[4]
    c_tbc12, c_tec12, c_ctr12 = crit_point_dict_q[12]
    
    z_depth = ((c_tec12 - c_tbc12) + (c_tec4 - c_tbc4)) // 2
    print(f"z_depth: {z_depth}")
    
    z_rot_adj = c_ctr12 - c_ctr4
    print(f"z_rot_adj: {z_rot_adj}")
    
    c_tbr = int(0.8 * max(c_tbr0, c_tbr8) + 0.2 * min(c_tbr0, c_tbr8))
    c_hbr = (c_hbr0 + c_hbr8) // 2

    # 16 slices, 17 zones (to keep zone indices equal to zone diagram adding one more. 0 index not used), 
    # 4 points for rectangle of zone
    #        8
    #     7     9
    #   6         10
    #  5           11
    # 4             12
    #  3           13
    #   2         14
    #     1     15
    #        0
    zones = np.zeros((16, 18, 4), dtype=np.uint16)
        
    zones[0][5], zones[0][6], zones[0][7], zones[0][8], zones[0][9], zones[0][10] = torso_rects(c_tbr, c_tbc0a, c_tec0a, c_hbr, rows)

    zones[1][5], zones[1][6], zones[1][7], zones[1][8], zones[1][9], zones[1][10] = torso_rects(c_tbr, c_tbc0a, c_tec0a, c_hbr, rows,
                                                                                                slice_cursor=-1, slice=1, x_rot_adj=x_rot_adj, z_rot_adj=z_rot_adj)
    zones[2][5], zones[2][6], zones[2][7], zones[2][8], zones[2][9], zones[2][10] = torso_rects(c_tbr, c_tbc0a, c_tec0a, c_hbr, rows,
                                                                                                slice_cursor=-2, slice=2, x_rot_adj=x_rot_adj, z_rot_adj=z_rot_adj)
    zones[3][7], zones[3][10] = torso_rects_side(c_tbr, c_tbc4, c_tec4, c_hbr)
    zones[4][7], zones[4][10] = torso_rects_side(c_tbr, c_tbc4, c_tec4, c_hbr)
    zones[5][7], zones[5][10] = torso_rects_side(c_tbr, c_tbc4, c_tec4, c_hbr)
    
    zones[6][17], zones[6][7], zones[6][6], zones[6][10], zones[6][9], zones[6][8] = torso_rects(c_tbr, c_tbc8a, c_tec8a, c_hbr, rows,
                                                                                                slice_cursor=2, slice=6, x_rot_adj=x_rot_adj, z_rot_adj=-z_rot_adj)
    zones[7][17], zones[7][7], zones[7][6], zones[7][10], zones[7][9], zones[7][8] = torso_rects(c_tbr, c_tbc8a, c_tec8a, c_hbr, rows,
                                                                                                slice_cursor=1, slice=7, x_rot_adj=x_rot_adj, z_rot_adj=-z_rot_adj)    
    zones[8][17], zones[8][7], zones[8][6], zones[8][10], zones[8][9], zones[8][8] = torso_rects(c_tbr, c_tbc8a, c_tec8a, c_hbr, rows, slice=8)

    zones[9][17], zones[9][7], zones[9][6], zones[9][10], zones[9][9], zones[9][8] = torso_rects(c_tbr, c_tbc8a, c_tec8a, c_hbr, rows,
                                                                                                 slice_cursor=-1, slice=9, x_rot_adj=x_rot_adj, z_rot_adj=-z_rot_adj)
    zones[10][17], zones[10][7], zones[10][6], zones[10][10], zones[10][9], zones[10][8] = torso_rects(c_tbr, c_tbc8a, c_tec8a, c_hbr, rows, 
                                                                                                       slice_cursor=-2, slice=10, x_rot_adj=x_rot_adj, z_rot_adj=-z_rot_adj)
    zones[11][6], zones[11][8] = torso_rects_side(c_tbr, c_tbc12, c_tec12, c_hbr)
    zones[12][6], zones[12][8] = torso_rects_side(c_tbr, c_tbc12, c_tec12, c_hbr)
    zones[13][6], zones[13][8] = torso_rects_side(c_tbr, c_tbc12, c_tec12, c_hbr)
    
    zones[14][5], zones[14][6], zones[14][7], zones[14][8], zones[14][9], zones[14][10] = torso_rects(c_tbr, c_tbc0a, c_tec0a, c_hbr, rows,
                                                                                                slice_cursor=2, slice=14, x_rot_adj=x_rot_adj, z_rot_adj=z_rot_adj)
    zones[15][5], zones[15][6], zones[15][7], zones[15][8], zones[15][9], zones[15][10] = torso_rects(c_tbr, c_tbc0a, c_tec0a, c_hbr, rows,
                                                                                                slice_cursor=1, slice=15, x_rot_adj=x_rot_adj, z_rot_adj=z_rot_adj)
    return zones     

def draw_zones(file=None, slices=range(16), src_dir='train', save_file='zone_test', animation=False):
    # Read first file from shuffled list
    if file is None:
        file = shuffled_files(src_dir)[0]
        print(file)

    file_images = util.read_data(file, as_images=True)

    zones = create_zones16(file_images)    

    for slice in slices:
        img = file_images[slice]
        draw = ImageDraw.Draw(img)
        for zone in range(1,18,1):
            if np.sum(zones[slice, zone]) > 0:
                rect = list(zones[slice, zone])
                draw.rectangle(rect, outline='white')
                draw.text((rect[0]+2, rect[1]+2), str(zone), fill='white')            
        del draw
        
        if animation:
            file_images[slice] = img
                
        if save_file is not None:
            img.save(os.path.join('zones', save_file + str(slice) + '.png'))
            
    if animation:
        util.animate_images(file_images)        
