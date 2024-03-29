import numpy as np
from scipy import misc
from skimage.filters import gaussian, threshold_li, threshold_mean, threshold_otsu
from skimage.morphology import convex_hull_image, reconstruction, closing, opening, disk
import util
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

def close_filter(img, rad=10, times=1):
    selem = disk(rad)
    close_img = img
    for i in range(times):
        close_img = closing(close_img, selem)
        
    return close_img

def open_filter(img, rad=10, times=1):
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

def bounding_columns(img, row, up=10, down=10, outliers=5, meth='mean'):
    """
        Find bounding columns of 'row' by looking 'up' and 'down' rows, throwing out top and bottom
        'outliers' values and using 'meth' (default np.mean) on the rest. (Note that 'up' is subtracting rows 
        because image starts from top.)
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
    if meth == 'max':
        begin = np.min(begins[outliers:-outliers])
        end = np.max(ends[outliers:-outliers])
    else:
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

def adjust_bounding_points(points, adjust_ratio=0.8):
    """
        Will adjust the bounding columns by apportioning up the smaller width by the 'adjust_ratio' 
        (larger width stays the same). Input is a numpy array of points [[tbc1, tec1], [tbc2, tec2]...]
    """
    diffs = points[:,1] - points[:,0]
    diff_adj = np.asarray((np.max(diffs) - diffs) * adjust_ratio // 2, dtype=int)
    diff_adj_matrix = np.column_stack((-diff_adj, diff_adj))
    
    return points + diff_adj_matrix
    
def critical_points(imga_dict):
    """
        Takes the (front or back) image array and returns the critical points needed to create zones:
    """
    transforms = OrderedDict()
    transforms[gaussian_filter] = {}
    transforms[threshold] = {}
    transforms[open_filter] = {}
    
    point_dict = OrderedDict()
    # First, determine the torso begin row
    torso_begin_rows = []
    for slice, imga in imga_dict.items():
        rows, columns = imga.shape
        midrow = rows // 2
        
        sigmas = [4, 6]
        rads = [10, 15]
        imgt = imga[midrow:]
        for i in range(len(sigmas)):
            transforms[gaussian_filter] = {'sigma': sigmas[i]}
            transforms[open_filter] = {'rad': rads[i]}
            
            transformed_img = run_transforms(imgt, transforms)
            
            # Get two torso begin rows, looking down when two peaks are hit 
            torso_begin_rows.append(binary_peaks(transformed_img, 2, hits=10) + midrow)
            # Looking up when 1 peak is hit, starting at the 3/4 point (of original image)
            torso_begin_rows.append(binary_peaks(transformed_img[:midrow//2], 1, hits=10, down=False) + midrow)            
        
    print(f"torso_begin_rows(0 and 8): {torso_begin_rows}")
    # Drop 3 high and low and take min of rest
    torso_begin_rows.sort()
    torso_begin_row = min(torso_begin_rows[3:-3])
    print(f"torso_begin_row: {torso_begin_row}")
    
    # Next, get the torso bounding columns
    torso_columns =  []
    for slice, imga in imga_dict.items():
        sigmas = [6]
        rads = [18]
        imgt = imga[midrow:]
        for i in range(len(sigmas)):
            transforms[gaussian_filter] = {'sigma': sigmas[i]}
            transforms[open_filter] = {'rad': rads[i]}
            
            transformed_img = run_transforms(imgt, transforms)
            
            tbc, tec = bounding_columns(transformed_img, torso_begin_row-midrow, up=20, down=5)
            torso_columns.append([tbc, tec])

    print(f"torso_columns: {torso_columns}")
      
    # Finally, the head begin row
    head_begin_rows = []
    for idx, (slice, imga) in enumerate(imga_dict.items()):
        rows, columns = imga.shape
        midrow = rows // 2

        sigmas = [4, 6]
        rads = [10, 15]
        imgt = imga[:midrow, torso_columns[idx][0]:torso_columns[idx][1]]    
        for i in range(len(sigmas)):
            transforms[gaussian_filter] = {'sigma': sigmas[i]}
            transforms[open_filter] = {'rad': rads[i]}
            
            transformed_img = run_transforms(imgt, transforms)
            
            head_begin_rows.extend(nub_head_begin(transformed_img))       

    print(f"head_begin_rows(0 and 8): {head_begin_rows}")
    # Drop high and low and take max of rest
    #TODO: blend this and tbr, or rather use the sample to come up with a good point to use
    head_begin_rows.sort()
    head_begin_row = max(head_begin_rows[3:-3])
       
    torso_columns_adj = adjust_bounding_points(np.asarray(torso_columns))
    print(f"torso_columns_adj: {list(torso_columns_adj)}")
    for idx, (slice, _) in enumerate(imga_dict.items()):
        point_dict[slice] = [torso_begin_row, *torso_columns_adj[idx], head_begin_row]
        
    return point_dict

def critical_points_quarters(imga_dict, torso_begin_row):
    """
        Takes quarter images and gets the torso begin and end columns and the center point based on the torso begin row
    """
    SIDE_MARGIN = 100
    
    point_dict = OrderedDict()
    for slice, imga in imga_dict.items():    
        rows, columns = imga.shape
        midrow = rows // 2
        
        torso_begin_columns = []
        torso_end_columns = []
        
        transforms = OrderedDict()
        transforms[gaussian_filter] = {}
        transforms[threshold] = {}
        transforms[open_filter] = {}
        #transforms[convex_hull] = {}
        
        sigmas = [4, 6]
        rads = [10, 15]
        imgt = imga[midrow:,SIDE_MARGIN:-SIDE_MARGIN]
        for i in range(len(sigmas)):
            transforms[gaussian_filter] = {'sigma': sigmas[i]}
            transforms[open_filter] = {'rad': rads[i]}
            
            transformed_img = run_transforms(imgt, transforms)
            
            mod_tbr = torso_begin_row - midrow
            torso_begin_column, torso_end_column = bounding_columns(transformed_img, mod_tbr, 
                                                                    up=mod_tbr, down=5, outliers=1, meth='max')
            
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

def valid_rect(x1, y1, x2, y2):
     if x1 >= x2 or y1 >= y2:
         return [0,0,0,0]
     else:
        return [x1, y1, x2, y2]
    
def z_depth_adjustment(z_depth):
    if z_depth < 70: return 0
    SLOPE = 0.1
    INTERCEPT = -3.0
    
    return int(round(SLOPE * z_depth + INTERCEPT))

def extract_rects(tbr, tbc, tec, hbr, rows, slice_cursor=0, slice=0, x_rot_adj=0, z_rot_adj=0, z_depth=0):
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
    UPPER_TORSO_SLICE_ADJ = 0.05
    zd_adj = z_depth_adjustment(z_depth)
    print(f"z_depth_adjustment:  {zd_adj}")
    slice_col_adj1 = int(slice_cursor * torso_width * UPPER_TORSO_SLICE_ADJ) + (slice_cursor * zd_adj)

    # Lower torso column movement is determined by a gaussian with these parameters (to account for scanner
    # accelerating then decelerating as it starts and stops).
    G_HEIGHT = 0.1
    G_SIGMA = 5.0
    lower_torso_slice_adj = gaussian_fit(slice, G_HEIGHT, 8, G_SIGMA)
    slice_col_adj2 = int(round(slice_cursor * torso_width * lower_torso_slice_adj)) + (slice_cursor * zd_adj)
    #print(f"slice, adj: {slice}, {lower_torso_slice_adj}")
    #print(f"slice_cursor: {slice_cursor}")
    #print(f"slice_col_adj1: {slice_col_adj1}")
    #print(f"slice_col_adj2: {slice_col_adj2}")
   
    LEG_PORTIONS = 16
    WAIST_EXT_PORTION = 2
    leg_size = rows - tbr
    leg_portion = leg_size // LEG_PORTIONS
    print(f"leg_portion: {leg_portion}") 
    torso_zones_end_row = tbr + WAIST_EXT_PORTION*leg_portion
    
    # LEG_PORTIONS-WAIST_EXT_PORTION portions left to allot to legs (currently 5,5,3)
    TOP_LEG_PORTION = 5
    MID_LEG_PORTION = 5
    mid_leg_start = torso_zones_end_row + TOP_LEG_PORTION*leg_portion
    bottom_leg_start = mid_leg_start + MID_LEG_PORTION*leg_portion
        
    TORSO_PORTIONS = 15
    UPPER_TORSO_PORTION = 4
    LOWER_TORSO_PORTION = 10
    torso_height = tbr - hbr
    torso_unit = torso_height // TORSO_PORTIONS
    
    TOP_MARGIN = 50
    
    torso_split_row = hbr + UPPER_TORSO_PORTION * torso_unit
    waist_split_row = hbr + LOWER_TORSO_PORTION * torso_unit
    #TODO: Don't like this slice hardcoding
    if slice in [3,4,5,11,12,13]:
        torso_rect = valid_rect(tbc_adj, hbr+20, tec_adj, waist_split_row) #TODO: better then +20
        waist_rect = valid_rect(tbc_adj, waist_split_row, tec_adj, torso_zones_end_row)
        arm_rect = valid_rect(tbc_adj, TOP_MARGIN, tec_adj, hbr+20)
        
        leg_top_rect = valid_rect(tbc_adj, torso_zones_end_row, tec_adj, mid_leg_start)
        leg_mid_rect = valid_rect(tbc_adj, mid_leg_start, tec_adj, bottom_leg_start)
        leg_bottom_rect = valid_rect(tbc_adj, bottom_leg_start, tec_adj, rows)
        return torso_rect, waist_rect, arm_rect, leg_top_rect, leg_mid_rect, leg_bottom_rect

    lower_torso_split_column = (torso_width // 2 + tbc_adj) + slice_col_adj1                          

    WAIST_PORTIONS = 15
    SIDE_WAIST_PORTION = 4
    side_waist_size = torso_width // WAIST_PORTIONS * SIDE_WAIST_PORTION
    waist_split_column1 = side_waist_size + tbc_adj + slice_col_adj2
    waist_split_column2 = torso_width - side_waist_size + tbc_adj + slice_col_adj2
    
    #TODO: build rects but have 'valid_rect' method that makes sure rect doesn't have (-) values o/w returns (0,0,0,0),
    # but if padding zone an 'invalid' rect may become valid?
    
    # Note: left/right are as the observer of the image
    # These next two are for the upper chest/upper back adjustment from side body
    #TODO: fine tune this
    utl_adj = slice_col_adj1 if slice_cursor > 0 else 0
    utr_adj = slice_col_adj1 if slice_cursor < 0 else 0
    upper_torso_rect = valid_rect(tbc_adj + utl_adj, hbr, tec_adj + utr_adj, torso_split_row)
    left_torso_rect = valid_rect(tbc_adj, torso_split_row, min(tec_adj, lower_torso_split_column), waist_split_row)
    right_torso_rect = valid_rect(max(tbc_adj, lower_torso_split_column), torso_split_row, tec_adj, waist_split_row)
    
    left_waist_rect = valid_rect(tbc_adj, waist_split_row, max(tbc_adj, waist_split_column1), torso_zones_end_row)
    mid_waist_rect = valid_rect(max(tbc_adj, waist_split_column1), waist_split_row, min(tec_adj, waist_split_column2), torso_zones_end_row)
    right_waist_rect = valid_rect(min(tec_adj, waist_split_column2), waist_split_row, tec_adj, torso_zones_end_row)
    
    torso_midpoint = (tbc_adj + tec_adj) // 2
    
    ARM_HEAD_MARGIN = 20
    left_arm_rect = valid_rect(tbc_adj, TOP_MARGIN, torso_midpoint-ARM_HEAD_MARGIN, hbr)
    right_arm_rect = valid_rect(torso_midpoint+ARM_HEAD_MARGIN, TOP_MARGIN, tec_adj, hbr)
    
    left_top_leg_rect = valid_rect(tbc_adj, torso_zones_end_row, torso_midpoint, mid_leg_start)
    right_top_leg_rect = valid_rect(torso_midpoint, torso_zones_end_row, tec_adj, mid_leg_start)
    left_mid_leg_rect = valid_rect(tbc_adj, mid_leg_start, torso_midpoint, bottom_leg_start)
    right_mid_leg_rect = valid_rect(torso_midpoint, mid_leg_start, tec_adj, bottom_leg_start)
    left_bottom_leg_rect = valid_rect(tbc_adj, bottom_leg_start, torso_midpoint, rows)
    right_bottom_leg_rect = valid_rect(torso_midpoint, bottom_leg_start, tec_adj, rows)
   
    return upper_torso_rect, left_torso_rect, right_torso_rect, left_waist_rect, mid_waist_rect, right_waist_rect, left_arm_rect, right_arm_rect, left_top_leg_rect, left_mid_leg_rect, left_bottom_leg_rect, right_top_leg_rect, right_mid_leg_rect, right_bottom_leg_rect

def create_zones16(file_images):
    """
        Takes the 16 slice file images and returns the zone rectangles by slice
    """
    crit_point_slices = [0, 8]
    crit_point_imga_dict = OrderedDict()
    for slice in crit_point_slices:
        crit_point_imga_dict[slice] = np.asarray(file_images[slice])
    
    crit_point_dict = critical_points(crit_point_imga_dict)

    # tbr and hbr are the same across slices
    c_tbr, c_tbc0, c_tec0, c_hbr = crit_point_dict[0]
    _, c_tbc8, c_tec8, _ = crit_point_dict[8]

    #c_tbr = int(0.8 * max(c_tbr0, c_tbr8) + 0.2 * min(c_tbr0, c_tbr8))
    #c_hbr = (c_hbr0 + c_hbr8) // 2
    
    #c_tbc0a, c_tec0a, c_tbc8a, c_tec8a = adjust_torso_bounds(c_tbc0, c_tec0, c_tbc8, c_tec8)    
    
    crit_point_slices_q = [4, 12]
    crit_point_imga_dict_q = OrderedDict()
    for slice in crit_point_slices_q:
        crit_point_imga_dict_q[slice] = np.asarray(file_images[slice])
    
    # Difference of calculated midpoint from actual midpoint (do I even need this)?
    #TODO: still need this? Is there a better way to calc?
    rows, columns = 660, 512 # TODO: parameterize
    x_rot_adj = (columns // 2) - ((c_tbc0 + c_tec0) // 2)
    print(f"x_rot_adj: {x_rot_adj}")
    #print(f"tbc diff: {c_tbc8-c_tbc0}")
    #print(f"tec diff: {c_tec8-c_tec0}")

    crit_point_dict_q = critical_points_quarters(crit_point_imga_dict_q, c_tbr)

    c_tbc4, c_tec4, c_ctr4 = crit_point_dict_q[4]
    c_tbc12, c_tec12, c_ctr12 = crit_point_dict_q[12]
    
    z_depth = ((c_tec12 - c_tbc12) + (c_tec4 - c_tbc4)) // 2
    print(f"z_depth: {z_depth}")
    
    z_rot_adj = c_ctr12 - c_ctr4
    print(f"z_rot_adj: {z_rot_adj}")
    
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
    zones = np.zeros((16, 18, 4), dtype=np.int32)
        
    zones[0][5], zones[0][6], zones[0][7], zones[0][8], zones[0][9], zones[0][10], zones[0][1], zones[0][3], zones[0][11], zones[0][13], zones[0][15], zones[0][12], zones[0][14], zones[0][16] = extract_rects(c_tbr, c_tbc0, c_tec0, c_hbr, rows)

    zones[1][5], zones[1][6], zones[1][7], zones[1][8], zones[1][9], zones[1][10], zones[1][1], zones[1][3], zones[1][11], zones[1][13], zones[1][15], zones[1][12], zones[1][14], zones[1][16] = extract_rects(c_tbr, c_tbc0, c_tec0, c_hbr, rows, 
                                                                                                                                                                                                                slice_cursor=-1, slice=1, x_rot_adj=x_rot_adj, z_rot_adj=z_rot_adj, z_depth=z_depth)
    zones[2][5], zones[2][6], zones[2][7], zones[2][8], zones[2][9], zones[2][10], zones[2][1], zones[2][3], zones[2][11], zones[2][13], zones[2][15], zones[2][12], zones[2][14], zones[2][16] = extract_rects(c_tbr, c_tbc0, c_tec0, c_hbr, rows,
                                                                                                                                                                                                                slice_cursor=-2, slice=2, x_rot_adj=x_rot_adj, z_rot_adj=z_rot_adj, z_depth=z_depth)
    zones[3][7], zones[3][10], zones[3][3], zones[3][12], zones[3][14], zones[3][16] = extract_rects(c_tbr, c_tbc4, c_tec4, c_hbr, rows, 
                                                                                                     slice_cursor=-1, slice=3, x_rot_adj=x_rot_adj, z_rot_adj=z_rot_adj)
    zones[4][7], zones[4][10], zones[4][3], zones[4][12], zones[4][14], zones[4][16] = extract_rects(c_tbr, c_tbc4, c_tec4, c_hbr, rows, slice=4)
    zones[5][7], zones[5][10], zones[5][3], zones[5][12], zones[5][14], zones[5][16] = extract_rects(c_tbr, c_tbc4, c_tec4, c_hbr, rows,
                                                                                                     slice_cursor=1, slice=5, x_rot_adj=x_rot_adj, z_rot_adj=z_rot_adj)
    
    zones[6][17], zones[6][7], zones[6][6], zones[6][10], zones[6][9], zones[6][8], zones[6][3], zones[6][1], zones[6][12], zones[6][14], zones[6][16], zones[6][11], zones[6][13], zones[6][15] = extract_rects(c_tbr, c_tbc8, c_tec8, c_hbr, rows,
                                                                                                                                                                                                                 slice_cursor=2, slice=6, x_rot_adj=x_rot_adj, z_rot_adj=-z_rot_adj, z_depth=z_depth)
    zones[7][17], zones[7][7], zones[7][6], zones[7][10], zones[7][9], zones[7][8], zones[7][3], zones[7][1], zones[7][12], zones[7][14], zones[7][16], zones[7][11], zones[7][13], zones[7][15]  = extract_rects(c_tbr, c_tbc8, c_tec8, c_hbr, rows,
                                                                                                                                                                                                                  slice_cursor=1, slice=7, x_rot_adj=x_rot_adj, z_rot_adj=-z_rot_adj, z_depth=z_depth)    
    zones[8][17], zones[8][7], zones[8][6], zones[8][10], zones[8][9], zones[8][8], zones[8][3], zones[8][1], zones[8][12], zones[8][14], zones[8][16], zones[8][11], zones[8][13], zones[8][15]  = extract_rects(c_tbr, c_tbc8, c_tec8, c_hbr, rows, slice=8)

    zones[9][17], zones[9][7], zones[9][6], zones[9][10], zones[9][9], zones[9][8], zones[9][3], zones[9][1], zones[9][12], zones[9][14], zones[9][16], zones[9][11], zones[9][13], zones[9][15] = extract_rects(c_tbr, c_tbc8, c_tec8, c_hbr, rows,
                                                                                                                                                                                                                 slice_cursor=-1, slice=9, x_rot_adj=x_rot_adj, z_rot_adj=-z_rot_adj, z_depth=z_depth)
    zones[10][17], zones[10][7], zones[10][6], zones[10][10], zones[10][9], zones[10][8], zones[10][3], zones[10][1], zones[10][12], zones[10][14], zones[10][16], zones[10][11], zones[10][13], zones[10][15] = extract_rects(c_tbr, c_tbc8, c_tec8, c_hbr, rows, 
                                                                                                                                                                                                                               slice_cursor=-2, slice=10, x_rot_adj=x_rot_adj, z_rot_adj=-z_rot_adj, z_depth=z_depth)
    zones[11][6], zones[11][8], zones[11][1], zones[11][11], zones[11][13], zones[11][15] = extract_rects(c_tbr, c_tbc12, c_tec12, c_hbr, rows,
                                                                                                          slice_cursor=-1, slice=11, x_rot_adj=x_rot_adj, z_rot_adj=z_rot_adj)
    zones[12][6], zones[12][8], zones[12][1], zones[12][11], zones[12][13], zones[12][15] = extract_rects(c_tbr, c_tbc12, c_tec12, c_hbr, rows, slice=12)
    zones[13][6], zones[13][8], zones[13][1], zones[13][11], zones[13][13], zones[13][15] = extract_rects(c_tbr, c_tbc12, c_tec12, c_hbr, rows,
                                                                                                          slice_cursor=1, slice=13, x_rot_adj=x_rot_adj, z_rot_adj=z_rot_adj)                                             
    
    zones[14][5], zones[14][6], zones[14][7], zones[14][8], zones[14][9], zones[14][10], zones[14][1], zones[14][3], zones[14][11], zones[14][13], zones[14][15], zones[14][12], zones[14][14], zones[14][16] = extract_rects(c_tbr, c_tbc0, c_tec0, c_hbr, rows,
                                                                                                                                                                                                                             slice_cursor=2, slice=14, x_rot_adj=x_rot_adj, z_rot_adj=z_rot_adj, z_depth=z_depth)
    zones[15][5], zones[15][6], zones[15][7], zones[15][8], zones[15][9], zones[15][10], zones[15][1], zones[15][3], zones[15][11], zones[15][13], zones[15][15], zones[15][12], zones[15][14], zones[15][16] = extract_rects(c_tbr, c_tbc0, c_tec0, c_hbr, rows,
                                                                                                                                                                                                                              slice_cursor=1, slice=15, x_rot_adj=x_rot_adj, z_rot_adj=z_rot_adj, z_depth=z_depth)
    
    # Zeroing out these fringe slice zones to shrink model input. 
    zones[2][6] = zones[2][8] = zones[6][6] = zones[6][8] = zones[10][7] = zones[10][10] = zones[14][7] = zones[14][10] = [0,0,0,0]
    zones[11][11] = zones[11][13] = zones[11][15] = zones[13][11] = zones[13][13] = zones[13][15] = [0,0,0,0]
    zones[3][12] = zones[3][14] = zones[3][16] = zones[5][12] = zones[5][14] = zones[5][16] = [0,0,0,0]
    
    #zones[2][1] = zones[6][1] = zones[10][3] = zones[14][3] = [0,0,0,0]    
    zones[11][1] = zones[13][1] = zones[3][3] = zones[5][3] = [0,0,0,0]
    
    return zones

                        