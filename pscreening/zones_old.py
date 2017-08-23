import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import misc, signal
from setup_data import shuffled_files, label_dict
import util
import os
import peakutils

COLUMN_MARGIN = 30 # Amount of pixels out to start looking at columns (avoiding uninteresting space on the side margins)
    
WINDOW_SIZE = 10 # Size in pixels of original image to convolute the image to. Ex. 500x600 would go to 50x60

PEAK_SLOPE_THRESHOLD = 0.15

# Curve smoothing parameters
F_WINDOW_LENGTH = 9
F_POLY_ORDER = 3

def smooth_curve(row, win_len=F_WINDOW_LENGTH, order=F_POLY_ORDER):
    """
        Returns data sans noise
    """ 

    window_length = win_len
    if row.shape[0] < window_length:
        window_length = row.shape[0] // 2
        if window_length % 2 == 0:
            window_length += 1
    
    return signal.savgol_filter(row, window_length, order)

def gaussian_fit(row):
    """
        Attempts to fit a Gaussian on the input row and return the amplitude, mean and standard deviation.
        Returns all 0s if failed.
    """ 

    srow = smooth_curve(row)
    amp = mean = std = 0
    try:    
        amp, mean, std = peakutils.peak.gaussian_fit(np.asarray(range(srow.size)), srow, center_only=False)               
    except:
        pass
    return amp, mean, std

def peaks2(row, slope_threshold=PEAK_SLOPE_THRESHOLD):
    """
        Returns the number of peaks detected in a row of data, and the indices of the peaks
    """ 
    PEAK_DIST_MIN = 5

    frow = smooth_curve(row)
    peak_arr = peakutils.peak.indexes(frow, thres=PEAK_SLOPE_THRESHOLD, min_dist=PEAK_DIST_MIN)
    
    return len(peak_arr), peak_arr   

def peaks(row, slope_threshold=0.3): # 0.3 was the original amount
    """
        Returns the number of peaks detected in a row of data
    """
    #slopes = np.asarray(row, dtype=np.int64)
    from sklearn import preprocessing
    slopes = preprocessing.normalize([row], norm='l2')[0]
    slopes = [(data[1]-data[0]) for data in zip(slopes[:], slopes[1:])]
    
    peak_count = 0
    peak_sum = 0
    peak_top = 0
    peak_hit = False
    for slope in slopes:
        peak_sum += slope
        peak_sum = max(0, peak_sum)
        
        if peak_hit:
            if peak_top - peak_sum > slope_threshold:
                peak_count += 1            
                peak_sum = 0
                peak_hit = False
                peak_top = 0
        
        if peak_sum > slope_threshold:
            peak_hit = True
            if peak_sum > peak_top:
                peak_top = peak_sum            

    return peak_count

def convoluted_rows(imga, window_size=WINDOW_SIZE):
    """
        Will shrink image dimensions by a factor of 'window_size' by summing all pixel 
        values in window_size X window_size square. Also reverses the rows (starts from bottom)
    """
        
    rows, columns = imga.shape
    
    crows = []
    idx = []
    for i in range(rows-1, -1, -window_size):
        #row = imga[i]
        crow = []
        for j in range(0, columns, window_size):
            crow.append(imga[i:i+window_size, j:j+window_size].sum())
            
        crows.append(crow)

    return np.asarray(crows)

def image_bounding_columns(crow):
    """
        Returns the columns of convoluted row where start and end of image is detected
        TODO: make this better
    """
    #print(crow)
    HIT_THRESHOLD = 3
    HIST_BINS = 15
    HIST_BIN_THRESHOLD = 4
    count, bins = np.histogram(crow, bins=HIST_BINS)
    threshold = bins[HIST_BIN_THRESHOLD]
    
    image_start = 0
    for i in range(crow.size):
        if np.sum(crow[i:i+HIT_THRESHOLD] > threshold) == HIT_THRESHOLD:
            image_start = i
            break
    
    image_end = 0
    for i in range(crow.size-1, -1, -1):
        if np.sum(crow[i-HIT_THRESHOLD:i] > threshold) == HIT_THRESHOLD:
            image_end = i
            break
    return image_start, image_end
 
def find_peak_start(crows, peak_func, hit_threshold=3):
    """
        Returns the first row of input c-rows where peak_funk returns true hit_threshold times
        out of hit_threshold + n (depending on size of hit_threshold). Also returns the column where the first peak started
        TODO: better logic around hit_threshold, keep it small for now which will do n out of n+1 
    """
    rows, columns = crows.shape

    peak_hits = 0
    peak_start_row = 0
    peak_start_column = 0
    for i in range(rows):
        row = crows[i]
        cur_peaks, cur_peak_indexes = peaks2(row)
        if peak_func(cur_peaks):
            if peak_hits == 0:
                peak_start_row = i
                peak_start_column = cur_peak_indexes[0]
            peak_hits +=1
        else:
            peak_hits = max(0, peak_hits - 1)            
        
        if peak_hits == hit_threshold:
            return peak_start_row, peak_start_column
        
    return rows - 1, columns - 1
    
def torso_begin(crows):
    """
        Returns the row of input convoluted rows where the beginning of the torso is detected
        and the columns where the image starts and ends
    """
    PEAK_HIT_THRESHOLD = 3
    TORSO_PEAK_COUNT = 1
    peak_func = lambda cur_peaks: cur_peaks == TORSO_PEAK_COUNT
    
    torso_start_row, torso_start_column = find_peak_start(crows,
                                                          peak_func,
                                                          hit_threshold=PEAK_HIT_THRESHOLD)
    
    torso_start_row -= 3 # TODO: better adjustment?

    return torso_start_row, torso_start_column

def head_begin(crows, torso_begin_column, torso_end_column):
    """
        Returns the row of input convoluted rows head is detected. Starts looking at the begin_row
    """
    PEAK_HIT_THRESHOLD = 2
    HEAD_PEAK_COUNT = 3
    peak_func = lambda cur_peaks: cur_peaks >= HEAD_PEAK_COUNT
    
    head_begin_row, head_begin_column = find_peak_start(crows,
                                                        peak_func,
                                                        hit_threshold=PEAK_HIT_THRESHOLD)
    
    print(f"---------------head start row (1): {head_begin_row}")
    print(f"torso start/end: {(torso_begin_column, torso_end_column)}")
    
    LOOK_UP_NUM = 6 
    HIST_BINS = 8
    HIT_THRESHOLD = 3
    
    rows, columns = crows.shape
    
    examined_space = crows[head_begin_row:head_begin_row+LOOK_UP_NUM, torso_begin_column:torso_end_column]
    #count, bins = np.histogram(examined_space, bins=HIST_BINS)
    #print((count, bins))
    
    PERCENTILE_THRESHOLD = 52.0
    low_intensity_threshold = np.percentile(examined_space, PERCENTILE_THRESHOLD)    
        
    # Looking up rows and attempting to adjust to beginning of head by looking at average pixel
    # intensity in a range of pixels near shoulder and finding a low intensity spot. Need a certain
    # amount of hits to make adjustment
    HIST_BIN_LOW_THRESHOLD = 1
    RIGHT_MARGIN = 2
    CHECK_RANGE_LOW = 3

    #low_intensity_threshold = bins[HIST_BIN_LOW_THRESHOLD]
    print(f"low_intensity_threshold: {low_intensity_threshold}")

    low_intensity_hits = 0
    low_intensity_start = 0
    for i in range(LOOK_UP_NUM):
        edge_avg_intensity = np.mean(examined_space[i:i+CHECK_RANGE_LOW,RIGHT_MARGIN:RIGHT_MARGIN+CHECK_RANGE_LOW])
        print(f"edge avg intensity: {edge_avg_intensity}")
        if edge_avg_intensity < low_intensity_threshold:
            if low_intensity_hits == 0:
                low_intensity_start = i
            low_intensity_hits += 1
        else:
            low_intensity_hits = 0
            low_intensity_start = 0
            
        if low_intensity_hits == HIT_THRESHOLD:
            head_begin_row += low_intensity_start
            break

    print(f"---------------head start row (2): {head_begin_row}")
    
    return head_begin_row, head_begin_column

def torso_bounding_columns(crows):
    begin_column = 9999
    end_column = 0
    for i in range(crows.shape[0]):
        image_begin, image_end = image_bounding_columns(crows[i])
        begin_column = min(image_begin, begin_column)
        end_column = max(image_end, end_column)
        
    return begin_column, end_column

def critical_points(crows):
    """
        Takes the image array and returns the critical points needed to create zones:
        torso begin row and columns, head begin row, TODO: image end row
    """
    TORSO_MARGIN = 10
    HEAD_MARGIN = 10
     
    torso_begin_crow, _ = torso_begin(crows[TORSO_MARGIN:])
    torso_begin_crow += TORSO_MARGIN # Add back the margin to get value from the beginning
  
    torso_begin_ccolumn, torso_end_ccolumn = image_bounding_columns(crows[torso_begin_crow])
    
    print(torso_begin_crow) 
    head_begin_crow, _ = head_begin(crows[torso_begin_crow + HEAD_MARGIN:], torso_begin_ccolumn, torso_end_ccolumn)
    head_begin_crow += (torso_begin_crow + HEAD_MARGIN)
    print(head_begin_crow)
    
    return torso_begin_crow, torso_begin_ccolumn, torso_end_ccolumn, head_begin_crow # TODO: add image end row (where hands end)    

def create_zones(file=None, slices=[], src_dir='train', save_file='zone_test'):

    # Read first file from shuffled list
    if file is None:
        file = shuffled_files(src_dir)[0]
        print(file)

    file_data = util.read_data(file)
    
    if 0 not in slices:
        slices.insert(0, 0)
    
    slices.sort()
    
    crows = None
    slice0_torso_begin_crow=0    
    slice0_torso_begin_row = 0
    slice0_head_begin_crow=0
    slice0_head_begin_row = 0
    for slice in slices:
        # Start with the base slice
        img = np.flipud(file_data[:,:,slice].transpose())
        img = misc.toimage(img, channel_axis=2)
        imga = np.asarray(img)
        rows, columns = imga.shape        
    
        # This remainder will make sure the last convolution has the exact amount of columns for the window size
        remainder = WINDOW_SIZE - ((columns-2*COLUMN_MARGIN) % WINDOW_SIZE)
        imga_view = imga[:,COLUMN_MARGIN:columns-COLUMN_MARGIN+remainder]
        crows = convoluted_rows(imga_view)
    
        torso_begin_column = 0
        torso_end_column = 0

        if slice == 0:
            slice0_torso_begin_crow, torso_begin_ccolumn, torso_end_ccolumn, slice0_head_begin_crow = critical_points(crows)
            
            slice0_torso_begin_row = rows - (slice0_torso_begin_crow * WINDOW_SIZE)
            torso_begin_column = torso_begin_ccolumn * WINDOW_SIZE + COLUMN_MARGIN
            torso_end_column = torso_end_ccolumn * WINDOW_SIZE + COLUMN_MARGIN
            slice0_head_begin_row = rows - (slice0_head_begin_crow * WINDOW_SIZE)        
            print(f"torso width: {torso_end_column - torso_begin_column}")
        else:
            torso_begin_ccolumn, torso_end_ccolumn = torso_bounding_columns(crows[slice0_torso_begin_crow:slice0_head_begin_crow,:])
            torso_begin_column = torso_begin_ccolumn * WINDOW_SIZE + COLUMN_MARGIN
            torso_end_column = torso_end_ccolumn * WINDOW_SIZE + COLUMN_MARGIN

        torso_size = slice0_head_begin_row - slice0_torso_begin_row
        torso_unit = torso_size // 15
        zone_5_endrow = slice0_head_begin_row - 4 * torso_unit
        zone_67_endrow = slice0_head_begin_row - 10 * torso_unit
        torso_width = torso_end_column - torso_begin_column
        zone_67_column = (torso_width // 2 + torso_begin_column) - int(slice * torso_width * 0.1)

        draw = ImageDraw.Draw(img)
        # drawing crow numbers
        for i in range(crows.shape[0]):
            draw.text((2, rows - (i * 10) - 10), str(i), fill='white')            
    
        #draw.line([(0, torso_begin_row), (columns-1, torso_begin_row)], fill='white')
        draw.line([(0, slice0_head_begin_row), (columns-1, slice0_head_begin_row)], fill='white')
        draw.line([(torso_begin_column, slice0_torso_begin_row), (torso_begin_column, slice0_head_begin_row)], fill='white')
        draw.line([(torso_end_column, slice0_torso_begin_row), (torso_end_column, slice0_head_begin_row)], fill='white')
        
        #zone 5 bottom
        draw.line([(torso_begin_column, zone_5_endrow), (torso_end_column, zone_5_endrow)], fill='white')
        #zone 6/7 split
        draw.line([(zone_67_column, zone_5_endrow), (zone_67_column, zone_67_endrow)], fill='white')
        #zone 6/7 end
        draw.line([(torso_begin_column, zone_67_endrow), (torso_end_column, zone_67_endrow)], fill='white')
        
        del draw
        
        if save_file is not None:
            img.save(os.path.join('zones', save_file + str(slice) + '.png'))
        
    return crows

#-------------------------------------------------------------------------
# DEBUGGING CODE

import matplotlib.pyplot as plt

clrs = 'brgy'

def p(row):
    plt.plot(range(row.size), row)
    plt.show()
    
def psm(crows, start, count=4, win_len=F_WINDOW_LENGTH, order=F_POLY_ORDER):
    p = []
    for i in range(start, start + count):
        fr = smooth_curve(crows[i], win_len, order)
        plt.plot(range(fr.size), fr, clrs[i-start])
        p.append(peaks2(crows[i]))
        
    plt.show()
    return p
    
def pr(crows, start, count=4):  
    p = []
    for i in range(start, start + count):
        plt.plot(range(crows[i].size), crows[i], clrs[i-start])
        p.append(peaks(crows[i]))
        
    plt.show()
    return p

def points_file(src_dir='train'):
    import csv
    files = shuffled_files(src_dir)
    with open('points.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for f in files:
            file_data = util.read_data(f)
            img = np.flipud(file_data[:,:,0].transpose())
            img = misc.toimage(img, channel_axis=2)
            imga = np.asarray(img)
            tbr, tbc, tec, hbr, _ = critical_points(imga)
            writer.writerow([f, tbr, tbc, tec, hbr])
    
def sst(file, tm=20):
    import cv2
    from scipy import stats
    
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = stats.threshold(img, threshmin=tm, newval=0)
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
     
    cv2.imwrite('zones/clahe_2.png',cl1)
        
#r = create_zones(file='data/402bcaa39d6e36a90bf314207b110fa7.aps')
#peaks(r[40])