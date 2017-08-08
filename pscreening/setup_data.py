import os
import numpy as np
import math
import shutil
from PIL import Image, ImageDraw, ImageFont
import scipy.misc
import util
import csv

def remove_files(src):
    if os.path.isfile(src):
        os.unlink(src)
    elif os.path.isdir(src):
        # map lazy evaluates so must wrap in list to force evaluation
        list(map(remove_files, [os.path.join(src, fi) for fi in os.listdir(src)]))
        
def copy_files(files, src_dir, dest_dir, transform=False):
    for f in files:
        if transform:
            imga = np.array(Image.open(os.path.join(src_dir, f)))
            imga = imga[100:-100, 100:-100, :]
            img = Image.fromarray(imga)
            img.save(os.path.join(dest_dir, f))
        else:
            shutil.copy2(os.path.join(src_dir, f), dest_dir)

def shuffled_files(src_dir):
    """
        Returns array of shuffled files (with full path) from source directory
    """
    
    src_files = os.listdir(src_dir)
    total_files = len(src_files)
    print('Found %s files' % total_files)
    src_files = [os.path.join(src_dir, file) for file in src_files]
    
    return np.random.permutation(src_files)
            
def generate_combined(src_dir, num=100, method='max'):
    """
        Generates a combined file from aps files in the src_dir for help in identifying zones.
        This file is serialized as an npy file named 'combinedNUM.npy' where NUM is the number
        of files used in the combination.
    """
    
    shuffled_files = shuffled_files(src_dir)
    
    sample = util.read_data(shuffled_files[0])
    combined = np.zeros(sample.shape) + sample
    for file in shuffled_files[1:num]:
        if method == 'avg':
            combined = combined + util.read_data(file)
        else:
            combined = np.maximum(combined, util.read_data(file))
    
    if method == 'avg':
        combined = combined / num
    
    np.save('combined' + str(num), combined)
    
def generate_image_slices(file, dest_dir='zones'):
    """
        Takes as input a combined file from 'generate_combined' and splits it out into a png file
        per slice into the dest_dir
    """

    print('Clearing zones directory...')
    remove_files('zones')
    
    file_array = np.load(file)
    
    for i in range(file_array.shape[2]):
        img = np.flipud(file_array[:,:,i].transpose())
        scipy.misc.imsave(os.path.join(dest_dir, str(i) + '.png'), img)
    
def slice_zones(zones_file):
    """
        Reads the zones file csv and returns a list of hashes. Each item in the list
        is an image slice containing a hash of the locations of the 17 zones.
    """
    
    with open(zones_file, newline='') as csvfile:
        zone_reader = csv.reader(csvfile)
        headers = next(zone_reader) #skip header row
        #print(headers)

        slice_zone_list = []
        cur_slice_index = -1
        for row in zone_reader:
            slice_index = int(row[0])
            if slice_index != cur_slice_index:
                slice_zone_list.append({})
                
            zone_dict = {}
            zone_dict['rect'] = list(map(int, row[2:6]))
            zone_dict['valid'] = True if row[6] == '' else False
            slice_zone_list[int(row[0])][row[1]] = zone_dict          
            
            cur_slice_index = slice_index
    
    return slice_zone_list

def generate_image_slices_zones(zone_file='zones.csv', src_dir='data', dest_dir='scratch', zones=None, file=None):
    """
        Picks a random (aps) file from src_dir, draws rectangles around each zone (as described by zone_file)
        for each slice in zone_file and writes out to png file in dest_dir. If zone argument is blank, all zones
        are considered, otherwise output is restricted to the argument. zone should be a list of strings a la ['1', '3']
    """    
    #colors = ['orange', 'blue', 'green', 'grey', 'goldenrod', 'pink', 'lightgreen', 'magenta', 'lightblue', 'yellow', 'red', 'brown', 'white', 'turquoise', 'purple', 'cyan']
    remove_files(dest_dir)
    
    slice_zones_list = slice_zones(zone_file)
    
    # Read first file from shuffled list
    if file is None:
        file = shuffled_files(src_dir)[0]
        print(file)
    
    file_data = util.read_data(file)

    for i, slice_dict in enumerate(slice_zones_list):
        img = np.flipud(file_data[:,:,i].transpose())
        img = scipy.misc.toimage(img, channel_axis=2) # This is the key to convert the floats from data to a viewable image. Look at scipy.misc source code, the bytescale method.
        draw = ImageDraw.Draw(img)
        if zones is not None:
            zones = [str(z) for z in zones]
            slice_dict = {k: v for k, v in slice_dict.items() if k in zones}
        for key in slice_dict:
            rect = slice_dict[key]['rect']
            draw.rectangle([rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]], outline='white')
            draw.text((rect[0]+2, rect[1]+2), key, fill='white')            

        del draw
        #scipy.misc.imsave(os.path.join('scratch', str(i) + 'z.png'), np.asarray(img))
        img.save(os.path.join(dest_dir, str(i) + 'z.png'))
    
def label_dict(label_file):
    """
        Reads the label file and returns a dict of {id: [array of 0s and 1s]}. Index of label is (zone - 1).
    """    
    with open(label_file, newline='') as csvfile:
        label_reader = csv.reader(csvfile)
        headers = next(label_reader) #skip header row
        print(headers)

        label_dict = {}
        cur_id = ''
        for row in label_reader:
            id, zone_str = row[0].split('_')
            if id != cur_id:
                label_dict[id] = [0 for i in range(17)]

            zone_idx = int(zone_str.split('Zone')[-1])
            label_dict[id][zone_idx-1] = int(row[1])
            
            cur_id = id
            
        return label_dict

# ----------------------------------------------------------------------------------------------
# algorithmic zone extraction
COLUMN_MARGIN = 30 # Amount of pixels out to start looking at columns (avoiding uninteresting space on the side margins)
    
PIXEL_THRESHOLD = 25 # Pixel intensity value threshold to be considered 'interesting'

WINDOW_SIZE = 10

PEAK_SLOPE_THRESHOLD = 0.1

from sklearn import preprocessing
from scipy import signal
import peakutils

def smooth_curve(row):
    """
        Returns data sans noise
    """ 
    F_WINDOW_LENGTH = 9
    F_POLY_ORDER = 3
    
    return signal.savgol_filter(row, F_WINDOW_LENGTH, F_POLY_ORDER)

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

def convoluted_rows(imga):
    rows, columns = imga.shape
    
    crows = []
    idx = []
    for i in range(rows-1, -1, -WINDOW_SIZE):
        #row = imga[i]
        crow = []
        for j in range(COLUMN_MARGIN, columns - COLUMN_MARGIN, WINDOW_SIZE):
            crow.append(imga[i:i+WINDOW_SIZE, j:j+WINDOW_SIZE].sum())
            
        crows.append(crow)

    return np.asarray(crows)
 
def find_peak_start(crows, num_peaks, hit_threshold=3):
    """
        Returns the first row of input c-rows where num_peaks was found hit_threshold times
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
        if cur_peaks == num_peaks:
            if peak_hits == 0:
                peak_start_row = i
                peak_start_column = cur_peak_indexes[0]
            peak_hits +=1
        else:
            peak_hits = max(0, peak_hits - 1)            
        
        if peak_hits == hit_threshold:
            return peak_start_row, peak_start_column
    
def torso_begin(crows):
    """
        Returns the row of input convoluted rows where the beginning of the torso is detected
    """
    PEAK_HIT_THRESHOLD = 3
    TORSO_PEAK_COUNT = 1
    
    torso_start_row, torso_start_column = find_peak_start(crows,
                                                          TORSO_PEAK_COUNT,
                                                          hit_threshold=PEAK_HIT_THRESHOLD)
    
    # TODO: more thought around adjustments here
    torso_start_row -= 3
    torso_start_column -= 3
        
    return torso_start_row, torso_start_column

def head_begin(crows):
    """
        Returns the row of input convoluted rows head is detected. Starts looking at the begin_row
    """
    PEAK_HIT_THRESHOLD = 2
    HEAD_PEAK_COUNT = 3
    
    head_start_row, head_start_column = find_peak_start(crows,
                                                        HEAD_PEAK_COUNT,
                                                        hit_threshold=PEAK_HIT_THRESHOLD)
    
    # Add back margin
    head_start_row
    
    # TODO: more thought around adjustments here
    head_start_row += 1
        
    return head_start_row, head_start_column
    
def create_zones(file=None, slice=0, src_dir='data'):

    # Read first file from shuffled list
    if file is None:
        file = shuffled_files(src_dir)[0]
        print(file)

    file_data = util.read_data(file)
    img = np.flipud(file_data[:,:,slice].transpose())
    img = scipy.misc.toimage(img, channel_axis=2)
    imga = np.asarray(img)
    
    crows = convoluted_rows(imga)
    
    rows, columns = imga.shape

    TORSO_MARGIN = 10
    HEAD_MARGIN = 10
     
    torso_begin_crow, torso_begin_column = torso_begin(crows[TORSO_MARGIN:])
    torso_begin_row = rows - (torso_begin_crow + TORSO_MARGIN) * WINDOW_SIZE
     
    head_begin_crow, head_begin_column = head_begin(crows[torso_begin_crow + HEAD_MARGIN:])
    head_begin_row = rows - (head_begin_crow + torso_begin_crow + HEAD_MARGIN) * WINDOW_SIZE
     
    draw = ImageDraw.Draw(img)
    for i in range(crows.shape[0]):
        draw.text((2, rows - (i * 10) - 10), str(i), fill='white')            

    draw.line([(0, torso_begin_row), (columns-1, torso_begin_row)], fill='white')
    draw.line([(0, head_begin_row), (columns-1, head_begin_row)], fill='white')
    del draw
#     
    img.save('aaatestfile.png')
#     
#     print("torso_begin_column: %s" % str(torso_begin_column))
#     print("torso_end_row: %s" % str(torso_end_row))
#     return imga
    return imga, crows

#-------------------------------------------------------------------------
# DEBUGGING CODE

import matplotlib.pyplot as plt

clrs = 'brgy'

def p(row):
    plt.plot(range(row.size), row)
    plt.show()
    
def psm(crows, start, count=4):
    p = []
    for i in range(start, start + count):
        fr = smooth_curve(crows[i])
        plt.plot(range(fr.size), fr, clrs[i-start])
        p.append(peaks2(fr))
        
    plt.show()
    return p
    
def pr(crows, start, count=4):  
    p = []
    for i in range(start, start + count):
        plt.plot(range(crows[i].size), crows[i], clrs[i-start])
        p.append(peaks(crows[i]))
        
    plt.show()
    return p    
    
r = create_zones(file='data/402bcaa39d6e36a90bf314207b110fa7.aps')
#peaks(r[40])