import os
import numpy as np
import shutil
from PIL import ImageDraw
import scipy.misc
import util
import csv
import math
import zones as z
import zones_config

PSCREENING_HOME = os.getenv('PSCREENING_HOME', '.')
RAW_DATA_DIR = 'raw-data'
TRAINING_DIR = 'training'

def label_dict(label_file='stage1_labels.csv'):
    """
        Reads the label file and returns a dict of {id: [array of 0s and 1s]}. Index of label is (zone - 1).
    """
    full_label_file = os.path.join(PSCREENING_HOME, label_file)
    
    with open(full_label_file, newline='') as csvfile:
        label_reader = csv.reader(csvfile)
        headers = next(label_reader) #skip header row
        #print(headers)

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
    
_LABEL_DICT_ = label_dict()

def get_zones(id):
    return np.where(_LABEL_DICT_[id])[0] + 1

def shuffled_files(src):
    """
        Returns array of shuffled files (with full path) from source (train, valid, test, etc.)
    """
    full_src_dir = os.path.join(PSCREENING_HOME, RAW_DATA_DIR, src)
    
    src_files = os.listdir(full_src_dir)
    src_files = [os.path.join(full_src_dir, file) for file in src_files if os.path.isfile(os.path.join(full_src_dir, file))]
    total_files = len(src_files)
    print('Found %s files' % total_files)
    
    return np.random.permutation(src_files)
            
def __remove_files(src):
    if os.path.isfile(src):
        os.unlink(src)
    elif os.path.isdir(src):
        # map lazy evaluates so must wrap in list to force evaluation
        list(map(__remove_files, [os.path.join(src, fi) for fi in os.listdir(src)]))
        
def __copy_files(files, src_dir, dest_dir, ext='aps'):
    for f in files:
        shutil.copy2(os.path.join(src_dir, f + '.' + ext), dest_dir)

def setup_data(num_valid=None, num_test=None, ext='aps'):
    """
        Moves the unlabeled files to submission dir, and the rest to train, valid and test directories
        Usage: setup_data('data','stage1_labels.csv', num_valid=100, num_test=100)
    """    
    all_src_dir = os.path.join(PSCREENING_HOME, RAW_DATA_DIR, 'all')
    train_dir = os.path.join(PSCREENING_HOME, RAW_DATA_DIR, 'train')
    valid_dir = os.path.join(PSCREENING_HOME, RAW_DATA_DIR, 'valid')
    test_dir = os.path.join(PSCREENING_HOME, RAW_DATA_DIR, 'test')
    submission_dir = os.path.join(PSCREENING_HOME, RAW_DATA_DIR, 'submission')

    print('Clearing train directory...')
    __remove_files(train_dir)
    print('Clearing valid directory...')
    __remove_files(valid_dir)
    print('Clearing test directory...')
    __remove_files(test_dir)
    print('Clearing submission directory...')
    __remove_files(submission_dir)
    
    src_files = os.listdir(all_src_dir)
    src_files = [f.split('.')[0] for f in src_files]
    total_files = len(src_files)
    print('Found %s files' % total_files)
    
    label_keys = list(_LABEL_DICT_.keys())
    labeled_count = len(label_keys)
    print('Found %s labeled files' % labeled_count)
    
    # files not labeled are submission data
    submission_keys = list(set(src_files) - set(label_keys))
    print('Copying %s submission files' % len(submission_keys))
    __copy_files(submission_keys, all_src_dir, submission_dir, ext=ext)    
    
    # shuffle the rest
    shuffled_files = np.random.permutation(label_keys)
    
    # Max of 20% of total is allowed for valid and test sets
    max_non_training = math.floor(labeled_count * 0.20)
    valid_count = max_non_training
    if num_valid is not None:
        valid_count = min(max_non_training, num_valid)
 
    test_count = max_non_training
    if num_test is not None:
        test_count = min(max_non_training, num_test)
         
    print('Copying %s validation files' % valid_count)
    __copy_files(shuffled_files[:valid_count], all_src_dir, valid_dir, ext=ext)
     
    print('Copying %s test files' % test_count)
    __copy_files(shuffled_files[valid_count:(valid_count + test_count)], all_src_dir, test_dir, ext=ext)
     
    print('Copying %s training files' % (labeled_count - (valid_count + test_count)))
    __copy_files(shuffled_files[(valid_count + test_count):], all_src_dir, train_dir, ext=ext)

def points_file(src='train', padding=False):
    """
        Creates points file for the given 'src' files ('train', 'valid', 'test', etc)
    """
    
    files = shuffled_files(src)
    file = os.path.join(PSCREENING_HOME, 'points-' + src + '.csv')
    with open(file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        f_count = 0
        for f in files:
            f_count += 1
            print(f"Reading file {f}...")
            file_images = util.read_data(f, as_images=True)
            print(f"Creating zones...")
            zones = z.create_zones16(file_images)
            if padding:
                zones_config.apply_padding(zones)
            print(f"Write record...")
            for i in range(16):
                row = [[f], [i], list(zones[i][5]), list(zones[i][6]), list(zones[i][7]), list(zones[i][8]), list(zones[i][9]), list(zones[i][10]), list(zones[i][17])]
                row = [val for sublist in row for val in sublist]
                writer.writerow(row)
            print(f"Record #{f_count} completed")
    
def zones_max_dict(file='points-all.csv', slice_count=16, zones=[5,6,7,8,9,10,17], area_threshold=0, round_up=False):
    """
        Returns a dict of zone => 3-tuple of (valid_slices, max_height, max_width)
        Calculates these values from the passed in points file
    """
    file = os.path.join(PSCREENING_HOME, file)    
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        all_rows = np.array(list(reader))
        zone_rects = np.array(all_rows[:,2:], dtype=np.int32)
        
        zones_max = {}
        for i, z in enumerate(zones):
           w = zone_rects[:,2+4*i] - zone_rects[:,0+4*i]
           h = zone_rects[:,3+4*i] - zone_rects[:,1+4*i]
           
           # Getting boolean array of areas greater than 'area_threshold'. Reshaping to get valid areas by slice and summing.
           # The max of the resulting array is the max number of slices needed in the extraction for that zone.
           a = w * h > area_threshold
           a = a.reshape(a.shape[0] // slice_count, slice_count)
           a = np.sum(a, axis=1)
           
           h_idx = np.argmax(h)
           w_idx = np.argmax(w)
           print(f"zone {z} max h is {all_rows[h_idx][0]}")
           print(f"zone {z} max w is {all_rows[w_idx][0]}")
           
           zones_max[z] = (np.max(a), h[h_idx], w[w_idx])
        
        if round_up:
            def roundup10(x):
                return int(math.ceil(x / 10.0)) * 10
            
            for k, v in zones_max.items():
                zones_max[k] = (v[0], roundup10(v[1]), roundup10(v[2]))
                           
        return zones_max
    
def extract_zones(src='train', sample_file='points-all.csv', slice_count=16, zones=[5,6,7,8,9,10,17], area_threshold=0):
    """
        For zones 'src', uses the associated points file to extract the zones and save them as numpy arrays in a directory by zone 
    """
    file = os.path.join(PSCREENING_HOME, 'points-' + src + '.csv')
    
    zones_max = zones_max_dict(file=sample_file, slice_count=slice_count, zones=zones, area_threshold=area_threshold, round_up=True)
    full_dest_dir = os.path.join(PSCREENING_HOME, 'training', src)
    
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        
        all_rows = np.array(list(reader))
        all_rows = all_rows.reshape(all_rows.shape[0] // slice_count, slice_count, all_rows.shape[1])
        
        cnt = 0
        for row in all_rows:

            file_data = util.read_data(row[0, 0])
            id = os.path.basename(row[0, 0]).split('.')[0]
            for i in range(len(zones)):
                # zone_rects starts as a matrix of all slices + rects. The area is calculated and zone_rects is
                # collapsed to only the rects that pass the area_threshold
                zone_rects = np.array(np.hstack((row[:,1:2], row[:,2+4*i:6+4*i])), dtype=np.int32)
                a = (zone_rects[:,4] - zone_rects[:,2]) * (zone_rects[:,3] - zone_rects[:,1])
                zone_rects = zone_rects[a > area_threshold]
                
                slice_data = np.zeros(zones_max[zones[i]])
                for j in range(zone_rects.shape[0]):
                    rb = zone_rects[j,2]
                    re = zone_rects[j,4]
                    cb = zone_rects[j,1]
                    ce = zone_rects[j,3]
                    slice_data[j][0:re-rb,0:ce-cb] = np.asarray(file_data[zone_rects[j,0]][rb:re,cb:ce])
                    
                np.save(os.path.join(full_dest_dir, str(zones[i]), id), slice_data)
            
            cnt += 1
            print(f"Finished row {cnt}")

