import os
import numpy as np
import shutil
from PIL import ImageDraw
import scipy.misc
import util
import csv
import math

import config
import zones as z
import zones_config

ZONE_EXTRACTIONS = [1,3,5,6,7,8,9,10,11,12,13,14,15,16,17]

def get_file_name(file_name):
    return os.path.splitext(os.path.basename(file_name))[0]

def label_dict(label_file='stage1_labels.csv'):
    """
        Reads the label file and returns a dict of {id: [numpy array of 0s and 1s]}. Index of label is (zone - 1).
    """
    full_label_file = os.path.join(config.PSCREENING_HOME, label_file)
    
    with open(full_label_file, newline='') as csvfile:
        label_reader = csv.reader(csvfile)
        headers = next(label_reader) #skip header row
        #print(headers)

        label_dict = {}
        cur_id = ''
        for row in label_reader:
            id, zone_str = row[0].split('_')
            if id != cur_id:
                label_dict[id] = np.array([0 for i in range(17)])

            zone_idx = int(zone_str.split('Zone')[-1])
            label_dict[id][zone_idx-1] = int(row[1])
            
            cur_id = id
            
        return label_dict    

def get_zones(id):
    d = label_dict()
    return np.where(d[id])[0] + 1

def shuffled_files(src):
    """
        Returns array of shuffled files (with full path) from source (train, valid, test, etc.)
    """
    full_src_dir = os.path.join(config.PSCREENING_HOME, config.RAW_DATA_DIR, src)
    
    src_files = os.listdir(full_src_dir)
    src_files = [os.path.join(full_src_dir, file) for file in src_files if os.path.isfile(os.path.join(full_src_dir, file))]
    total_files = len(src_files)
    print('Found %s files' % total_files)
    
    return np.random.permutation(src_files)
                        
def generate_combined(src='all', num=None, method='avg', img_scale=False):
    """
        Generates a combined file from aps files in the src_dir for help in identifying zones.
        This file is serialized as an npy file named 'combinedNUM.npy' where 'num' is the number
        of files used in the combination. If 'num' is not given, all files are used.
    """
    files = shuffled_files(src)
    if num is None:
        num = len(files)
    
    sample = np.asarray(util.read_data(files[0]))
    combined = np.zeros(sample.shape)
    for file in files[0:num]:
        file_data = np.asarray(util.read_data(file))
        if img_scale:
            file_data = scipy.misc.bytescale(file_data)
        
        if method == 'avg':
            combined = combined + file_data
        else:
            combined = np.maximum(combined, file_data)
    
    if method == 'avg':
        combined = combined / num
    # np.sum(combined, axis=(1,2))
    
    np.save(f"combined-{src}-{num}", combined)
    
def generate_images(src='all', img_perms=10):
    zone_dict = {
        6: ([0,8,12], [10,14])
    }

    files = shuffled_files(src)
    for file in files:
        print(f"noop")
           
def __remove_files(src):
    if os.path.isfile(src):
        os.unlink(src)
    elif os.path.isdir(src):
        # map lazy evaluates so must wrap in list to force evaluation
        list(map(__remove_files, [os.path.join(src, fi) for fi in os.listdir(src)]))
        
def __copy_files(files, src_dir, dest_dir, ext='a3daps', to_npy=False):
    for f in files:
        full_file_path = os.path.join(src_dir, f + '.' + ext)
        if to_npy:
            file_data = np.asarray(util.read_data(full_file_path))
            np.save(os.path.join(dest_dir, f + '.npy'), file_data)
        else:
            shutil.copy2(full_file_path, dest_dir)

def setup_data(num_valid=None, num_test=None, ext='a3daps', to_npy=False):
    """
        Moves the unlabeled files to submission dir, and the rest to train, valid and test directories
        Usage: setup_data('data','stage1_labels.csv', num_valid=100, num_test=100)
    """    
    all_src_dir = os.path.join(config.PSCREENING_HOME, config.RAW_DATA_DIR, 'all')
    train_dir = os.path.join(config.PSCREENING_HOME, config.RAW_DATA_DIR, 'train')
    valid_dir = os.path.join(config.PSCREENING_HOME, config.RAW_DATA_DIR, 'valid')
    test_dir = os.path.join(config.PSCREENING_HOME, config.RAW_DATA_DIR, 'test')
    submission_dir = os.path.join(config.PSCREENING_HOME, config.RAW_DATA_DIR, 'submission')

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
    
    label_keys = list(label_dict().keys())
    labeled_count = len(label_keys)
    print('Found %s labeled files' % labeled_count)
    
    # files not labeled are submission data
    submission_keys = list(set(src_files) - set(label_keys))
    print('Copying %s submission files' % len(submission_keys))
    __copy_files(submission_keys, all_src_dir, submission_dir, ext=ext, to_npy=to_npy)    
    
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
    __copy_files(shuffled_files[:valid_count], all_src_dir, valid_dir, ext=ext, to_npy=to_npy)
     
    print('Copying %s test files' % test_count)
    __copy_files(shuffled_files[valid_count:(valid_count + test_count)], all_src_dir, test_dir, ext=ext, to_npy=to_npy)
     
    print('Copying %s training files' % (labeled_count - (valid_count + test_count)))
    __copy_files(shuffled_files[(valid_count + test_count):], all_src_dir, train_dir, ext=ext, to_npy=to_npy)

def points_file(src='train', padding=False):
    """
        Creates points file for the given 'src' files ('train', 'valid', 'test', etc)
    """
    
    files = shuffled_files(src)
    file = os.path.join(config.PSCREENING_HOME, 'points-' + src + '.csv')
    with open(file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        f_count = 0
        for f in files:
            f_count += 1
            print(f"Reading file {f}...")
            file_images = util.read_data(f, as_images=True)
            print(f"Creating zones...")
            zone_rects = z.create_zones16(file_images)
            if padding:
                zones_config.apply_padding(zone_rects)
            print(f"Write record...")
            for i in range(16):
                row = [[f], [i]] + [list(zone_rects[i][j]) for j in ZONE_EXTRACTIONS]
                row = [val for sublist in row for val in sublist]
                writer.writerow(row)
            print(f"Record #{f_count} completed")
    
def zones_max_dict(file='points-all.csv', slice_count=16, area_threshold=0, round_up=False):
    """
        Returns a dict of zone => 3-tuple of (valid_slices, max_height, max_width)
        Calculates these values from the passed in points file
    """
    file = os.path.join(config.PSCREENING_HOME, file)
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        all_rows = np.array(list(reader))
        zone_rects = np.array(all_rows[:,2:], dtype=np.int32)
        
        zones_max = {}
        for i, z in enumerate(ZONE_EXTRACTIONS):
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
    
def _valid_rects(zone_idx, sample_chunk, area_threshold=0):
    """
      From a sample chunk (points file output for a given sample), the rects from the zone_idx are extracted, removing any
      that are below the area threshold.
    """
    
    # zone_rects starts as a matrix of all slices + rects. The area is calculated and zone_rects is
    # collapsed to only the rects that pass the area_threshold    
    zone_rects = np.array(np.hstack((sample_chunk[:,1:2], sample_chunk[:,2+4*zone_idx:6+4*zone_idx])), dtype=np.int32)
    a = (zone_rects[:,4] - zone_rects[:,2]) * (zone_rects[:,3] - zone_rects[:,1])
    zone_rects = zone_rects[a > area_threshold]
    
    return zone_rects
    
def extract_zones(src='train', sample_file='points-all.csv', slice_count=16, 
                  area_threshold=0, overwrite=True, start=0, 
                  img_scale=True, mean_file=None):
    """
        For zones 'src', uses the associated points file to extract the zones and save them as numpy arrays in a directory by zone 
    """
    file = os.path.join(config.PSCREENING_HOME, 'points-' + src + '.csv')
    
    zones_max = zones_max_dict(file=sample_file, slice_count=slice_count, area_threshold=area_threshold, round_up=True)
    full_dest_dir = os.path.join(config.PSCREENING_HOME, config.TRAINING_DIR, src)
    
    if not os.path.exists(full_dest_dir):
        os.mkdir(full_dest_dir)

    if mean_file is not None:
        mean_file = np.load(os.path.join(config.PSCREENING_HOME, mean_file))
    
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        
        all_rows = np.array(list(reader))
        all_rows = all_rows.reshape(all_rows.shape[0] // slice_count, slice_count, all_rows.shape[1])
        total_rows = all_rows.shape[0]
        
        cnt = start
        for row in all_rows[start:total_rows]:

            file_data = util.read_data(row[0, 0])
            if img_scale or mean_file is not None:
                file_data = scipy.misc.bytescale(np.asarray(file_data))
                if mean_file is not None:
                    file_data = file_data - mean_file
            
            id = get_file_name(row[0, 0])
            for i in range(len(ZONE_EXTRACTIONS)):
                zone_dir = os.path.join(full_dest_dir, str(ZONE_EXTRACTIONS[i]))
                if not os.path.exists(zone_dir):
                    os.mkdir(zone_dir)
                file_name = os.path.join(zone_dir, id) + '.npy'
                if not overwrite and os.path.exists(file_name):
                    print(f"File {file_name} already exists. Skipping!")
                    continue
                
                zone_rects = _valid_rects(i, row, area_threshold=area_threshold)
                
                slice_data = np.zeros(zones_max[ZONE_EXTRACTIONS[i]], dtype=np.float32)
                for j in range(zone_rects.shape[0]):
                    rb = zone_rects[j,2]
                    re = zone_rects[j,4]
                    cb = zone_rects[j,1]
                    ce = zone_rects[j,3]
                    slice_data[j][0:re-rb,0:ce-cb] = np.asarray(file_data[zone_rects[j,0]][rb:re,cb:ce])
                    
                np.save(file_name, slice_data)
            
            cnt += 1
            print(f"Finished row {cnt}")
            
def sample_dict(all_file='points-all.csv', slice_count=16, zone=None, area_threshold=0):
    """
        Gets a dict of sample id => sample chunk from points file. If zone is given, then only the slice and rects will be returned
        and collapsed to remove rects under the area_threshold
    """
    all_file = os.path.join(config.PSCREENING_HOME, all_file)
    sample_dict = {}
    with open(all_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',') 
        all_rows = np.array(list(reader))
        sample_chunks = all_rows.reshape(all_rows.shape[0] // slice_count, slice_count, all_rows.shape[1])
        
        for sample_chunk in sample_chunks:
            id = get_file_name(sample_chunk[0, 0])
            if zone is not None:
                zone_idx = ZONE_EXTRACTIONS.index(zone)
                sample_chunk = _valid_rects(zone_idx, sample_chunk, area_threshold=area_threshold)
                
            sample_dict[id] = sample_chunk
            
    return sample_dict
    

def generate_points_files(dirs=['train', 'valid', 'test', 'submission']):
    """
      Using the main points file (points-all.csv) generates all the other points files in 'dirs' based on what samples are in
      their respective raw-data directory
    """
    sample_dict = sample_dict()
            
    for dir in dirs:
        ids = [sample_id_from_file(f) for f in shuffled_files(dir)]
        
        points = np.asarray([sample_dict[id] for id in ids])
        points = points.reshape(points.shape[0] * points.shape[1], points.shape[2])
        points = np.char.replace(points, f"{config.RAW_DATA_DIR}/all", f"{config.RAW_DATA_DIR}/{dir}")
        
        points_file = os.path.join(config.PSCREENING_HOME, f"points-{dir}.csv")
        with open(points_file, 'w', newline='\n') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for row in points:
                writer.writerow(row)
                
def zone_info(zones=17):
    d = label_dict()
    ttl = len(d)
    
    labels = []
    label_probs = []
    for i in range(zones):
        labels.append([label for label,vals in d.items() if vals[i]])
        label_probs.append(len(labels[i]) / ttl)
        
    return labels, label_probs
