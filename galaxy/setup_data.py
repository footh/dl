import os
import numpy as np
import math
import shutil
from PIL import Image

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
        
def setup_data(src_dir, numTrain=None, numValid=None, numTest=None, transform=False):
    print('Clearing train directory...')
    remove_files('train/unknown')
    print('Clearing valid directory...')
    remove_files('valid/unknown')
    print('Clearing test directory...')
    remove_files('test/unknown')
    
    src_files = os.listdir(src_dir)
    total_files = len(src_files)
    print('Found %s files' % total_files)
    
    shuffled_files = np.random.permutation(src_files)
    
    # Max of 20% of total is allowed for valid and test sets
    max_non_training = math.floor(total_files * 0.20)
    valid_count = max_non_training
    if numValid is not None:
        valid_count = min(max_non_training, numValid)

    test_count = max_non_training
    if numTest is not None:
        test_count = min(max_non_training, numTest)    
         
    print('Copying %s validation files' % valid_count)
    copy_files(shuffled_files[:valid_count], src_dir, 'valid/unknown', transform=transform)
    
    print('Copying %s test files' % test_count)
    copy_files(shuffled_files[valid_count:(valid_count + test_count)], src_dir, 'test/unknown', transform=transform)
    
    print('Copying %s training files' % (total_files - (valid_count + test_count)))
    copy_files(shuffled_files[(valid_count + test_count):], src_dir, 'train/unknown', transform=transform)
        
        