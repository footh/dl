import os
import numpy as np
import math
import shutil
from PIL import Image
from PIL import ImageDraw
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
    src_files = os.listdir(src_dir)
    total_files = len(src_files)
    print('Found %s files' % total_files)
    
    return np.random.permutation(src_files)
            
def generate_combined(src_dir, num=100, method='max'):
    shuffled_files = shuffled_files(src_dir)
    
    sample = util.read_data(os.path.join(src_dir, shuffled_files[0]))
    combined = np.zeros(sample.shape) + sample
    for file in shuffled_files[1:num]:
        if method == 'avg':
            combined = combined + util.read_data(os.path.join(src_dir, file))
        else:
            combined = np.maximum(combined, util.read_data(os.path.join(src_dir, file)))
    
    if method == 'avg':
        combined = combined / num
    
    np.save('combined' + str(num), combined)
    
def generate_image_slices(file, dest_dir='zones'):
    print('Clearing zones directory...')
    remove_files('zones')
    
    file_array = np.load(file)
    
    for i in range(file_array.shape[2]):
        img = np.flipud(file_array[:,:,i].transpose())
        scipy.misc.imsave(os.path.join(dest_dir, str(i) + '.png'), img)
    
def slice_zones(zones_file, img_file):
    with open(zones_file, newline='') as csvfile:
        zone_reader = csv.reader(csvfile)
        headers = next(zone_reader) #skip header row
        print(headers)

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

def generate_image_slices_zones(file):
    slice_zones_list = slice_zones('zones.csv', '')
    file_data = util.read_data('data/822f26a77eaca1f06fcda124a494710e.aps')
    for i, slice_dict in enumerate(slice_zones_list):
        img = np.flipud(file_data[:,:,i].transpose())
        img = scipy.misc.toimage(img, channel_axis=2) # This is the key to convert the floats from data to a viewable image. Look at scipy.misc source code, the bytescale method.
        draw = ImageDraw.Draw(img)
        for key in slice_dict:
            rect = slice_dict[key]['rect']
            draw.rectangle([rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]], outline="white")
            #draw.text((20, 70), "something123", font=ImageFont.truetype("font_path123"))
            

        scipy.misc.imsave(os.path.join('scratch', str(i) + 'z.png'), np.asarray(img))
        #img.save(os.path.join('scratch', str(i) + 'z.tiff'))


    
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
        
        