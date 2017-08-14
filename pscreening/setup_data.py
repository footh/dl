import os
import numpy as np
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
