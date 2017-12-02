import os
import numpy as np
import threading
import multiprocessing.pool
from functools import partial
import util
import setup_data as sd
import scipy

SLICE_MEANS = [39.091, 
               37.774, 
               36.193, 
               34.306, 
               32.791, 
               35.935, 
               38.585, 
               40.521, 
               41.894, 
               40.369, 
               38.303, 
               35.670, 
               33.085,  
               35.161,  
               36.799,  
               38.065]

ZONE_SLICE_DICT = {
        1: [7,8,9,10,12,14,15,0,1],
        3: [15,0,1,2,4,6,7,8,9],
        5: [15,0,1],
        6: [8,9,10,11,12,13,14,15,0],
        7: [0,1,2,3,4,5,6,7,8],
        8: [8,9,10,11,12,13,14,15,0],
        9: [15,0,1,7,8,9],
        10: [0,1,2,3,4,5,6,7,8],
        11: [6,8,9,10,12,14,0,1,2],
        13: [6,8,9,10,12,14,0,1,2],
        15: [6,8,9,10,12,14,0,1,2],
        12: [14,0,1,2,4,6,8,9,10],
        14: [14,0,1,2,4,6,8,9,10],
        16: [14,0,1,2,4,6,8,9,10],
        17: [7,8,9]
    }

class ZoneApsGenerator():
    def __init__(self, 
                 dynamic_padding=False):

        self.dynamic_paddig=dynamic_padding
        
    def random_padding(self):
        if self.dynamic_padding:
            return 'dynamic_padding'
        else:
            return 'no dynamic padding'
        
    # Data shape here should be as it is saved on disk. The channels argument will assure
    # the channels are added properly
    def flow_from_directory(self, 
                            base_dir,
                            zones,
                            data_shape=None,
                            channels=1,
                            batch_size=32, 
                            shuffle=True,
                            labels=True,
                            subtract_mean=False):

        return ZoneApsFileIterator(base_dir,
                                   zones, 
                                   self, 
                                   data_shape=data_shape,
                                   channels=channels,
                                   batch_size=batch_size, 
                                   shuffle=shuffle,
                                   labels=labels,
                                   subtract_mean=subtract_mean)        
    
# COPYRIGHT
# 
# All contributions by François Chollet:
# Copyright (c) 2015, François Chollet.
# All rights reserved.
# 
# All contributions by Google:
# Copyright (c) 2015, Google, Inc.
# All rights reserved.
# 
# All contributions by Microsoft:
# Copyright (c) 2017, Microsoft, Inc.
# All rights reserved.
# 
# All other contributions:
# Copyright (c) 2015 - 2017, the respective contributors.
# All rights reserved.
# 
# Each contributor holds copyright over their respective contributions.
# The project versioning (Git) records all such contribution source information.
# 
# LICENSE
# 
# The MIT License (MIT)
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
class Iterator(object):
    """Abstract base class for data iterators.

    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """

    def __init__(self, n, batch_size, shuffle):
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(n, batch_size, shuffle)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.batch_index == 0:
                index_array = np.arange(n)
                if shuffle:
                    index_array = np.random.permutation(n)

            current_index = (self.batch_index * batch_size) % n
            if n > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


def _count_valid_files_in_directory(directory, white_list_formats):
    """Count files with extension in `white_list_formats` contained in a directory.

    # Arguments
        directory: absolute path to the directory containing files to be counted
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.

    # Returns
        the count of files with extension in `white_list_formats` contained in
        the directory.
    """
    def _recursive_list(subpath):
        return sorted(os.walk(subpath), key=lambda tpl: tpl[0])

    samples = 0
    for root, _, files in _recursive_list(directory):
        for fname in files:
            is_valid = False
            for extension in white_list_formats:
                if fname.lower().endswith('.' + extension):
                    is_valid = True
                    break
            if is_valid:
                samples += 1
    return samples


def _list_valid_filenames_in_directory(directory, white_list_formats):
    """List paths of files in `subdir` relative from `directory` whose extensions are in `white_list_formats`.

    # Arguments
        directory: absolute path to a directory containing the files to list.
            The directory name is used as class label and must be a key of `class_indices`.
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
    # Returns
        filenames: the path of valid files in `directory`, relative from
            `directory`'s parent (e.g., if `directory` is "dataset/class1",
            the filenames will be ["class1/file1.jpg", "class1/file2.jpg", ...]).
    """
    def _recursive_list(subpath):
        return sorted(os.walk(subpath), key=lambda tpl: tpl[0])

    filenames = []
    for root, _, files in _recursive_list(directory):
        for fname in files:
            is_valid = False
            for extension in white_list_formats:
                if fname.lower().endswith('.' + extension):
                    is_valid = True
                    break
            if is_valid:
                # add filename relative to directory
                filenames.append(os.path.join(directory, fname))
    return filenames

class ZoneApsFileIterator(Iterator):
    """Iterator capable of reading aps files from a directory on disk and extracting the zones to numpy arrays

    # Arguments
        base_dir: base directory to read images from, that forms the full path
        zones: list, zone #s, used to get the proper labels. First element is used to get zone rects
        zone_data_generator: Instance of `ZoneGenerator` to use for random transformations and normalization.
        data_shape: Data shape as it should be extracted as (doesn't include channel) Ex. (5, 80, 180). The image dimensions (ex. 80, 180)
            will be resized to match the data_shape image dimensions
        channels: Channels to reshape to. When channels > 1, data is duplicated along that channel
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        img_scale: Boolean, whether to scale raw data values to [0, 255]
        labels: Boolean, whether to return labels in results (don't need labels for predicting)
    """

    def __init__(self, 
                 base_dir,
                 zones,
                 zone_data_generator,
                 data_shape=None,
                 channels=1,
                 batch_size=32,
                 shuffle=True,
                 img_scale=True,
                 labels=True,
                 subtract_mean=False):

        self.data_shape = data_shape + (channels,)
        
        self.channels = channels
        
        self.directory = base_dir
        self.zones = zones
        self.zone_indices = [z-1 for z in zones]
        self.zone_data_generator = zone_data_generator
        
        self.label_dict = sd.label_dict()
        self.sample_dict = sd.sample_dict(zone=zones[0])
        self.img_scale = img_scale
        self.labels=labels
        self.subtract_mean=subtract_mean

        white_list_formats = {'a3daps', 'aps', 'npy'}

        # first, count the number of samples
        self.samples = 0

        pool = multiprocessing.pool.ThreadPool()
        function_partial = partial(_count_valid_files_in_directory,
                                   white_list_formats=white_list_formats)
        # NOTE: really don't need to parallelize this - keeping this here to mimic Keras impl which
        # expected directory per label
        self.samples = sum(pool.map(function_partial,[self.directory]))

        print(f"Found {self.samples} files...")

        # second, build an index of the images in the different class subfolders
        results = []

        self.filenames = []
        i = 0
        # NOTE: again don't really need this
        for dirpath in [self.directory]:
            results.append(pool.apply_async(_list_valid_filenames_in_directory,
                                            (dirpath, white_list_formats)))
        for res in results:
            filenames = res.get()
            self.filenames += filenames
        pool.close()
        pool.join()
        super().__init__(self.samples, batch_size, shuffle)
        
    def _extract_zones(self, zone_rects, data):
        """
            Extracts zones from data based on zone_rects, resizes rects to data_shape, and reshapes to channel. Result is an array of the 
            form self.data_shape
        """
        slice_data = np.zeros(self.data_shape, dtype=np.float32)
        
        zone_rect_dict = {r[0]: r[1:] for r in zone_rects}
        zone_slices = ZONE_SLICE_DICT[self.zones[0]]
        for i, j in enumerate(zone_slices):
            rb = zone_rect_dict[j][1]
            re = zone_rect_dict[j][3]
            cb = zone_rect_dict[j][0]
            ce = zone_rect_dict[j][2]
            
            extraction = np.asarray(data[j][rb:re,cb:ce])
            extraction = scipy.misc.imresize(extraction, (self.data_shape[1], self.data_shape[2]))
            if self.img_scale and self.subtract_mean:
                extraction = extraction - SLICE_MEANS[j]

            # Zone data is extracted without the channel. Need to reshape here. If one channel, reshape is simple. If more than one
            # data is duplicated 'channels' number of times
            if self.channels == 1:
                extraction = extraction.reshape(self.data_shape[1:])
            else:
                extraction = np.stack((extraction,)*self.channels, axis=-1)
            
            slice_data[i] = extraction

        return slice_data

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.data_shape, dtype=np.float32)
        batch_y = np.zeros((current_batch_size, len(self.zones)), dtype=np.float32)
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            id, ext =  os.path.splitext(os.path.basename(fname))
             
            if ext == '.npy':
                file_data = np.load(fname)
            else:
                file_data = util.read_data(fname)
            
            # TODO: convert to float here?
            if self.img_scale:
                file_data = scipy.misc.bytescale(np.asarray(file_data))
                
            zone_rects = self.sample_dict[id]
            
            batch_x[i] = self._extract_zones(zone_rects, file_data)
            
            if self.labels:
                batch_y[i] = self.label_dict[id][self.zone_indices]

        if self.labels:
            return batch_x, batch_y
        else:
            return batch_x
