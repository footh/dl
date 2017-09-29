import os
import numpy as np
import threading
import multiprocessing.pool
from functools import partial
from keras import backend as K
import setup_data as sd

class ZoneGenerator():
    def __init__(self, 
                 dynamic_padding=False):

        self.dynamic_paddig=dynamic_padding
        
    def random_padding(self):
        if self.dynamic_padding:
            return 'dynamic_padding'
        else:
            return 'no dynamic padding'
        
    def flow_from_directory(self, 
                            directory,
                            zone,
                            target_size=(16, 160, 280),
                            batch_size=32, 
                            shuffle=True):

        return NumpyFileIterator(directory,
                                 zone, 
                                 self, 
                                 target_size=target_size, 
                                 batch_size=batch_size, 
                                 shuffle=shuffle)        
    
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

class NumpyFileIterator(Iterator):
    """Iterator capable of reading numpy files from a directory on disk.

    # Arguments
        directory: Path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        zone_data_generator: Instance of `ZoneGenerator`
            to use for random transformations and normalization.
        target_size: TBD tuple of integers, dimensions to resize input images to.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
    """

    def __init__(self, 
                 directory,
                 zone,
                 zone_data_generator,
                 target_size=(16, 160, 280),
                 batch_size=32,
                 shuffle=True):

        self.data_format = K.image_data_format()
        if self.data_format == 'channels_last':
            self.data_shape = target_size + (1,)
        else:
            self.data_shape = tuple(target_size[:1] + (1,) + target_size[1:]) 
        
        self.directory = os.path.join(directory, str(zone))
        self.zone = zone
        self.zone_data_generator = zone_data_generator
        
        self.label_dict = sd.label_dict()

        white_list_formats = {'npy'}

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

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.data_shape, dtype=K.floatx())
        batch_y = np.zeros(current_batch_size)
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            id =  os.path.splitext(os.path.basename(fname))[0]
            data = np.load(fname)
            batch_x[i] = data.reshape(self.data_shape)
            batch_y[i] = self.label_dict[id][self.zone-1]

        return batch_x, batch_y
