import csv
import numpy as np

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.optimizers import Adam
import galaxy_image as gimage
from galaxy_image import DirectoryIterator
import os
import datetime
import sys

run = 'train'
param = 'test'
if len(sys.argv) > 1:
    run = sys.argv[1]
if len(sys.argv) > 2:
    param = sys.argv[2]

LABEL_SRC_FILE = 'training_solutions_rev1.csv'

def get_label_dict(src_file, rows=None):
    """
        Get a dict of galaxyId -> numpy array of 37 classification values
    """    
    with open(src_file) as csvfile:
        reader = csv.reader(csvfile)
        next(reader) #skip header row
        rows_out = {}
        for row in reader:
            rows_out[int(row[0])] = np.asarray(row[1:], dtype=np.float32)
            if rows is not None and len(rows_out) > rows:
                break
        
    return rows_out

#vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1))
vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1,1,3))
def vgg_preprocess(x):
    """
        Subtracts the mean RGB value, and transposes RGB to BGR.
        The mean RGB was computed on the image set used to train the VGG model.
        Need to crop the images to the center since the galaxies are centered on
        all the photos I viewed.
 
        Args: 
            x: Image array (height x width x channels)
        Returns:
            Image array (height x width x transposed_channels)
    """

    x = x[100:-100, 100:-100, :] # cropping the image which is 424x424
    x = x - vgg_mean
    
    #return x[:, ::-1] # reverse axis rgb->bgr (bug, this does nothing)
    #return x[::-1, :] # reverse axis rgb->bgr (theano)
    return x[:, :, ::-1] # reverse axis rgb->bgr (tensorflow)


def vectorized_root_mean_squared_error(y_true, y_pred):
    """
        Square root of typical mean squared error but expects each input row to be vector
    """    
    vector_mean_squared_error = K.mean(K.square(y_pred - y_true))
    return K.sqrt(vector_mean_squared_error)

def vectorized_cosine_proximity(y_true, y_pred):
    """
        Square root of typical mean squared error but expects each input row to be vector
    """    
    print('noop')


class Galaxy():
    
    def __init__(self):
        self.create()

    def create(self):
        """
            Uses the trained vgg16 model with loaded weights, but freezes weights at a certain point
            see: https://github.com/fchollet/keras/issues/1728
            note: this method appears to be old, read about the new functional API in the keras documentation.
            this: https://github.com/fchollet/keras/issues/3465
            
            Args:   None
            Returns:   None
        """
        
        #load vgg16 without dense layer and with theano dim ordering
        base_model = VGG16(weights = 'imagenet', include_top = False, input_shape = (3,224,224))
        #base_model = VGG16(weights = 'imagenet', include_top = False, input_shape = (224,224,3))
        
        #number of classes in your dataset e.g. 20
        num_classes = 37
        
        x = Flatten()(base_model.output)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        predictions = Dense(num_classes, activation = 'sigmoid')(x)
        #predictions = Dense(num_classes)(x)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        self.model.summary()
        
    def compile(self, lr=0.001):
        """
            Configures the model for training.
            See Keras documentation: https://keras.io/models/model/
        """
        self.model.compile(optimizer=Adam(lr=lr), loss=vectorized_root_mean_squared_error)
        
    def get_batches(self, path, shuffle=True, batch_size=8):
        """
            Takes the path to a directory, and generates batches of augmented/normalized data. Yields batches indefinitely, in an infinite loop.
        """
        gen = GalaxyImageDataGenerator()
        return gen.flow_from_directory(path, target_size=(224,224), shuffle=shuffle, batch_size=batch_size)

    def fit(self, batches, val_batches, epochs=1, steps_per_epoch=None):
        """
            Fits the model on data yielded batch-by-batch by a Python generator.
        """
        if steps_per_epoch is None:
            steps_per_epoch = batches.samples // batches.batch_size
            
        validation_steps = val_batches.samples // val_batches.batch_size

        self.model.fit_generator(batches, steps_per_epoch=steps_per_epoch, epochs=epochs,
                validation_data=val_batches, validation_steps=validation_steps)
        
        return self.model   

    def evaluate(self, test_batches, steps=1, weights_file=None):
        if weights_file is not None:
            self.model.load_weights(weights_file)
        
        return self.model.evaluate_generator(test_batches, steps)
        

class GalaxyImageDataGenerator():
    
    def __init__(self):
        self.generator = gimage.ImageDataGenerator(preprocessing_function=vgg_preprocess)
    
    def flow_from_directory(self, path, target_size=(224,224), class_mode=None, shuffle=True, batch_size=32):
        return GalaxyDirectoryIterator(path, self.generator, 
                                       target_size=target_size, class_mode=class_mode, 
                                       shuffle=shuffle, batch_size=batch_size)
        
class GalaxyDirectoryIterator(DirectoryIterator):
    
    def __init__(self, path, image_data_generator, target_size=(224,224), class_mode=None, shuffle=True, batch_size=32):
        self.label_dict = get_label_dict(LABEL_SRC_FILE)
        #super(GalaxyDirectoryIterator, self).__init__()  # python 2 compatible
        super().__init__(path, image_data_generator,
                         target_size=target_size, class_mode=class_mode,
                         shuffle=shuffle, batch_size=batch_size)

    def galaxy_ids_from_filenames(self, filenames):
        return [int(os.path.splitext(os.path.basename(name))[0]) for name in filenames]
        
    def next(self):
        #batch_X, filenames_X = super(GalaxyDirectoryIterator, self).next()
        batch_X, filenames_X = super().next()
        
        galaxy_ids = self.galaxy_ids_from_filenames(filenames_X)
        
        # Retrieving labels from filename list
        batch_Y = np.asarray(list(map(self.label_dict.get, galaxy_ids)))
        
        return batch_X, batch_Y
    
def train_galaxy(batch_size=64, steps_per_epoch=None, learning_rate=0.001, version=None):
    galaxy = Galaxy()
    galaxy.compile(learning_rate)
    
    train_batches = galaxy.get_batches('train', shuffle=True, batch_size=batch_size)
    val_batches = galaxy.get_batches('valid', shuffle=True, batch_size=batch_size*2)
    
    trained_model = galaxy.fit(train_batches, val_batches, steps_per_epoch=steps_per_epoch)
    
    weights_version = datetime.datetime.now().strftime("%Y%m%d-%M%S")
    if version is not None:
        weights_version = version + '-' + weights_version
    
    trained_model.save_weights(os.path.join('weights', weights_version+'.h5'))   

def evaluate_galaxy(batch_size=100, weights_file=None):
    galaxy = Galaxy()
    galaxy.compile()
    
    test_batches = galaxy.get_batches('test', shuffle=False, batch_size=batch_size)
    
    steps = test_batches.samples // batch_size
    weights_file_path = os.path.join('weights', weights_file)
    loss = galaxy.evaluate(test_batches, steps=steps, weights_file=weights_file_path)
    print(loss)
    
# if run == 'train':
#     train_galaxy(version=param)
# else:
#     evaluate_galaxy(weights_file=param)
