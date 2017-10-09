import numpy as np

from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, Flatten, Dense, Dropout, TimeDistributed, LSTM
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import setup_data as sd
import zone_generator
import math
import datetime
import os

_HOME_DIR_ = os.getenv('PSCREENING_HOME', '.')

# NOTES:
# All models will have an input_shape argument that includes the channel. Ex. (5, 80, 180, 1)

class VGG16Model():
    def __init__(self, input_shape):
        self.input_shape = None

    def create(self, input_shape=None):
        """
            Build the model and display the summary
        """
        if input_shape is not None:
            print(f"input_shape: {self.input_shape}")
        else:
            print(f"No input shape given. Model cannot be created")
            return

        
        #---------------------
        # Vision model creation for just one frame of the input data. This will be used in the TimeDistributed layer to consume all the frames.
        # This section will convert the 1-channel image into three channels. Got idea from here: http://forums.fast.ai/t/black-and-white-images-on-vgg16/2479/13
        vision_model = Sequential()
        vision_model.add(Conv2D(10, kernel_size=(1,1), padding='same', activation='relu', input_shape=(self.input_shape[1:])))
        vision_model.add(Conv2D(3, kernel_size=(1,1), padding='same', activation='relu'))
        
        # Now getting the vgg16 model with pre-trained weights for some transfer learning
        # Note on adding 'input_shape', if I didn't do this, the input shape would be (None, None, None, 3). This might be OK since it's a convnet but
        # I'd rather be explicit. I'm wondering why Keras doesn't figure out since it's added to an output of this shape?
        vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape[1:3] + (3,))
        # Freezing the weights for the pre-trained VGG16 model (TODO: should I let later layers be trained?)
        for layer in vgg16_model.layers:
            layer.trainable = False
        vision_model.add(vgg16_model)
        vision_model.add(Flatten())
        #---------------------
        
        frame_input = Input(shape=self.input_shape)
        # Now adding the TimeDistributed wrapper to the entire vision model. Output will be the of 
        # shape (num_frames, flattened output of vision model)
        td_frame_sequence = TimeDistributed(vision_model)(frame_input)
        # Run the frames through an LSTM
        lstm_output = LSTM(256)(td_frame_sequence)
        # Add a dense layer similar to vgg16 (TODO: may not need this?)
        x = Dense(4096, activation='relu')(lstm_output)
        x = Dropout(0.5)(x)
        #x = BatchNormalization()(x)
        predictions = Dense(1, activation = 'sigmoid')(x)
        
        self.model = Model(inputs=frame_input, outputs=predictions)
        
        self.model.summary()
        
    def compile(self, lr=0.001):
        self.model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])
        

def get_batches(base_dir, zone, data_shape, batch_size=20, shuffle=True):
    """
        Get generator for files in base_dir for given zone.
        TODO: For now, channels are assumed to be 1
    """
    zg = zone_generator.ZoneGenerator()
    return zg.flow_from_directory(base_dir,
                                  zone,
                                  data_shape,
                                  batch_size=batch_size,
                                  shuffle=shuffle)
             
def train(model, zone, epochs=1, batch_size=20, learning_rate=0.001, version=None):
    data_shape = sd.zones_max_dict(round_up=True)[zone]
    # Assuming one-channel inputs for now.
    model.create(input_shape=data_shape + (1,))
    model.compile(learning_rate)
    
    train_dir = os.path.join(_HOME_DIR_, 'train')
    train_batches = ps.get_batches(train_dir, zone, data_shape, batch_size=batch_size, shuffle=True)
    steps_per_epoch = math.ceil(train_batches.samples / train_batches.batch_size)
    print(f"training sample size: {train_batches.samples}")
    print(f"training batch size: {train_batches.batch_size}, steps: {steps_per_epoch}")

    valid_dir = os.path.join(_HOME_DIR_, 'valid')
    val_batches = ps.get_batches(valid_dir, zone, data_shape, batch_size=batch_size, shuffle=True)
    validation_steps = math.ceil(val_batches.samples / val_batches.batch_size)
    print(f"validation sample size: {val_batches.samples}")
    print(f"validation batch size: {val_batches.batch_size}, steps: {validation_steps}")
 
    ps.model.fit_generator(train_batches, 
                           steps_per_epoch=steps_per_epoch, 
                           epochs=epochs, 
                           validation_data=val_batches, 
                           validation_steps=validation_steps)
    
    weights_version = 'zone' + str(zone) + '-' + datetime.datetime.now().strftime("%Y%m%d-%M%S")
    if version is not None:
        weights_version = version + '-' + weights_version
        
    
    ps.model.save_weights(os.path.join(_HOME_DIR_, 'weights', weights_version+'.h5'))   

def test(model, zone, batch_size=10, weights_file=None, evaluate=False):
    data_shape = sd.zones_max_dict(round_up=True)[zone]
    # Assuming one-channel inputs for now.
    model.create(input_shape=data_shape + (1,))
    model.compile(learning_rate)

    
    test_dir = os.path.join(_HOME_DIR_, 'test')
    test_batches = ps.get_batches(test_dir, batch_size=batch_size, shuffle=False)
    test_steps = math.ceil(test_batches.samples / test_batches.batch_size)
    print(f"test sample size: {test_batches.samples}")
    print(f"test batch size: {test_batches.batch_size}, steps: {test_steps}")

    weights_file_path = os.path.join(_HOME_DIR_, 'weights', weights_file)
    ps.model.load_weights(weights_file_path)
    
    results = None
    if evaluate:
        results = ps.model.evaluate_generator(test_batches, test_steps)
    else:
        results = ps.model.predict_generator(test_batches, test_steps)

    return results

def vggtest():
    vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(80, 180, 3))
    vgg16_model.summary()