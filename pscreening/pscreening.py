import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, Flatten, Dense, Dropout, TimeDistributed, LSTM
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

import config
import setup_data as sd
import zone_generator
import math
import datetime
import os

# NOTES:
# All models will have an input_shape argument that includes the channel. Ex. (5, 80, 180, 1)

class PScreeningModel():
    def __init__(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))


class VGG16Model(PScreeningModel):
    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.name = 'vgg16'

    def create(self, input_shape=None):
        """
            Build the model and display the summary
        """
        if input_shape is not None:
            self.input_shape = input_shape
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
        

def get_batches(src, zone, data_shape, batch_size=20, shuffle=True):
    """
        Get generator for files in src (train, valid, test, etc.) for given zone.
        TODO: For now, channels are assumed to be 1
    """
    base_dir = os.path.join(config.PSCREENING_HOME, config.TRAINING_DIR, src)
    
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
    model.compile(lr=learning_rate)
    
    train_batches = get_batches('train', zone, data_shape, batch_size=batch_size, shuffle=True)
    steps_per_epoch = math.ceil(train_batches.samples / train_batches.batch_size)
    print(f"training sample size: {train_batches.samples}")
    print(f"training batch size: {train_batches.batch_size}, steps: {steps_per_epoch}")

    val_batches = get_batches('valid', zone, data_shape, batch_size=batch_size, shuffle=True)
    validation_steps = math.ceil(val_batches.samples / val_batches.batch_size)
    print(f"validation sample size: {val_batches.samples}")
    print(f"validation batch size: {val_batches.batch_size}, steps: {validation_steps}")
 
    model.model.fit_generator(train_batches,
                              steps_per_epoch=steps_per_epoch,
                              epochs=epochs,
                              validation_data=val_batches, 
                              validation_steps=validation_steps)
    
    weights_version = f"zone{zone}-{model.name}-e{epochs}-bs{batch_size}-lr{str(learning_rate).split('.')[1]}"
    weights_version += f"-{datetime.datetime.now().strftime('%Y%m%d-%M%S')}" 
    if version is not None:
        weights_version += f"-{version}"
        
    weights_file = weights_version + '.h5'
    model.model.save_weights(os.path.join(config.PSCREENING_HOME, config.WEIGHTS_DIR, weights_file))
    
    return weights_file

def test(model, zone, batch_size=10, weights_file=None, evaluate=False):
    data_shape = sd.zones_max_dict(round_up=True)[zone]
    # Assuming one-channel inputs for now.
    model.create(input_shape=data_shape + (1,))
    model.compile()

    test_batches = get_batches('test', batch_size=batch_size, shuffle=False)
    test_steps = math.ceil(test_batches.samples / test_batches.batch_size)
    print(f"test sample size: {test_batches.samples}")
    print(f"test batch size: {test_batches.batch_size}, steps: {test_steps}")

    weights_file_path = os.path.join(config.PSCREENING_HOME, config.WEIGHTS_DIR, weights_file)
    model.model.load_weights(weights_file_path)
    
    results = None
    if evaluate:
        results = model.model.evaluate_generator(test_batches, test_steps)
    else:
        results = model.model.predict_generator(test_batches, test_steps)

    return results

def vggtest():
    vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(80, 180, 3))
    vgg16_model.summary()