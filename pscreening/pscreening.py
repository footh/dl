import numpy as np

from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, Flatten, Dense, Dropout, TimeDistributed, LSTM
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.optimizers import Adam




class PScreening():
    def __init__(self):
        self.create()

    def create(self):
        """
        """
        # Starting data shape TODO: this will eventually be parameterized
        full_data_shape = (16, 160, 280, 1)
        
        #---------------------
        # Vision model creation for just one frame of the input data. This will be used in the TimeDistributed layer to consume all the frames.
        # This section will convert the 1-channel image into three channels. Got idea from here: http://forums.fast.ai/t/black-and-white-images-on-vgg16/2479/13
        vision_model = Sequential()
        vision_model.add(Conv2D(10, kernel_size=(1,1), padding='same', activation='relu', input_shape=(full_data_shape[1:])))
        vision_model.add(Conv2D(3, kernel_size=(1,1), padding='same', activation='relu'))
        
        # Now getting the vgg16 model with pre-trained weights for some transfer learning
        # Note on adding 'input_shape', if I didn't do this, the input shape would be (None, None, None, 3). This might be OK since it's a convnet but
        # I'd rather be explicit. I'm wondering why Keras doesn't figure out since it's added to an output of this shape?
        vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=full_data_shape[1:3] + (3,))
        # Freezing the weights for the pre-trained VGG16 model (TODO: should I let later layers be trained?)
        for layer in vgg16_model.layers:
            layer.trainable = False
        vision_model.add(vgg16_model)
        vision_model.add(Flatten())
        #---------------------
        
        frame_input = Input(shape=full_data_shape)
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
        
    def vgtest(self):
        vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(160, 280, 3))
        vgg16_model.summary()