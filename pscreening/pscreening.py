import numpy as np

import tensorflow as tf

import config
import setup_data as sd
import zone_generator
import zone_aps_generator
import math
import datetime
import os
import tf_util

# NOTES:
# All models will have an input_shape argument that includes the channel. Ex. (5, 80, 180, 1) or (5, 1, 80, 180)

class PScreeningModel():
    def __init__(self, multi_gpu=False):
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        #cfg.log_device_placement = True
        cfg.allow_soft_placement=True

        session = tf.Session(config=cfg)
        tf.keras.backend.set_session(session)
        
        self.input_shape = None
        self.multi_gpu = multi_gpu

    def get_image_model(self, input_shape):
        """
            Children will implement this and return full image model
        """
        raise NotImplementedError
    
    def build_model(self, img_shape):
        #---------------------
        # Vision model creation for just one frame of the input data. This will be used in the TimeDistributed layer to consume all the frames.
        # This section will convert the 1-channel image into three channels. Got idea from here: http://forums.fast.ai/t/black-and-white-images-on-vgg16/2479/13
        vision_model = tf.keras.models.Sequential()
        vision_model.add(tf.keras.layers.Conv2D(10, kernel_size=(1,1), padding='same', activation='relu', input_shape=(self.input_shape[1:])))
        vision_model.add(tf.keras.layers.Conv2D(3, kernel_size=(1,1), padding='same', activation='relu'))

        # Now getting the image model with pre-trained weights for some transfer learning. Implemented by child classes.
        img_model = self.get_image_model(img_shape)

        vision_model.add(img_model)
        vision_model.add(tf.keras.layers.Flatten())
        #---------------------
        
        frame_input = tf.keras.layers.Input(shape=self.input_shape)
        # Now adding the TimeDistributed wrapper to the entire vision model. Output will be the of 
        # shape (num_frames, flattened output of vision model)
        td_frame_sequence = tf.keras.layers.TimeDistributed(vision_model)(frame_input)
        # Run the frames through an LSTM
        lstm_output = tf.keras.layers.LSTM(256)(td_frame_sequence)
        # Add a dense layer similar to vgg16 (TODO: may not need this?)
        x = tf.keras.layers.Dense(4096, activation='relu')(lstm_output)
        x = tf.keras.layers.Dropout(0.5)(x)
        #x = BatchNormalization()(x)
        predictions = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)
        
        return tf.keras.models.Model(inputs=frame_input, outputs=predictions)
    
    def create(self, input_shape=None):
        """
            Build the model and display the summary
        """
        img_shape = None
        if input_shape is not None:
            self.input_shape = input_shape
            print(f"input_shape: {self.input_shape}")
            img_shape = input_shape[1:3] + (3,)
            print(f"img_shape: {img_shape}")
        else:
            print(f"No input shape given. Model cannot be created")
            return

        if self.multi_gpu:
            with tf.device('/device:CPU:0'):
                self.model = self.build_model(img_shape)
        else:
            self.model = self.build_model(img_shape)
        
        self.model.summary()        

class InceptionModel(PScreeningModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'inception'

    def get_image_model(self, img_shape):
        # Note on adding 'input_shape', if I didn't do this, the input shape would be (None, None, None, 3). This might be OK since it's a convnet but
        # I'd rather be explicit. I'm wondering why Keras doesn't figure out since it's added to an output of this shape?
        inception_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=img_shape)
        # Freezing the weights for the pre-trained VGG16 model (TODO: should I let later layers be trained?)
        for layer in inception_model.layers:
            layer.trainable = False

        return inception_model

class VGG16Model(PScreeningModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'vgg16'
        
    def get_image_model(self, img_shape):
        # Note on adding 'input_shape', if I didn't do this, the input shape would be (None, None, None, 3). This might be OK since it's a convnet but
        # I'd rather be explicit. I'm wondering why Keras doesn't figure out since it's added to an output of this shape?
        #TODO: look into the pooling argument here!!!
        vgg16_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=img_shape)
        # Freezing the weights for the pre-trained VGG16 model (TODO: should I let later layers be trained?)
        for layer in vgg16_model.layers:
            layer.trainable = False

        return vgg16_model
       
def get_batches(src, zone, data_shape, batch_size=24, shuffle=True):
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
             
def get_batches_aps(src, zone, data_shape, batch_size=24, shuffle=True):
    """
        Get generator for files in src (train, valid, test, etc.) for given zone.
        TODO: For now, channels are assumed to be 1
    """
    base_dir = os.path.join(config.PSCREENING_LOCAL_HOME, config.RAW_DATA_DIR, src)
    
    zg = zone_aps_generator.ZoneApsGenerator()
    return zg.flow_from_directory(base_dir,
                                  zone,
                                  data_shape,
                                  batch_size=batch_size,
                                  shuffle=shuffle)
             
def train(zone, epochs=1, batch_size=24, learning_rate=0.001, 
          version=None, gpus=None, mtype='vgg16', starting_weights_file=None):
    data_shape = sd.zones_max_dict(round_up=True)[zone]
    # TODO: parameterize!
    data_shape = (data_shape[0], 200, 200)

    train_batches = get_batches_aps('train', zone, data_shape, batch_size=batch_size, shuffle=True)
    steps_per_epoch = math.ceil(train_batches.samples / train_batches.batch_size)
    print(f"training sample size: {train_batches.samples}")
    print(f"training batch size: {train_batches.batch_size}, steps: {steps_per_epoch}")

    val_batches = get_batches_aps('valid', zone, data_shape, batch_size=batch_size, shuffle=True)    
    validation_steps = math.ceil(val_batches.samples / val_batches.batch_size)
    print(f"validation sample size: {val_batches.samples}")
    print(f"validation batch size: {val_batches.batch_size}, steps: {validation_steps}")
    
    ps_model = None
    if mtype == 'inception':
        ps_model = InceptionModel(multi_gpu=(gpus is not None))
    else:
        ps_model = VGG16Model(multi_gpu=(gpus is not None))

    #TODO: create the model with None as the time dimension? When looking at the code it looked like TimeDistributed
    #acts differently when None is passed as opposed to a fixed dimension. 
    ps_model.create(input_shape=train_batches.data_shape)
    
    if starting_weights_file is not None:
        swf_path = os.path.join(config.PSCREENING_HOME, config.WEIGHTS_DIR, starting_weights_file)
        ps_model.model.load_weights(swf_path)
    
    train_model = ps_model.model
    if gpus is not None:
        train_model = tf_util.multi_gpu_model(ps_model.model, gpus)
     
    train_model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy']) 
    train_model.fit_generator(train_batches,
                              steps_per_epoch=steps_per_epoch,
                              epochs=epochs,
                              validation_data=val_batches, 
                              validation_steps=validation_steps,
                              class_weight={0:0.1, 1:0.90})
     
    weights_version = f"zone{zone}-{ps_model.name}-e{epochs}-bs{batch_size}-lr{str(learning_rate).split('.')[1]}"
    weights_version += f"-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}" 
    if version is not None:
        weights_version += f"-{version}"
         
    weights_file = weights_version + '.h5'
    
    if gpus is not None:
        train_model = [l for l in train_model.layers if l.__class__.__name__ == 'Model'][0]
    
    train_model.save_weights(os.path.join(config.PSCREENING_HOME, config.WEIGHTS_DIR, weights_file))
     
    return weights_file

def test(zone, batch_size=10, weights_file=None, evaluate=True, gpus=None):
    data_shape = sd.zones_max_dict(round_up=True)[zone]

    test_batches = get_batches('test', zone, data_shape, batch_size=batch_size, shuffle=False)
    test_steps = math.ceil(test_batches.samples / test_batches.batch_size)
    print(f"test sample size: {test_batches.samples}")
    print(f"test batch size: {test_batches.batch_size}, steps: {test_steps}")

    mtype = weights_file.split('-')[1]
    ps_model = None
    if mtype == 'inception':
        ps_model = InceptionModel()
    else:
        ps_model = VGG16Model()

    # Assuming one-channel inputs for now.
    ps_model.create(input_shape=test_batches.data_shape)
    ps_model.model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    weights_file_path = os.path.join(config.PSCREENING_HOME, config.WEIGHTS_DIR, weights_file)
    ps_model.model.load_weights(weights_file_path)
    
    results = None
    if evaluate:
        results = ps_model.model.evaluate_generator(test_batches, test_steps)
    else:
        results = ps_model.model.predict_generator(test_batches, test_steps)

    return results

    