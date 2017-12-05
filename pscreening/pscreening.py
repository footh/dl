import numpy as np

import tensorflow as tf

import config
import setup_data as sd
import zone_generator
import zone_aps_generator
import callbacks
import math
import datetime
import os
import tf_util
from collections import defaultdict


class PScreeningModel():
    def __init__(self, output=1, multi_gpu=False, train_layer_start=None):
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        #cfg.log_device_placement = True
        cfg.allow_soft_placement=True

        session = tf.Session(config=cfg)
        tf.keras.backend.set_session(session)
        
        self.input_shape = None
        self.multi_gpu = multi_gpu
        self.output=output
        self.train_layer_start = train_layer_start

    def get_image_model(self, input_shape):
        """
            Children will implement this and return full image model
        """
        raise NotImplementedError
    
    def build_model(self):
        learn_channels = True
        if self.input_shape[3] == 3: learn_channels = False

        img_shape = self.input_shape[1:3] + (3,)
        print(f"img_shape: {img_shape}")
        
        #---------------------
        # Vision model creation for just one frame of the input data. This will be used in the TimeDistributed layer to consume all the frames.
        # This section will convert the 1-channel image into three channels. Got idea from here: http://forums.fast.ai/t/black-and-white-images-on-vgg16/2479/13
        vision_model = tf.keras.models.Sequential()
        if learn_channels:
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
        x = tf.keras.layers.Dense(2048)(lstm_output)
        #x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        predictions = tf.keras.layers.Dense(self.output, activation = 'sigmoid')(x)
        
        return tf.keras.models.Model(inputs=frame_input, outputs=predictions)
    
    def create(self, input_shape=None):
        """
            Build the model and display the summary
            Models will have an input_shape argument that includes the channel. Ex. (5, 200, 200, 1)
        """
        img_shape = None
        if input_shape is not None:
            self.input_shape = input_shape
            print(f"input_shape: {self.input_shape}")            
        else:
            print(f"No input shape given. Model cannot be created")
            return

        if self.multi_gpu:
            with tf.device('/device:CPU:0'):
                self.model = self.build_model()
        else:
            self.model = self.build_model()
        
        self.model.summary()        

class InceptionModel(PScreeningModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'inception'

    def get_image_model(self, img_shape):
        # Note on adding 'input_shape', if I didn't do this, the input shape would be (None, None, None, 3). This might be OK since it's a convnet but
        # I'd rather be explicit. I'm wondering why Keras doesn't figure out since it's added to an output of this shape?
        model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=img_shape)
        # Freezing the weights for the pre-trained model (TODO: should I let later layers be trained?)
        for layer in model.layers:
            #print(f"{self.name} layer: {layer}")
            layer.trainable = False
            
        if self.train_layer_start is not None:
            print(f"TRAINING ALLOWED FROM LAYER {train_layer_start}!!!!!!!!!!!!!!!!!")
            for layer in model.layers[train_layer_start:]:
                layer.trainable = True
        else:
            print(f"NO TRAINING ALLOWED!!!!!!!!!!!!!!")

        return model

class VGG16Model(PScreeningModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'vgg16'
        
    def get_image_model(self, img_shape):
        # Note on adding 'input_shape', if I didn't do this, the input shape would be (None, None, None, 3). This might be OK since it's a convnet but
        # I'd rather be explicit. I'm wondering why Keras doesn't figure out since it's added to an output of this shape?
        #TODO: look into the pooling argument here!!!
        model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=img_shape)
        # Freezing the weights for the pre-trained model (TODO: should I let later layers be trained?)
        for layer in model.layers:
            #print(f"{self.name} layer: {layer}")
            layer.trainable = False
        
        if self.train_layer_start is not None:
            print(f"TRAINING ALLOWED FROM LAYER {train_layer_start}!!!!!!!!!!!!!!!!!")
            for layer in model.layers[train_layer_start:]:
                layer.trainable = True
        else:
            print(f"NO TRAINING ALLOWED!!!!!!!!!!!!!!")
            
        return model
    
class ResNet50Model(PScreeningModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'resnet50'
        
    def get_image_model(self, img_shape):
        # Note on adding 'input_shape', if I didn't do this, the input shape would be (None, None, None, 3). This might be OK since it's a convnet but
        # I'd rather be explicit. I'm wondering why Keras doesn't figure out since it's added to an output of this shape?
        #TODO: look into the pooling argument here!!!
        model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=img_shape)
        # Freezing the weights for the pre-trained model (TODO: should I let later layers be trained?)
        for layer in model.layers:
            #print(f"{self.name} layer: {layer}")
            layer.trainable = False

        return model
    
class InceptionResNetModel(PScreeningModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'inceptionresnet'
        
    def get_image_model(self, img_shape):
        # Note on adding 'input_shape', if I didn't do this, the input shape would be (None, None, None, 3). This might be OK since it's a convnet but
        # I'd rather be explicit. I'm wondering why Keras doesn't figure out since it's added to an output of this shape?
        #TODO: look into the pooling argument here!!!
        model = tf.keras.applications.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=img_shape)
        # Freezing the weights for the pre-trained model (TODO: should I let later layers be trained?)
        for layer in model.layers:
            #print(f"{self.name} layer: {layer}")
            layer.trainable = False

        return model
    
class XceptionModel(PScreeningModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'xception'
        
    def get_image_model(self, img_shape):
        # Note on adding 'input_shape', if I didn't do this, the input shape would be (None, None, None, 3). This might be OK since it's a convnet but
        # I'd rather be explicit. I'm wondering why Keras doesn't figure out since it's added to an output of this shape?
        #TODO: look into the pooling argument here!!!
        model = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=img_shape)
        # Freezing the weights for the pre-trained model (TODO: should I let later layers be trained?)
        for layer in model.layers:
            #print(f"{self.name} layer: {layer}")
            layer.trainable = False

        return model
    
class MobileNetModel(PScreeningModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'mobilenet'
        
    def get_image_model(self, img_shape):
        # Note on adding 'input_shape', if I didn't do this, the input shape would be (None, None, None, 3). This might be OK since it's a convnet but
        # I'd rather be explicit. I'm wondering why Keras doesn't figure out since it's added to an output of this shape?
        #TODO: look into the pooling argument here!!!
        model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=img_shape)
        # Freezing the weights for the pre-trained model (TODO: should I let later layers be trained?)
        for layer in model.layers:
            #print(f"{self.name} layer: {layer}")
            layer.trainable = False

        return model
   
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
             
def get_batches_aps(src, zones, data_shape, channels=1, batch_size=24, shuffle=True, labels=True, img_scale=True, subtract_mean=False):
    """
        Get generator for files in src (train, valid, test, etc.) for given zone.
        TODO: For now, channels are assumed to be 1
    """
    base_dir = os.path.join(config.PSCREENING_LOCAL_HOME, config.RAW_DATA_DIR, src)
    
    zg = zone_aps_generator.ZoneApsGenerator()
    return zg.flow_from_directory(base_dir,
                                  zones,
                                  data_shape=data_shape,
                                  channels=channels,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  labels=labels,
                                  img_scale=img_scale,
                                  subtract_mean=subtract_mean)
           
def _get_model(mtype, output=1, multi_gpu=False, train_layer_start=None):
    if mtype == 'inception':
        ps_model = InceptionModel(output=output, multi_gpu=multi_gpu, train_layer_start=train_layer_start)
    elif mtype == 'resnet50':
        ps_model = ResNet50Model(output=output, multi_gpu=multi_gpu, train_layer_start=train_layer_start)
    elif mtype == 'inceptionresnet':
        ps_model = InceptionResNetModel(output=output, multi_gpu=multi_gpu, train_layer_start=train_layer_start)
    elif mtype == 'xception':
        ps_model = XceptionModel(output=output, multi_gpu=multi_gpu, train_layer_start=train_layer_start)
    elif mtype == 'mobilenet':
        ps_model = MobileNetModel(output=output, multi_gpu=multi_gpu, train_layer_start=train_layer_start)
    else:
        ps_model = VGG16Model(output=output, multi_gpu=multi_gpu, train_layer_start=train_layer_start)
        
    return ps_model
        
def key_zone_to_zones(key_zone):
    zones = [key_zone]
    if key_zone == 1: zones += [2]
    if key_zone == 3: zones += [4]
    #if key_zone == 11: zones += [13,15]
    #if key_zone == 12: zones += [14,16]
    return zones

def _model_params(model_file):
    params = sd.get_file_name(model_file).split('-')
    
    key_zone = int(params[0].replace('zone', ''))
    zones = key_zone_to_zones(key_zone)
    mtype = params[1]
    img_dim = int(params[2].replace('d', ''))
    channels = int(params[3].replace('c', ''))
    
    return key_zone, zones, mtype, img_dim, channels
             
def train(zones, epochs=1, batch_size=32, learning_rate=0.001,
          version=None, gpus=None, mtype='vgg16', starting_model_file=None,
          img_dim=224, channels=1, train_layer_start=None):
    if not isinstance(zones, list): zones = [zones]
    
    #data_shape = sd.zones_max_dict(round_up=True)[zones[0]]
    data_shape = (len(zone_aps_generator.ZONE_SLICE_DICT[zones[0]]),) + (img_dim, img_dim)

    img_scale = True if mtype=='vgg16' else False
    train_batches = get_batches_aps('train', zones, data_shape, channels=channels, batch_size=batch_size, shuffle=True, img_scale=img_scale)
    steps_per_epoch = math.ceil(train_batches.samples / train_batches.batch_size)
    print(f"training sample size: {train_batches.samples}")
    print(f"training batch size: {train_batches.batch_size}, steps: {steps_per_epoch}")

    val_batches = get_batches_aps('valid', zones, data_shape, channels=channels, batch_size=batch_size, shuffle=True)    
    validation_steps = math.ceil(val_batches.samples / val_batches.batch_size)
    print(f"validation sample size: {val_batches.samples}")
    print(f"validation batch size: {val_batches.batch_size}, steps: {validation_steps}")
    
    #----------------------------------
    train_model = None
    if starting_model_file is not None:
        # https://github.com/fchollet/keras/issues/6865 (why I must compile saved model
        smf_path = os.path.join(config.PSCREENING_HOME, config.MODEL_DIR, starting_model_file)
        train_model = tf.keras.models.load_model(smf_path)
        _, _, mtype, _, _ = _model_params(starting_model_file)
    else:
        ps_model = _get_model(mtype, output=len(zones), multi_gpu=(gpus is not None), train_layer_start=train_layer_start)
        #TODO: create the model with None as the time dimension? When looking at the code it looked like TimeDistributed
        #acts differently when None is passed as opposed to a fixed dimension. 
        ps_model.create(input_shape=train_batches.data_shape)
        train_model = ps_model.model
    
    if gpus is not None:
        train_model = tf_util.multi_gpu_model(train_model, gpus)
    
    train_model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy']) 

    model_version = f"zone{zones[0]}-{mtype}-d{img_dim}-c{channels}-e{epochs}-bs{batch_size}-lr{str(learning_rate).split('.')[1]}"
    model_version += f"-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}" 
    if version is not None:
        model_version += f"-{version}"
    print(f"model_version: {model_version}")
    model_version_el = model_version + "-{epoch:02d}-{loss:.3f}"
    
    model_file = model_version_el + '.h5'
    model_file = os.path.join(config.PSCREENING_HOME, config.MODEL_DIR, model_file)
    
    cb_model_save = callbacks.ModelCheckpoint(model_file, multi_gpu=(gpus is not None))
    
    weight1 = round(0.9 ** len(zones), 2)
    train_model.fit_generator(train_batches,
                              steps_per_epoch=steps_per_epoch,
                              epochs=epochs,
                              validation_data=val_batches, 
                              validation_steps=validation_steps,
                              callbacks=[cb_model_save],
                              class_weight={0:1-weight1, 1:weight1})
         
    return model_version

def test(weights_file, src='test', batch_size=7, evaluate=True):
    if weights_file is None:
        print(f"Need weights file to test.")
        return
    
    key_zone, zones, mtype, img_dim, channels = _model_params(weights_file)
    
    data_shape = sd.zones_max_dict(round_up=True)[zones[0]]
    data_shape = (data_shape[0],) + (img_dim, img_dim)

    test_batches = get_batches_aps(src, zones, data_shape, channels=channels, batch_size=batch_size, shuffle=False, labels=evaluate)
    test_steps = math.ceil(test_batches.samples / test_batches.batch_size)
    print(f"test sample size: {test_batches.samples}")
    print(f"test batch size: {test_batches.batch_size}, steps: {test_steps}")

    ps_model = _get_model(mtype, output=len(zones))

    # Assuming one-channel inputs for now.
    ps_model.create(input_shape=test_batches.data_shape)
    ps_model.model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    weights_dir = config.WEIGHTS_DIR
    if src == 'submission': weights_dir = 'submission-' + weights_dir
    weights_file_path = os.path.join(config.PSCREENING_HOME, weights_dir, weights_file)
    ps_model.model.load_weights(weights_file_path)
    
    results = None
    if evaluate:
        results = ps_model.model.evaluate_generator(test_batches, test_steps)
    else:
        results = ps_model.model.predict_generator(test_batches, test_steps)
        # The 'filenames' argument is expected to be the order of the results since shuffle is set to False
        ids = [sd.get_file_name(fname) for fname in test_batches.filenames]
        results = dict(zip(ids, results))

    return results

def testm(model_file, src='test', batch_size=6, evaluate=True):
    if model_file is None:
        print(f"Need model file to test.")
        return
    
    _, zones, mtype, img_dim, channels = _model_params(model_file)
    
    #data_shape = sd.zones_max_dict(round_up=True)[zones[0]]
    data_shape = (len(zone_aps_generator.ZONE_SLICE_DICT[zones[0]]),) + (img_dim, img_dim)

    img_scale = True if mtype=='vgg16' else False
    test_batches = get_batches_aps(src, zones, data_shape, channels=channels, batch_size=batch_size, shuffle=False, labels=evaluate, img_scale=img_scale)
    test_steps = math.ceil(test_batches.samples / test_batches.batch_size)
    print(f"test sample size: {test_batches.samples}")
    print(f"test batch size: {test_batches.batch_size}, steps: {test_steps}")

    model_file_path = os.path.join(config.PSCREENING_HOME, config.MODEL_DIR, model_file)
    ps_model = tf.keras.models.load_model(model_file_path)
    ps_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    results = None
    if evaluate:
        results = ps_model.evaluate_generator(test_batches, test_steps)
    else:
        results = ps_model.predict_generator(test_batches, test_steps)
        # The 'filenames' argument is expected to be the order of the results since shuffle is set to False
        ids = [sd.get_file_name(fname) for fname in test_batches.filenames]
        results = dict(zip(ids, results))

    return results

def _ensemble(results_dict_list):
    """
        Just taking the average...
    """
    ttl = len(results_dict_list)
    
    results_dict_sample = results_dict_list[0]
    keys = np.asarray(list(results_dict_sample.keys()))
    values = np.zeros(np.asarray(list(results_dict_sample.values())).shape, dtype=np.float32)
    
    for results_dict in results_dict_list:
        values += np.asarray(list(results_dict.values()))
        
    values = values / ttl
    
    return dict(zip(keys, values))

# TODO: *************** If subtract_mean works well, add it here
def create_submission_file():
    import csv
    
    # zone_model_dict is a dict of key_zone: [list of model files]
    zone_model_dict = defaultdict(list)
    for model_file in config.SUBMISSION_MODELS:
        key_zone, _, _, _, _ = _model_params(model_file)
        zone_model_dict[key_zone].append(model_file)

    submission_results = []
    for key_zone, model_files in sorted(zone_model_dict.items()):
        
        results_dict_list = []
        for model_file in model_files:
            _, zones, _, _, _ = _model_params(model_file)            

            print(f"Getting results for key_zone {key_zone} using model_file: {model_file}...")
            # Clear session after each run? tf.keras.backend.clear_session()
            results_dict = testm(model_file, src='submission', batch_size=4, evaluate=False)
            print(f"Finished getting results...")
            results_dict_list.append(results_dict)
            
        results_dict = _ensemble(results_dict_list)
        
        for id, results in results_dict.items():
            for i, zone in enumerate(zones):
                submission_results.append([id, zone, results[i]])
                
    print(f"Writing to file...")
    submission_results.sort()
    
    submission_file_name = f"submission-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    submission_file_name = os.path.join(config.PSCREENING_HOME, config.SUBMISSION_DIR, submission_file_name)
    with open(submission_file_name, 'w') as submission_file:
        wr = csv.writer(submission_file, delimiter=',')
        wr.writerow(['Id', 'Probability'])

        for submission_result in submission_results:
            id_zone = submission_result[0] + '_Zone' + str(submission_result[1])
            wr.writerow([id_zone, submission_result[2]])

def convert_weights_to_model(file, sn, slices, output):
    weights_file = os.path.join(config.PSCREENING_HOME, 'submission-weights', file)
    
    ps_model = VGG16Model(output=output, multi_gpu=False)
    ps_model.create(input_shape=(slices,200,200,1))
    
    ps_model.model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    ps_model.model.load_weights(weights_file)
    
    ps_model.model.save(os.path.join(config.PSCREENING_HOME, config.MODEL_DIR, f"{sn}-SUBWGHTS.h5"))
    