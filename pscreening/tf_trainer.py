import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Modes

def _vgg16_model_fn(features, labels, mode, params):
    #TODO: Does features include batch size?
    
    #TODO: features['x'] will return a tensor. get_shape() on a tensor is not a tuple?
    img_shape = features['x'].get_shape()[1:3] + (3,)
       
    #---------------------
    # Vision model creation for just one frame of the input data. This will be used in the TimeDistributed layer to consume all the frames.
    # This section will convert the 1-channel image into three channels. Got idea from here: http://forums.fast.ai/t/black-and-white-images-on-vgg16/2479/13
    vision_model = tf.contrib.keras.models.Sequential()
    vision_model.add(tf.contrib.keras.layers.Conv2D(10, kernel_size=(1,1), padding='same', activation='relu', input_shape=(self.input_shape[1:])))
    vision_model.add(tf.contrib.keras.layers.Conv2D(3, kernel_size=(1,1), padding='same', activation='relu'))
    
    # Now getting the image model with pre-trained weights for some transfer learning. Implemented by child classes.
    #TODO: look into the pooling argument here instead of flattening?!!!
    img_model = tf.contrib.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=img_shape)
    # Freezing the weights for the pre-trained VGG16 model (TODO: should I let later layers be trained?)
    for layer in img_model.layers:
        layer.trainable = False
    
    vision_model.add(img_model)
    vision_model.add(tf.contrib.keras.layers.Flatten())
    #---------------------
    
    frame_input = tf.contrib.keras.layers.Input(shape=features['x'].get_shape())
    # Now adding the TimeDistributed wrapper to the entire vision model. Output will be the of 
    # shape (num_frames, flattened output of vision model)
    td_frame_sequence = tf.contrib.keras.layers.TimeDistributed(vision_model)(frame_input)
    # Run the frames through an LSTM
    lstm_output = tf.contrib.keras.layers.LSTM(256)(td_frame_sequence)
    # Add a dense layer similar to vgg16 (TODO: may not need this?)
    x = tf.contrib.keras.layers.Dense(4096, activation='relu')(lstm_output)
    x = tf.contrib.keras.layers.Dropout(0.5)(x)
    #x = BatchNormalization()(x)
    predictions = tf.contrib.keras.layers.Dense(1, activation = 'sigmoid')(x)
    
    if mode in (Modes.TRAIN, Modes.EVAL):
        loss = tf.losses.sigmoid_cross_entropy(targets, predictions)
        tf.summary.scalar('OptimizeLoss', loss)
        
    if mode == Modes.TRAIN:
        global_step = tf.contrib.framework.get_or_create_global_step()
        #TODO: Classes are weighted by the gradient_multipliers argument. See: https://www.tensorflow.org/api_docs/python/tf/contrib/layers/optimize_loss
        # See: https://medium.com/towards-data-science/https-medium-com-manishchablani-useful-keras-features-4bac0724734c
        train_op = tf.contrib.layers.optimize_loss(loss=loss, 
                                                   global_step=global_step, 
                                                   learning_rate=params["learning_rate"], 
                                                   optimizer="Adam") #  'SGD', 'Adam', 'Adagrad'
        #TODO: See: https://www.tensorflow.org/api_docs/python/tf/estimator/EstimatorSpec
        return tf.estimator.EstimatorSpec(mode, 
                                          loss=loss, 
                                          train_op=train_op)
    
    if mode == Modes.EVAL:
        eval_metric_ops = {
            'acc': tf.metrics.accuracy(tf.cast(targets, tf.float32), predictions)
        }
        return tf.estimator.EstimatorSpec(mode, 
                                          loss=loss, 
                                          eval_metric_ops=eval_metrics_ops)
    
    if mode == Modes.PREDICT:
        predictions_dict = {
            'threat': predictions
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions_dict)
        }
        return tf.estimator.EstimatorSpec(mode,
                                          predictions=predictions_dict,
                                          export_outputs=export_outputs)

def build_estimator(model_save_dir):
    # TODO: do I need to save the checkpoints?
    return tf.estimator.Estimator(model_fn=_vgg16_model_fn,
                                  model_dir=model_save_dir,
                                  config=tf.contrib.learn.RunConfig(save_checkpoints_secs=180))

#train_input_fn = tf.estimator.inputs.numpy_input_fn()

# from tensorflow.contrib.learn.python.learn.estimators import estimator
# LEARNING_RATE = 0.001
# # Set model params
# model_params = {"learning_rate": LEARNING_RATE}
# 
# # Instantiate Estimator
# est = estimator.Estimator(model_fn=model_fn, params=model_params)
# 
# #est.fit(x=training_set.data, y=training_set.target, steps=5000)