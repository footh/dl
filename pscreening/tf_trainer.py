import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Modes

#https://www.tensorflow.org/versions/master/api_docs/python/tf/keras/estimator/model_to_estimator

def _vgg16_model_fn(features, labels, mode, params):
    if mode == Modes.TRAIN:
        tf.contrib.keras.backend.set_learning_phase(1)

    print(f"features['x']: {features['x']}")
    print(f"labels: {labels}")
    
    # Feature shape will contain full shape including unknown batch size, ex. (None, 5, 80, 180, 1)
    feature_shape = tuple(features['x'].shape.as_list())
    input_shape = feature_shape[1:]  #  Ex. (5, 80, 180, 1)
    img_shape = feature_shape[2:4] + (3,)  # Ex. (80, 180, 3)
       
    #---------------------
    # Vision model creation for just one frame of the input data. This will be used in the TimeDistributed layer to consume all the frames.
    # This section will convert the 1-channel image into three channels. Got idea from here: http://forums.fast.ai/t/black-and-white-images-on-vgg16/2479/13
    vision_model = tf.contrib.keras.models.Sequential()
    vision_model.add(tf.contrib.keras.layers.Conv2D(10, kernel_size=(1,1), padding='same', activation='relu', input_shape=(input_shape[1:])))
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
    
    #frame_input = tf.contrib.keras.layers.Input(shape=input_shape)
    frame_input = tf.contrib.keras.layers.Input(tensor=features['x'])

    # Now adding the TimeDistributed wrapper to the entire vision model. Output will be the of 
    # shape (num_frames, flattened output of vision model)
    td_frame_sequence = tf.contrib.keras.layers.TimeDistributed(vision_model)(frame_input)
    # Run the frames through an LSTM
    lstm_output = tf.contrib.keras.layers.LSTM(256)(td_frame_sequence)
    # Add a dense layer similar to vgg16 (TODO: may not need this?)
    x = tf.contrib.keras.layers.Dense(4096, activation='relu')(lstm_output)
    print(f"Output of Dense Layer after LSTM: {x}")
    x = tf.contrib.keras.layers.Dropout(0.5)(x)
    #x = BatchNormalization()(x)
    predictions = tf.contrib.keras.layers.Dense(1, activation = 'sigmoid')(x)
    
    if mode in (Modes.TRAIN, Modes.EVAL):
        loss = tf.losses.sigmoid_cross_entropy(labels, predictions)
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

def build_estimator(model_save_dir, model_params):
    # TODO: do I need to save the checkpoints?
    return tf.estimator.Estimator(model_fn=_vgg16_model_fn,
                                  model_dir=model_save_dir,
                                  config=tf.contrib.learn.RunConfig(save_checkpoints_secs=180),
                                  params=model_params)

#train_input_fn = tf.estimator.inputs.numpy_input_fn()

def get_input_fn(num_epochs=None, shuffle=True):
    import pscreening
    g = pscreening.get_batches('train', 5, (5, 80, 180), batch_size=100)
    x, y = next(g)
    y = y.reshape(y.shape[0], 1)
    
    return tf.estimator.inputs.numpy_input_fn(x={'x': x}, 
                                              y=y,
                                              batch_size=25,
                                              num_epochs=num_epochs, 
                                              shuffle=shuffle)

def get_eval_fn(num_epochs=None, shuffle=True):
    import pscreening
    g = pscreening.get_batches('valid', 5, (5, 80, 180), batch_size=100)
    x, y = next(g)
    
    return tf.estimator.inputs.numpy_input_fn(x={'x': x}, 
                                              y=y,
                                              batch_size=10,
                                              num_epochs=num_epochs, 
                                              shuffle=shuffle)


def go():
    LEARNING_RATE = 0.001
    # Set model params
    model_params = {"learning_rate": LEARNING_RATE}
    
    # Instantiate Estimator
    est = build_estimator('weights', model_params)
    
    est.train(input_fn=get_input_fn(num_epochs=3))
