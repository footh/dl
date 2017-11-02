# NOTES: In TF Keras video, model was compiled and then used in an Experiment class? This appears to be older code as the docs
# only show the estimator as the argument

def model_fn(features, targets, mode, params):
        #TODO: Does features include batch size?
    
       img_shape = features.get_shape()[1:3] + (3,)
       
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

        frame_input = tf.contrib.keras.layers.Input(shape=self.input_shape)
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

        loss = tf.losses.sigmoid_cross_entropy(targets, predictions)
        #TODO: Classes are weighted by the gradient_multipliers argument. See: https://www.tensorflow.org/api_docs/python/tf/contrib/layers/optimize_loss
        # See: https://medium.com/towards-data-science/https-medium-com-manishchablani-useful-keras-features-4bac0724734c
        train_op = tf.contrib.layers.optimize_loss(loss=loss, 
                                                   global_step=tf.contrib.framework.get_global_step(), 
                                                   learning_rate=params["learning_rate"], 
                                                   optimizer="Adam") #  'SGD', 'Adam', 'Adagrad'

        #TODO: What to do for binary result?
        predictions_dict = {"ages": predictions}
        
        #TODO: Is this all I need?
        eval_metric_ops = {
            "acc": tf.metrics.accuracy(tf.cast(targets, tf.float32), predictions)
        }
        
        #TODO: See: https://www.tensorflow.org/api_docs/python/tf/estimator/EstimatorSpec
        return tf.estimator.EstimatorSpec(mode=mode, 
                                          predictions=predictions_dict, 
                                          loss=loss, 
                                          train_op=train_op, 
                                          eval_metric_ops=eval_metric_ops)
        

from tensorflow.contrib.learn.python.learn.estimators import estimator
LEARNING_RATE = 0.001
# Set model params
model_params = {"learning_rate": LEARNING_RATE}

# Instantiate Estimator
est = estimator.Estimator(model_fn=model_fn, params=model_params)

#est.fit(x=training_set.data, y=training_set.target, steps=5000)