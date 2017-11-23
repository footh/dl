import pscreening
import glob
import tensorflow as tf

def train(zones, epochs=10, batch_size=32, learning_rate=0.001,
          version=None, gpus=4, mtype='vgg16', starting_model_file=None,
          img_dim=200, channels=1):
    
    file_prefix = pscreening.train(zones, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                   version=version, gpus=gpus, mtype=mtype, starting_model_file=starting_model_file,
                                   img_dim=200, channels=channels)
    
    print(f"Training completed. File prefix: {file_prefix}. Running tests...")
    
    model_files = glob.glob(f"{file_prefix}*.h5")
    
    tf.keras.backend.clear_session()
    results_dict = {}
    for model_file in model_files:
        results = pscreening.testm(model_file)
        results_dict[model_file] = results
        
    print(f"Testing completed. Printing results...")
    
    for model_file, results in results_dict.items():
        print(f"{model_file}")
        print(f"{results}")
    
    