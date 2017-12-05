import pscreening
import glob
import tensorflow as tf
import os
import config
import setup_data as sd

def _print_results(results_dict):
    for model_file, results in results_dict.items():
        print(f"{model_file}")
        print(f"{results}")

def test(file_prefix, csv=True):
    import csv

    model_files = glob.glob(os.path.join(config.PSCREENING_HOME, config.MODEL_DIR, f"{file_prefix}*.h5"))
    
    with open(f"{file_prefix}.csv", 'w') as results_file:
        wr = csv.writer(results_file, delimiter=',')

        try:
            results_dict = {}
            for model_file in model_files:
                tf.keras.backend.clear_session()
        
                model_file = os.path.basename(model_file)
                val_loss = float(sd.get_file_name(model_file).split('-')[-1])
                results = pscreening.testm(model_file)
                results_dict[model_file] = [val_loss] + results
                wr.writerow([model_file])
                wr.writerow([[val_loss] + results])
                print(f"{model_file}")
                print(f"{[val_loss] + results}")
                
        except:
            _print_results(results_dict)
            
        
    print(f"Testing completed. Printing results...")
    
    _print_results(results_dict)

def train(zones, epochs=10, batch_size=20, learning_rate=0.001,
          version=None, gpus=4, mtype='vgg16', starting_model_file=None,
          img_dim=224, channels=1, train_layer_start=None, run_tests=False):
    
    file_prefix = pscreening.train(zones, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                   version=version, gpus=gpus, mtype=mtype, starting_model_file=starting_model_file,
                                   img_dim=img_dim, channels=channels, train_layer_start=train_layer_start)
    
    print(f"Training completed. File prefix: {file_prefix}. Running tests...")
    
    if run_tests:
        test(file_prefix)    
    