## Passenger Screening Algorithm Challenge

* zones.py: used to parse the 16 frame files into the 17 body zone parts 
* zones_config.py: static configuration for parsing the body zones, padding for example
* zone_generator.py, zone_aps_generator.py, zone_aps_generator2.py: generator code that produces zones input for training. The latter one, which works directly with aps files was used in the final solution
* util.py: contains utility methods, mainly for reading the unique file types and for debugging
* trainer.py: allows for single entry point to train model
* tf_util.py: allows for training on multiple GPUs. File taken directly from Keras library and modified to fix bugs
* tf_trainer.py: playing with the Tensorflow Estimator interface. Ultimately not used.
* setup_data.py: contains methods to preprocess and prepare the data for training
* sandbox.py: experimental code
* pscreening.py: contains parent model class and child classes that implement a specific pre-trained model. Contains training and evaluation code as well as the code that generates the submission file.
* config.py: global static configuration
* callbacks.py: taken from Keras, allows for saving of model at epoch end

## Setting up environment:

1. Install Python 3.6. I used the anaconda distribution and tested these instructions on this recent release:
```
Version 5.0.1 | Release Date: October 25, 2017
```
Which is found at this link:
https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh

By using this distribution, I didn't have to manually install any other dependencies.

2. Install Tensorflow via this command (assuming you are in the project directory):

pip install ./tf_builds/tensorflow-1.4.0rc1-cp36-cp36m-linux_x86_64.whl

This is custom built tensorflow that fixes an issue based on this PR authored by yours truly:
https://github.com/tensorflow/tensorflow/pull/14377

Though that PR hasn't been merged, the issue has been fixed from another PR but it is not in any release of 
tensorflow as of this writing.

3. I'm assuming the instance this will be running on is loaded with a CUDA-enabled nVidia GPU.

## Running predictions:

1. Place all the stage 2 **.a3daps** files **only** in the *./raw-data/all* directory.

2. Within a python CLI run:

```python
import setup_data as sd
sd.points_file()
```

This will pre-compute rectangles for the various zones/slices and place it in a file called 'points-all.csv'.
It takes ~3-4 seconds *per* file. This could be done on the fly during prediction, but with multiple models it's
faster to just do it ahead of time.

3. Within a python CLI run:

```python
import pscreening
pscreening.build_submission_file(src='all')
```

The submission file will be written to the *./submissions* directory in the format *submission-date-time.csv*

Note: I'll occasionally get this error when building the submission file:

```
InternalError (see above for traceback): tensorflow/core/kernels/cuda_solvers.cc:541: cuSolverDN call failed with status =7
     [[Node: lstm_1/recurrent_kernel/Initializer/Qr = Qr[T=DT_FLOAT, _class=["loc:@lstm_1/recurrent_kernel"], full_matrices=false, _device="/job:localhost/replica:0/task:0/device:GPU:0"](lstm_1/recurrent_kernel/Initializer/random_normal)]]
```
This will happen when resources are constrained (like for instance when I don't shut down X this error will occur on my meager 4GB GPU). I just want to point out that this is a transient error and not a bug in the code. The method can simply be run again if this happens.
