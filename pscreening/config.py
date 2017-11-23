import os

PSCREENING_HOME = os.getenv('PSCREENING_HOME', '.')
RAW_DATA_DIR = 'raw-data'
TRAINING_DIR = 'training'
WEIGHTS_DIR = 'weights'
MODEL_DIR = 'models'
SUBMISSION_DIR = 'submissions'
PSCREENING_LOCAL_HOME = os.getenv('PSCREENING_LOCAL_HOME', '.')

SUBMISSION_MODELS = [
    'zone5-vgg16-d200-c1-e2-bs32-lr001-20171121-052129-02-0.102.h5',
    'zone17-vgg16-d200-c1-e3-bs32-lr001-20171121-053939-03-0.012.h5',
    'zone1-vgg16-d200-c1-e6-bs32-lr001-20171121-150637-06-0.394.h5',
    'zone3-vgg16-d200-c1-e6-bs32-lr001-20171121-155008-04-0.324.h5',
    'zone6-vgg16-d200-c1-e6-bs32-lr001-20171121-161816-04-0.125.h5',
    'zone7-vgg16-d200-c1-e6-bs32-lr001-20171121-164023-06-0.264.h5',
    'zone8-vgg16-d200-c1-e6-bs32-lr001-20171121-172658-06-0.142.h5',
    'zone9-vgg16-d200-c1-e6-bs32-lr001-20171121-225605-06-0.176.h5',
    'zone10-vgg16-d200-c1-e6-bs32-lr001-20171122-023138-05-0.119.h5',
    'zone11-vgg16-d200-c1-e10-bs32-lr001-20171122-194622-09-0.247.h5',
    'zone12-vgg16-d200-c1-e10-bs32-lr001-20171122-134713-05-0.289.h5'
]