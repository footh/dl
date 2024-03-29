import os

PSCREENING_HOME = os.getenv('PSCREENING_HOME', '.')
RAW_DATA_DIR = 'raw-data'
TRAINING_DIR = 'training'
WEIGHTS_DIR = 'weights'
MODEL_DIR = 'models'
SUBMISSION_DIR = 'submissions'
PSCREENING_LOCAL_HOME = os.getenv('PSCREENING_LOCAL_HOME', '.')

SUBMISSION_MODELS = [
    'zone5-vgg16-d150-c1-e10-bs20-lr001-20171202-010615-10-0.017.h5',
    'zone17-vgg16-d150-c1-e15-bs20-lr001-20171207-004002-08-0.024.h5',
    'zone9-vgg16-d150-c1-e10-bs20-lr001-20171202-051459-08-0.047.h5'
]

SUBMISSION_MODEL_DICT = {
    'zone1-inception-d150-c3-e15-bs23-lr001-20171209-002357-15-0.116.h5': [[1,2], [3,4]],
    'zone6-inception-d150-c3-e20-bs23-lr001-20171210-102044-13-0.086.h5': [[6], [7], [8], [10], [11], [12], [13], [14], [15], [16]]
}