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
    'zone17-vgg16-d150-c1-e10-bs20-lr001-20171202-040224-02-0.010.h5',
    'zone1-vgg16-d150-c1-e10-bs20-lr001-20171202-041353-08-0.168.h5',
    'zone3-vgg16-d150-c1-e10-bs20-lr001-20171202-044239-07-0.096.h5',
    'zone6-vgg16-d150-c1-e10-bs20-lr001-20171202-011939-07-0.018.h5',
    'zone7-vgg16-d150-c1-e10-bs20-lr001-20171202-045855-08-0.012.h5',
    'zone8-vgg16-d150-c1-e10-bs20-lr001-20171202-015247-08-0.010.h5',
    'zone9-vgg16-d150-c1-e10-bs20-lr001-20171202-051459-08-0.047.h5',
    'zone10-vgg16-d150-c1-e10-bs20-lr001-20171202-022606-07-0.001.h5',
    'zone11-vgg16-d150-c1-e10-bs20-lr001-20171202-024033-08-0.034.h5',
    'zone13-vgg16-d150-c1-e10-bs20-lr001-20171202-025416-09-0.004.h5',
    'zone15-vgg16-d150-c1-e10-bs20-lr001-20171202-004003-08-0.028.h5',
    'zone12-vgg16-d150-c1-e10-bs20-lr001-20171202-030907-10-0.026.h5',
    'zone14-vgg16-d150-c1-e10-bs20-lr001-20171202-033344-08-0.007.h5',
    'zone16-vgg16-d150-c1-e10-bs20-lr001-20171202-034801-10-0.017.h5'
]