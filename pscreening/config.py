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
    'zone1-inception-d150-c3-e10-bs20-lr001-20171203-174625-07-0.112.h5',
    'zone1-vgg16-d150-c1-e10-bs20-lr001-20171202-041353-08-0.168.h5',
    'zone3-inception-d150-c3-e10-bs20-lr001-20171203-192416-05-0.133.h5',
    'zone3-vgg16-d150-c1-e10-bs20-lr001-20171202-044239-07-0.096.h5',
    'zone8-vgg16-d150-c1-e15-bs20-lr001-20171206-165759-13-0.042.h5',
    'zone9-vgg16-d150-c1-e10-bs20-lr001-20171202-051459-08-0.047.h5',
    'zone10-vgg16-d150-c1-e15-bs20-lr001-20171206-173843-09-0.074.h5',
    'zone11-vgg16-d150-c1-e15-bs20-lr001-20171206-185310-12-0.033.h5',
    'zone13-vgg16-d150-c1-e15-bs20-lr001-20171206-192652-10-0.199.h5',
    'zone15-vgg16-d150-c1-e15-bs20-lr001-20171206-205741-13-0.154.h5',
    'zone12-vgg16-d150-c1-e10-bs20-lr001-20171202-030907-10-0.026.h5',
    'zone14-vgg16-d150-c1-e10-bs20-lr001-20171202-033344-08-0.007.h5',
    'zone16-vgg16-d150-c1-e10-bs20-lr001-20171202-034801-10-0.017.h5'
]

SUBMISSION_MODEL_DICT = {
    'zone6-inception-d150-c3-e15-bs23-lr001-20171207-215728-08-0.059.h5': [[6], [7]]
}