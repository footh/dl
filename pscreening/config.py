import os

PSCREENING_HOME = os.getenv('PSCREENING_HOME', '.')
RAW_DATA_DIR = 'raw-data'
TRAINING_DIR = 'training'
WEIGHTS_DIR = 'weights'
SUBMISSION_WEIGHTS_DIR = 'submission-weights'
PSCREENING_LOCAL_HOME = os.getenv('PSCREENING_LOCAL_HOME', '.')