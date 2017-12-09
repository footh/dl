Setting up environment:

1. Install Python 3.6 (I used the anaconda distribution)
2. Install Tensorflow via this command:

pip install ./tf_builds/tensorflow-1.4.0rc1-cp36-cp36m-linux_x86_64.whl

This is custom built tensorflow that fixes an issue based on PR this:

Though that PR hasn't been merged, the issue has been fixed though it is not in any release of tensorflow as
of this writing.

Running predictions:

1. Place all the stage 2 .a3daps files in the ./raw-data/all directory.

2. Open a python CLI and run:

import setup_data as sd
sd.stage2_setup()

This script will pre-compute rectangles for the various zones and place it in a file called 'points-all.csv'.
It takes about 4 seconds per file. This could be done on the fly during prediction, but with multiple models it's
faster to just do it ahead of time.




'models' directory
'submissions' directory
'raw-data/all' directory
'tf_builds' directory

Though labels aren't used, stage1_labels.csv needs to be in root because of calls to label_dict.
