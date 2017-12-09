Setting up environment:

1. Install Python 3.6 (I used the anaconda distribution)
2. Install Tensorflow via this command:

pip install ./tf_builds/tensorflow-1.4.0rc1-cp36-cp36m-linux_x86_64.whl

This is custom built tensorflow that fixes an issue based on PR this:

Though that PR hasn't been merged, the issue has been fixed though it is not in any release of tensorflow as
of this writing.

Running predictions:

1. Place all the stage 2 .a3daps files in the ./raw-data/all directory.

2. Within a python CLI run:

```python
import setup_data as sd
sd.points_file()
```

This will pre-compute rectangles for the various zones and place it in a file called 'points-all.csv'.
It takes about 4 seconds per file. This could be done on the fly during prediction, but with multiple models it's
faster to just do it ahead of time.

3. Within a python CLI run:

```python
import pscreening
pscreening.build_submission_file(src='all')
```



'models' directory
'submissions' directory
'submission-artifacts' directory
'raw-data/all' directory
'tf_builds' directory

Though labels aren't used, stage1_labels.csv needs to be in root because of calls to label_dict.
