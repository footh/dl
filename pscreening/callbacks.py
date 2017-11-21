# COPYRIGHT
# 
# All contributions by François Chollet:
# Copyright (c) 2015, François Chollet.
# All rights reserved.
# 
# All contributions by Google:
# Copyright (c) 2015, Google, Inc.
# All rights reserved.
# 
# All contributions by Microsoft:
# Copyright (c) 2017, Microsoft, Inc.
# All rights reserved.
# 
# All other contributions:
# Copyright (c) 2015 - 2017, the respective contributors.
# All rights reserved.
# 
# Each contributor holds copyright over their respective contributions.
# The project versioning (Git) records all such contribution source information.
# 
# LICENSE
# 
# The MIT License (MIT)
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
import numpy as np

class ModelCheckpoint(tf.keras.callbacks.Callback):
  """Save the model after every epoch.
  `filepath` can contain named formatting options,
  which will be filled the value of `epoch` and
  keys in `logs` (passed in `on_epoch_end`).
  For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
  then the model checkpoints will be saved with the epoch number and
  the validation loss in the filename.
  Arguments:
      filepath: string, path to save the model file.
      monitor: quantity to monitor.
      verbose: verbosity mode, 0 or 1.
      save_best_only: if `save_best_only=True`,
          the latest best model according to
          the quantity monitored will not be overwritten.
      mode: one of {auto, min, max}.
          If `save_best_only=True`, the decision
          to overwrite the current save file is made
          based on either the maximization or the
          minimization of the monitored quantity. For `val_acc`,
          this should be `max`, for `val_loss` this should
          be `min`, etc. In `auto` mode, the direction is
          automatically inferred from the name of the monitored quantity.
      save_weights_only: if True, then only the model's weights will be
          saved (`model.save_weights(filepath)`), else the full model
          is saved (`model.save(filepath)`).
      period: Interval (number of epochs) between checkpoints.
  """

  def __init__(self,
               filepath,
               monitor='val_loss',
               verbose=0,
               save_best_only=False,
               save_weights_only=False,
               mode='auto',
               period=1,
               multi_gpu=False):
    super(ModelCheckpoint, self).__init__()
    self.monitor = monitor
    self.verbose = verbose
    self.filepath = filepath
    self.save_best_only = save_best_only
    self.save_weights_only = save_weights_only
    self.period = period
    self.multi_gpu=multi_gpu
    self.epochs_since_last_save = 0
    

    if mode not in ['auto', 'min', 'max']:
      logging.warning('ModelCheckpoint mode %s is unknown, '
                      'fallback to auto mode.' % mode)
      mode = 'auto'

    if mode == 'min':
      self.monitor_op = np.less
      self.best = np.Inf
    elif mode == 'max':
      self.monitor_op = np.greater
      self.best = -np.Inf
    else:
      if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
        self.monitor_op = np.greater
        self.best = -np.Inf
      else:
        self.monitor_op = np.less
        self.best = np.Inf

  def _get_trained_model(self):
    if self.multi_gpu:  
        return [l for l in self.model.layers if l.__class__.__name__ == 'Model'][0]
    else:
        return self.model

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    self.epochs_since_last_save += 1
    if self.epochs_since_last_save >= self.period:
      self.epochs_since_last_save = 0
      filepath = self.filepath.format(epoch=epoch + 1, **logs)
      if self.save_best_only:
        current = logs.get(self.monitor)
        if current is None:
          logging.warning('Can save best model only with %s available, '
                          'skipping.' % (self.monitor))
        else:
          if self.monitor_op(current, self.best):
            if self.verbose > 0:
              print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                    ' saving model to %s' % (epoch + 1, self.monitor, self.best,
                                             current, filepath))
            self.best = current
            if self.save_weights_only:
              _get_trained_model.save_weights(filepath, overwrite=True)
            else:
              _get_trained_model.save(filepath, overwrite=True)
          else:
            if self.verbose > 0:
              print('Epoch %05d: %s did not improve' % (epoch + 1,
                                                        self.monitor))
      else:
        if self.verbose > 0:
          print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
        if self.save_weights_only:
          _get_trained_model.save_weights(filepath, overwrite=True)
        else:
          _get_trained_model.save(filepath, overwrite=True)
    