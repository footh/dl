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

def _get_available_devices():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


def multi_gpu_model(model, gpus):
    """Replicates a model on different GPUs.
    Specifically, this function implements single-machine
    multi-GPU data parallelism. It works in the following way:
    - Divide the model's input(s) into multiple sub-batches.
    - Apply a model copy on each sub-batch. Every model copy
        is executed on a dedicated GPU.
    - Concatenate the results (on CPU) into one big batch.
    E.g. if your `batch_size` is 64 and you use `gpus=2`,
    then we will divide the input into 2 sub-batches of 32 samples,
    process each sub-batch on one GPU, then return the full
    batch of 64 processed samples.
    This induces quasi-linear speedup on up to 8 GPUs.
    This function is only available with the TensorFlow backend
    for the time being.
    # Arguments
        model: A Keras model instance. To avoid OOM errors,
            this model could have been built on CPU, for instance
            (see usage example below).
        gpus: Integer >= 2, number of on GPUs on which to create
            model replicas.
    # Returns
        A Keras `Model` instance which can be used just like the initial
        `model` argument, but which distributes its workload on multiple GPUs.
    # Example
    ```python
        import tensorflow as tf
        from keras.applications import Xception
        from keras.utils import multi_gpu_model
        import numpy as np
        num_samples = 1000
        height = 224
        width = 224
        num_classes = 1000
        # Instantiate the base model
        # (here, we do it on CPU, which is optional).
        with tf.device('/cpu:0'):
            model = Xception(weights=None,
                             input_shape=(height, width, 3),
                             classes=num_classes)
        # Replicates the model on 8 GPUs.
        # This assumes that your machine has 8 available GPUs.
        parallel_model = multi_gpu_model(model, gpus=8)
        parallel_model.compile(loss='categorical_crossentropy',
                               optimizer='rmsprop')
        # Generate dummy data.
        x = np.random.random((num_samples, height, width, 3))
        y = np.random.random((num_samples, num_classes))
        # This `fit` call will be distributed on 8 GPUs.
        # Since the batch size is 256, each GPU will process 32 samples.
        parallel_model.fit(x, y, epochs=20, batch_size=256)
    ```
    """
    if gpus <= 1:
        raise ValueError('For multi-gpu usage to be effective, '
                         'call `multi_gpu_model` with `gpus >= 2`. '
                         'Received: `gpus=%d`' % gpus)

    target_devices = ['/device:CPU:0'] + ['/device:GPU:%d' % i for i in range(gpus)]
    available_devices = _get_available_devices()
    for device in target_devices:
        if device not in available_devices:
            raise ValueError(
                'To call `multi_gpu_model` with `gpus=%d`, '
                'we expect the following devices to be available: %s. '
                'However this machine only has: %s. '
                'Try reducing `gpus`.' % (gpus,
                                          target_devices,
                                          available_devices))

    def get_slice(data, i, parts):
#         data: Tensor("input_2:0", shape=(?, 5, 80, 180, 1), dtype=float32)
#         i: 0
#         parts: 2

#         shape: Tensor("replica_0/lambda/Shape:0", shape=(5,), dtype=int32, device=/device:GPU:0)
#         batch_size: Tensor("replica_0/lambda/strided_slice:0", shape=(1,), dtype=int32, device=/device:GPU:0)
#         input_shape: Tensor("replica_0/lambda/strided_slice_1:0", shape=(4,), dtype=int32, device=/device:GPU:0)
#         step: Tensor("replica_0/lambda/floordiv:0", shape=(1,), dtype=int32, device=/device:GPU:0)
#         size(1): Tensor("replica_0/lambda/floordiv:0", shape=(1,), dtype=int32, device=/device:GPU:0)
#         size(2): Tensor("replica_0/lambda/concat:0", shape=(5,), dtype=int32, device=/device:GPU:0)
#         stride: Tensor("replica_0/lambda/concat_1:0", shape=(5,), dtype=int32, device=/device:GPU:0)
#         start: Tensor("replica_0/lambda/mul_1:0", shape=(5,), dtype=int32, device=/device:GPU:0)
        
        print(f"data: {data}")
        print(f"i: {i}")
        print(f"parts: {parts}")
        shape = tf.shape(data)
        print(f"shape: {shape}")
        batch_size = shape[:1]
        print(f"batch_size: {batch_size}")
        input_shape = shape[1:]
        print(f"input_shape: {input_shape}")
        step = batch_size // parts
        print(f"step: {step}")
        if i == gpus - 1:
            size = batch_size - step * i
        else:
            size = step
        print(f"size(1): {size}")
        size = tf.concat([size, input_shape], axis=0)
        print(f"size(2): {size}")
        stride = tf.concat([step, input_shape * 0], axis=0)
        print(f"stride: {stride}")
        start = stride * i
        print(f"start: {start}")
        return tf.slice(data, start, size)
    
    def get_slice2(data, i, parts):
        # x: Tensor("input_2:0", shape=(?, 5, 80, 180, 1), dtype=float32)
        # step: 10
        # size(1): 10
        # start: 0
        # range(start, start+size: range(0, 10)
        # slice_i: Tensor("replica_0/lambda/Gather:0", shape=(10, 5, 80, 180, 1), dtype=float32, device=/device:GPU:0)
        batch_size = 20
        input_shape = tf.constant([5, 80, 180, 1])
        step = batch_size // parts
        print(f"step: {step}")
        if i == gpus - 1:
            size = batch_size - step * i
        else:
            size = step
        print(f"size(1): {size}")
        start = step * i
        print(f"start: {start}")
        print(f"range(start, start+size: {range(start, start+size)}")
        return tf.gather(data, list(range(start, start+size)))

    all_outputs = []
    for i in range(len(model.outputs)):
        all_outputs.append([])

    # Place a copy of the model on each GPU,
    # each getting a slice of the inputs.
    for i in range(gpus):
        with tf.device('/device:GPU:%d' % i):
            with tf.name_scope('replica_%d' % i):
                inputs = []
                # Retrieve a slice of the input.
                for x in model.inputs:
                    # x: Tensor("input_2:0", shape=(?, 5, 80, 180, 1), dtype=float32)
                    print(f"x: {x}")
                    #input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_i = tf.keras.layers.Lambda(get_slice2, 
                                                     #output_shape=input_shape,
                                                     arguments={'i': i,'parts': gpus})(x)
                    inputs.append(slice_i)
                    print(f"slice_i: {slice_i}")
                    # slice_i: Tensor("replica_0/lambda/Slice:0", shape=(?, ?, ?, ?, ?), dtype=float32, device=/device:GPU:0)

                # Apply model on slice
                # (creating a model replica on the target device).
                outputs = model(inputs)
                print(f"outputs: {outputs}")
                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save the outputs for merging back together later.
                for o in range(len(outputs)):
                    all_outputs[o].append(outputs[o])

    # Merge outputs on CPU.
    with tf.device('/device:CPU:0'):
        merged = []
        for outputs in all_outputs:
            merged.append(tf.keras.layers.concatenate(outputs, axis=0))
            
        return tf.keras.models.Model(model.inputs, merged)
    
