from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import glob
import logging
import pandas as pd
import PIL.Image
import numpy as np
import os
import re


def read_charset(filename):
    pattern = re.compile(r'(\d+)\t(.+)')
    charset = {}
    with tf.gfile.GFile(filename) as f:
        for i, line in enumerate(f):
            m = pattern.match(line)
            if m is None:
                logging.info('incorrect charset file. line #{}: {}'.format(i, line))
                continue
            code = int(m.group(1)) + 1
            char = m.group(2)
            if char == '<nul>':
                continue
            charset[code] = char
        inv_charset = {}
        for k, v in charset.items():
            inv_charset[v] = k
        return charset, inv_charset


def get_str_labels(char_map, v, add_eos=True):
    result = []
    for t in v:
        i = char_map.get(t, -1)
        if i >= 0:
            result.append(i)
    if add_eos:
        result.append(len(char_map) + 1)
    return result


def null_dataset():
    def _input_fn():
        return None

    return _input_fn


def get_input_fn(dataset_type):
    if dataset_type == 'synth-crop':
        return input_fn
    else:
        return generated_input_fn


def generated_input_fn(params, is_training):
    char_map = params['charset']
    batch_size = params['batch_size']
    inputs = []
    for img_file in glob.iglob(params['data_set'] + '/*.jpg'):
        name = os.path.basename(img_file)
        names = name.split('_')
        if len(names) > 1:
            label = get_str_labels(char_map, names[0])
            if len(label) < 2:
                continue
            inputs.append([img_file, names[0]])

    inputs = sorted(inputs, key=lambda row: row[0])
    input_size = len(inputs)
    logging.info('Dataset size {}'.format(input_size))

    def _input_fn():
        dataset = tf.data.Dataset.from_tensor_slices(inputs)
        shuffle_size = input_size
        if is_training:
            logging.info("Shuffle by %d", shuffle_size)
            if shuffle_size == 0:
                shuffle_size = 10
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(shuffle_size))

        def _features_labels(images, labels):
            return images, labels

        def _decode(filename_label):
            filename = str(filename_label[0], encoding='UTF-8')
            label = str(filename_label[1], encoding='UTF-8')
            label = get_str_labels(char_map, label)
            image = PIL.Image.open(filename)
            width, height = image.size
            ration_w = max(width / 150.0, 1.0)
            ration_h = max(height / 32.0, 1.0)
            ratio = max(ration_h, ration_w)
            if ratio > 1:
                width = int(width / ratio)
                height = int(height / ratio)
                image = image.resize((width, height))
            image = np.asarray(image)
            pw = max(0, 150 - image.shape[1])
            ph = max(0, 32 - image.shape[0])
            image = np.pad(image, ((0, ph), (0, pw), (0, 0)), 'constant', constant_values=0)
            image = image.astype(np.float32) / 127.5 - 1
            return image, np.array(label, dtype=np.int32)

        dataset = dataset.map(
            lambda filename_label: tuple(tf.py_func(_decode, [filename_label], [tf.float32, tf.int32])),
            num_parallel_calls=1)

        dataset = dataset.padded_batch(batch_size, padded_shapes=([32, 150, 3], [None]))
        dataset = dataset.map(_features_labels, num_parallel_calls=1)
        dataset = dataset.prefetch(2)
        return dataset

    return _input_fn


def input_fn(params, is_training):
    char_map = params['charset']
    labels = pd.read_csv(params['data_set'] + '/labels.csv', converters={'text': str}, na_values=[],
                         keep_default_na=False)
    limit = params['limit_train']
    if limit is None or limit < 1:
        alldata = labels.iloc[:].values
    else:
        alldata = labels.iloc[:limit].values
    batch_size = params['batch_size']
    count = len(alldata) // batch_size

    def _input_fn():
        def _gen():
            for _ in range(params['epoch']):
                data = np.random.permutation(alldata)
                maxlen = 0
                for j in range(count):
                    features = []
                    labels = []
                    for i in range(batch_size):
                        k = j * batch_size + i
                        image = PIL.Image.open('{}/{}.png'.format(params['data_set'], data[k, 0]))
                        width, height = image.size
                        # logging.info("Width: {} Height: {}".format(width,height))
                        ration_w = max(width / 150.0, 1.0)
                        ration_h = max(height / 32.0, 1.0)
                        ratio = max(ration_h, ration_w)
                        if ratio > 1:
                            width = int(width / ratio)
                            height = int(height / ratio)
                            image = image.resize((width, height))
                            w1, h1 = image.size
                            # logging.info("Resize Width: {} Height: {}".format(w1,h1))
                        image = np.asarray(image)
                        pw = max(0, 150 - image.shape[1])
                        ph = max(0, 32 - image.shape[0])
                        image = np.pad(image, ((0, ph), (0, pw), (0, 0)), 'constant', constant_values=0)
                        image = image.astype(np.float32) / 127.5 - 1
                        # logging.info("Text {}".format(data[k,1]))
                        label = get_str_labels(char_map, data[k, 1])
                        features.append(image)
                        if len(label) > maxlen:
                            maxlen = len(label)
                        labels.append(np.array(label, dtype=np.int32))
                    for i in range(len(labels)):
                        l = len(labels[i])
                        if l < maxlen:
                            labels[i] = np.pad(labels[i], (0, maxlen - l), 'constant', constant_values=0)

                    yield (np.stack(features), np.stack(labels))

        ds = tf.data.Dataset.from_generator(_gen, (tf.float32, tf.int32), (
            tf.TensorShape([params['batch_size'], 32, 150, 3]),
            tf.TensorShape([params['batch_size'], None])))
        ds = ds.prefetch(4)
        return ds

    return _input_fn


def _basic_lstm(mode, params, rnn_inputs):
    with tf.variable_scope('LSTM'):
        layers_list = []
        for _ in range(params['num_layers']):
            cell = tf.nn.rnn_cell.BasicLSTMCell(params['hidden_size'], state_is_tuple=True)
            layers_list.append(cell)
        cell = tf.nn.rnn_cell.MultiRNNCell(layers_list, state_is_tuple=True)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # make rnn state for training
        with tf.variable_scope('Hidden_state'):
            state_variables = []
            for state_c, state_h in cell.zero_state(params['batch_size'], tf.float32):
                state_variables.append(tf.nn.rnn_cell.LSTMStateTuple(
                    tf.Variable(state_c, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES]),
                    tf.Variable(state_h, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])))
            rnn_state = tuple(state_variables)
    else:
        # use default for evaluation
        rnn_state = cell.zero_state(params['batch_size'], tf.float32)
    with tf.name_scope('LSTM'):
        rnn_output, new_states = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=rnn_state, time_major=True)
    return rnn_output, rnn_state, new_states


def _cudnn_lstm_compatible(params, rnn_inputs):
    if params['lstm_direction_type'] == 'bidirectional':
        with tf.variable_scope('cudnn_lstm'):
            single_cell = lambda: tf.contrib.rnn.BasicLSTMCell(params['hidden_size'], forget_bias=0,
                                                               name="cudnn_compatible_lstm_cell")
            cells_fw = [single_cell() for _ in range(params['num_layers'])]
            cells_bw = [single_cell() for _ in range(params['num_layers'])]
            rnn_state_fw = [cell.zero_state(params['batch_size'], tf.float32) for cell in cells_fw]
            rnn_state_bw = [cell.zero_state(params['batch_size'], tf.float32) for cell in cells_bw]
        with tf.variable_scope('cudnn_lstm'):
            rnn_output, new_state_fw, new_states_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                cells_fw,
                cells_bw,
                rnn_inputs,
                initial_states_fw=rnn_state_fw,
                initial_states_bw=rnn_state_bw,
                sequence_length=None,
                time_major=True)
        return rnn_output, rnn_state_fw + rnn_state_bw, new_state_fw + new_states_bw
    else:
        with tf.variable_scope('cudnn_lstm'):
            single_cell = lambda: tf.contrib.rnn.BasicLSTMCell(params['hidden_size'], forget_bias=0,
                                                               name="cudnn_compatible_lstm_cell")
            cells = [single_cell() for _ in range(params['num_layers'])]
            cell = tf.nn.rnn_cell.MultiRNNCell(cells)
            rnn_state = cell.zero_state(params['batch_size'], tf.float32)
        with tf.variable_scope('cudnn_lstm'):
            rnn_output, new_states = tf.nn.dynamic_rnn(cell, rnn_inputs, sequence_length=None,
                                                       initial_state=rnn_state, time_major=True)
        return rnn_output, rnn_state, new_states


def _cudnn_lstm(mode, params, rnn_inputs):
    with tf.variable_scope('LSTM'):
        dir = 2 if params['lstm_direction_type'] == 'bidirectional' else 1
        cell = tf.contrib.cudnn_rnn.CudnnLSTM(params['num_layers'], params['hidden_size'],
                                              direction=params['lstm_direction_type'],
                                              dropout=float(params['output_keep_prob']))
        shape = [params['num_layers'] * dir, params['batch_size'], params['hidden_size']]
        rnn_state = (
            tf.Variable(tf.zeros(shape, tf.float32), trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES]),
            tf.Variable(tf.zeros(shape, tf.float32), trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES]))
    with tf.name_scope('LSTM'):
        rnn_output, new_states = cell(rnn_inputs, initial_state=rnn_state,
                                      training=(mode == tf.estimator.ModeKeys.TRAIN))
    return rnn_output, rnn_state, new_states


def _crnn_model_fn(features, labels, mode, params=None, config=None):
    global_step = tf.train.get_or_create_global_step()
    logging.info("Features {}".format(features.shape))
    images = tf.transpose(features, [0, 2, 1, 3])
    logging.info("Images {}".format(images.shape))
    if (mode == tf.estimator.ModeKeys.TRAIN or
            mode == tf.estimator.ModeKeys.EVAL):
        tf.summary.image('image', features)
        idx = tf.where(tf.not_equal(labels, 0))
        sparse_labels = tf.SparseTensor(idx, tf.gather_nd(labels, idx),
                                        [params['batch_size'], params['max_target_seq_length']])
        sparse_labels, _ = tf.sparse_fill_empty_rows(sparse_labels, params['num_labels'] - 1)

    # 64 / 3 x 3 / 1 / 1
    conv1 = tf.layers.conv2d(inputs=images, filters=64, kernel_size=(3, 3), padding="same", activation=tf.nn.relu)
    logging.info("conv1 {}".format(conv1.shape))

    # 2 x 2 / 1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    logging.info("pool1 {}".format(pool1.shape))

    # 128 / 3 x 3 / 1 / 1
    conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=(3, 3), padding="same", activation=tf.nn.relu)
    logging.info("conv2 {}".format(conv2.shape))
    # 2 x 2 / 1
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    logging.info("pool2 {}".format(pool2.shape))

    # 256 / 3 x 3 / 1 / 1
    conv3 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=(3, 3), padding="same", activation=tf.nn.relu)
    logging.info("conv3 {}".format(conv3.shape))

    # Batch normalization layer
    bnorm1 = tf.layers.batch_normalization(conv3)

    # 256 / 3 x 3 / 1 / 1
    conv4 = tf.layers.conv2d(inputs=bnorm1, filters=256, kernel_size=(3, 3), padding="same", activation=tf.nn.relu)
    logging.info("conv4 {}".format(conv4.shape))

    # 1 x 2 / 1
    pool3 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=[1, 2], padding="same")
    logging.info("pool3 {}".format(pool3.shape))

    # 512 / 3 x 3 / 1 / 1
    conv5 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=(3, 3), padding="same", activation=tf.nn.relu)
    logging.info("conv5 {}".format(conv5.shape))

    # Batch normalization layer
    bnorm2 = tf.layers.batch_normalization(conv5)

    # 512 / 3 x 3 / 1 / 1
    conv6 = tf.layers.conv2d(inputs=bnorm2, filters=512, kernel_size=(3, 3), padding="same", activation=tf.nn.relu)
    logging.info("conv6 {}".format(conv6.shape))

    # 1 x 2 / 2
    pool4 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=[1, 2], padding="same")
    logging.info("pool4 {}".format(pool4.shape))
    # 512 / 2 x 2 / 1 / 0
    conv7 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=(2, 2), padding="valid", activation=tf.nn.relu)
    logging.info("conv7 {}".format(conv7.shape))

    reshaped_cnn_output = tf.reshape(conv7, [params['batch_size'], -1, 512])
    rnn_inputs = tf.transpose(reshaped_cnn_output, perm=[1, 0, 2])

    max_char_count = rnn_inputs.get_shape().as_list()[0]
    logging.info("max_char_count {}".format(max_char_count))
    input_lengths = tf.zeros([params['batch_size']], dtype=tf.int32) + max_char_count
    logging.info("InpuLengh {}".format(input_lengths.shape))

    if params['rnn_type'] == 'CudnnLSTM':
        rnn_output, rnn_state, new_states = _cudnn_lstm(mode, params, rnn_inputs)
    elif params['rnn_type'] == 'CudnnCompatibleLSTM':
        rnn_output, rnn_state, new_states = _cudnn_lstm_compatible(params, rnn_inputs)
    else:
        rnn_output, rnn_state, new_states = _basic_lstm(mode, params, rnn_inputs)

    with tf.variable_scope('Output_layer'):
        logits = tf.layers.dense(rnn_output, params['num_labels'],
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())

    if params['beam_search_decoder']:
        decoded, _log_prob = tf.nn.ctc_beam_search_decoder(logits, input_lengths)
    else:
        decoded, _log_prob = tf.nn.ctc_greedy_decoder(logits, input_lengths)

    prediction = tf.to_int32(decoded[0])

    metrics = {}
    if (mode == tf.estimator.ModeKeys.TRAIN or
            mode == tf.estimator.ModeKeys.EVAL):
        levenshtein = tf.edit_distance(prediction, sparse_labels, normalize=True)
        errors_rate = tf.metrics.mean(levenshtein)
        mean_error_rate = tf.reduce_mean(levenshtein)
        metrics['Error_Rate'] = errors_rate
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.summary.scalar('Error_Rate', mean_error_rate)
        with tf.name_scope('CTC'):
            ctc_loss = tf.nn.ctc_loss(sparse_labels, logits, input_lengths, ignore_longer_outputs_than_inputs=True)
            mean_loss = tf.reduce_mean(tf.truediv(ctc_loss, tf.to_float(input_lengths)))
            loss = mean_loss
    else:
        loss = None

    training_hooks = []

    if mode == tf.estimator.ModeKeys.TRAIN:

        opt = tf.train.AdamOptimizer(params['learning_rate'])
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if params['grad_clip'] is None:
                train_op = opt.minimize(loss, global_step=global_step)
            else:
                gradients, variables = zip(*opt.compute_gradients(loss))
                gradients, _ = tf.clip_by_global_norm(gradients, params['grad_clip'])
                train_op = opt.apply_gradients([(gradients[i], v) for i, v in enumerate(variables)],
                                               global_step=global_step)
    elif mode == tf.estimator.ModeKeys.EVAL:
        train_op = None
    else:
        train_op = None
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = tf.sparse_to_dense(tf.to_int32(prediction.indices),
                                         tf.to_int32(prediction.dense_shape),
                                         tf.to_int32(prediction.values),
                                         default_value=-1,
                                         name="output")
        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
                predictions)}
    else:
        predictions = None
        export_outputs = None
    return tf.estimator.EstimatorSpec(
        mode=mode,
        eval_metric_ops=metrics,
        predictions=predictions,
        loss=loss,
        training_hooks=training_hooks,
        export_outputs=export_outputs,
        train_op=train_op)


class BaseOCR(tf.estimator.Estimator):
    def __init__(
            self,
            params=None,
            model_dir=None,
            config=None,
            warm_start_from=None
    ):
        def _model_fn(features, labels, mode, params, config):
            return _crnn_model_fn(
                features=features,
                labels=labels,
                mode=mode,
                params=params,
                config=config)

        super(BaseOCR, self).__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            config=config,
            params=params,
            warm_start_from=warm_start_from
        )
