from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from aocr.cnn import CNN
import tensorflow as tf
import logging
import glob
import numpy as np
from tensorflow.python.training import training_util
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.nets import inception_v3
from tensorflow.python.training import session_run_hook

ENGLISH_CHAR_MAP = [
    '',
    '_',
    # Alphabet normal
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '-', ':', '(', ')', '.', ',', '/', '$',
    "'",
    " "
]


def read_charset():
    charset = {}
    inv_charset = {}
    for i, v in enumerate(ENGLISH_CHAR_MAP):
        charset[i] = v
        inv_charset[v] = i

    return charset, inv_charset


def tf_input_fn(params, is_training):
    max_width = params['max_width']
    batch_size = params['batch_size']
    _, inv_charset = read_charset()
    blank_label = 1

    if params.get('last_blank_label'):
        inv_charset[" "], inv_charset[""] = inv_charset[""], inv_charset[" "]
        blank_label = len(inv_charset) - 1

    datasets_files = []
    for tf_file in glob.iglob(params['data_set'] + '/*.tfrecord'):
        datasets_files.append(tf_file)
    max_target_seq_length = params['max_target_seq_length']
    max_target_seq_length_1 = max_target_seq_length - 1

    def _input_fn():
        ds = tf.data.TFRecordDataset(datasets_files, buffer_size=256 * 1024 * 1024)

        def _emdeding(label):
            label = str(label, encoding='UTF-8')
            labels = []
            for c in label.lower():
                if c == '_':
                    continue
                v = inv_charset.get(c, -1)
                if v > 0:
                    labels.append(v)
            if len(labels) < 1:
                labels.append(inv_charset[' '])
            if len(labels) > max_target_seq_length_1:
                labels = labels[:max_target_seq_length_1]
            labels.append(blank_label)

            if params.get('normalize_length'):
                if len(labels) < max_target_seq_length:
                    labels.extend([blank_label] * (max_target_seq_length - len(labels)))

            return np.array(labels, dtype=np.int64)

        def _parser(example):
            zero = tf.zeros([1], dtype=tf.int64)
            features = {
                'image/encoded':
                    tf.FixedLenFeature((), tf.string, default_value=''),
                'image/height':
                    tf.FixedLenFeature([1], tf.int64, default_value=zero),
                'image/width':
                    tf.FixedLenFeature([1], tf.int64, default_value=zero),
                'image/text':
                    tf.FixedLenFeature((), tf.string, default_value=''),
            }
            res = tf.parse_single_example(example, features)
            img = tf.image.decode_image(res['image/encoded'], channels=3)
            original_w = tf.cast(res['image/width'][0], tf.int32)
            original_h = tf.cast(res['image/height'][0], tf.int32)
            img = tf.reshape(img, [original_h, original_w, 3])
            w = tf.maximum(tf.cast(original_w, tf.float32), 1.0)
            h = tf.maximum(tf.cast(original_h, tf.float32), 1.0)
            ratio_w = tf.maximum(w / max_width, 1.0)
            ratio_h = tf.maximum(h / 32.0, 1.0)
            ratio = tf.maximum(ratio_w, ratio_h)
            nw = tf.cast(tf.maximum(tf.floor_div(w, ratio), 1.0), tf.int32)
            nh = tf.cast(tf.maximum(tf.floor_div(h, ratio), 1.0), tf.int32)
            img = tf.image.resize_images(img, [nh, nw])
            padw = tf.maximum(0, int(max_width) - nw)
            padh = tf.maximum(0, 32 - nh)
            img = tf.image.pad_to_bounding_box(img, 0, 0, nh + padh, nw + padw)
            img = tf.cast(img, tf.float32) / 127.5 - 1
            label = res['image/text']
            label = tf.py_func(_emdeding, [label], tf.int64)
            logging.info('label out {}'.format(label))
            return img, label

        ds = ds.map(_parser)
        ds = ds.apply(tf.contrib.data.shuffle_and_repeat(1000))
        ds = ds.padded_batch(batch_size, padded_shapes=([32, max_width, 3], [None]), padding_values=(0.0, np.int64(1)))
        return ds

    return _input_fn


def encode_coordinates_fn(net):
    _, h, w, _ = net.shape.as_list()
    x, y = tf.meshgrid(tf.range(w), tf.range(h))
    w_loc = slim.one_hot_encoding(x, num_classes=w)
    h_loc = slim.one_hot_encoding(y, num_classes=h)
    loc = tf.concat([h_loc, w_loc], 2)
    loc = tf.tile(tf.expand_dims(loc, 0), tf.stack([tf.shape(net)[0], 1, 1, 1]))
    return tf.concat([net, loc], 3)


def _inception_v3_arg_scope(is_training=True,
                            weight_decay=0.00004,
                            stddev=0.1,
                            batch_norm_var_collection='moving_vars'):
    """Defines the default InceptionV3 arg scope.
    Args:
      is_training: Whether or not we're training the model.
      weight_decay: The weight decay to use for regularizing the model.
      stddev: The standard deviation of the trunctated normal weight initializer.
      batch_norm_var_collection: The name of the collection for the batch norm
        variables.
    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    batch_norm_params = {
        'is_training': is_training,
        # Decay for the moving averages.
        'decay': 0.9997,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # collection containing the moving mean and moving variance.
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection],
        }
    }
    normalizer_fn = slim.batch_norm

    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
                [slim.conv2d],
                weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                activation_fn=tf.nn.relu6,
                normalizer_fn=normalizer_fn,
                normalizer_params=batch_norm_params) as sc:
            return sc


def _inception(freatures, encode_coordinate):
    with slim.arg_scope(_inception_v3_arg_scope(is_training=True)):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected, slim.batch_norm],
                trainable=True):
            with slim.arg_scope(
                    [slim.batch_norm, slim.dropout], is_training=True):
                net, _ = inception_v3.inception_v3_base(
                    freatures,
                    scope='InceptionV3',
                    final_endpoint='Mixed_5d')
    logging.info("Inception : {}".format(net))
    if encode_coordinate:
        net = encode_coordinates_fn(net)
    batch_size, h, w, f = tf.unstack(tf.shape(net))
    logging.info("Inception : {}".format(net))
    net = tf.squeeze(net, axis=1)
    return net


def _aocr_model_fn(features, labels, mode, params=None, config=None):
    if isinstance(features, dict):
        features = features['images']
    if params['conv'] == 'inception':
        conv_output = _inception(features, True)
    else:
        cnn_model = CNN(features, mode == tf.estimator.ModeKeys.TRAIN)
        conv_output = cnn_model.tf_output()
    _, t, _ = tf.unstack(tf.shape(conv_output))
    input_lengths = tf.zeros((params['batch_size']), dtype=tf.int64) + tf.cast(t, tf.int64)
    with tf.variable_scope('embedding'):
        embedding = tf.get_variable(
            'embedding_op',
            [params['num_labels'], 512],
            initializer=tf.random_uniform_initializer(-1, 1),
            trainable=True,
            dtype=tf.float32)

    enc_state = _encoder(params, conv_output, input_lengths)
    start_tokens = tf.zeros([params['batch_size']], dtype=tf.int64)
    if mode != tf.estimator.ModeKeys.TRAIN:
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding, start_tokens=tf.to_int32(start_tokens), end_token=1)
    else:
        train_output = tf.concat([tf.expand_dims(start_tokens, 1), labels], 1)
        output_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(train_output, 1)), 1)
        #output_lengths = tf.ones([params['batch_size']], tf.int32) * params['max_target_seq_length']
        with tf.variable_scope('embedding'):
            output_embed = tf.nn.embedding_lookup(embedding, train_output)


        logging.info('output_embed {}'.format(output_embed))
        logging.info('output_lengths {}'.format(output_lengths))
        logging.info('TrainingHelper')
        helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(output_embed, output_lengths, embedding, 0.5)

    outputs = _decoder(params, enc_state, conv_output, input_lengths, helper)
    hooks = []
    if mode == tf.estimator.ModeKeys.TRAIN:
        predictions = None
        export_outputs = None
        train_outputs = outputs[0]
        weights = tf.to_float(tf.not_equal(train_output[:, :-1], 1))
        loss = tf.contrib.seq2seq.sequence_loss(train_outputs.rnn_output, labels, weights=weights)
        tf.summary.image('input', features, 2)
        opt = tf.train.AdamOptimizer(params['learning_rate'])
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        logging.info("Updates: {}".format(update_ops))
        with tf.control_dependencies([]):
            if params['grad_clip'] is None:
                train_op = opt.minimize(loss, global_step=tf.train.get_or_create_global_step())
            else:
                logging.info("Clip")
                gradients, variables = zip(*opt.compute_gradients(loss))
                gradients, _ = tf.clip_by_global_norm(gradients, params['grad_clip'])
                train_op = opt.apply_gradients([(gradients[i], v) for i, v in enumerate(variables)],
                                               global_step=tf.train.get_or_create_global_step())
            if params['inception_checkpoint'] is not None:
                hooks.append(IniInceptionHook(params['inception_checkpoint']))
    else:
        train_op = None
        loss = None
        predictions = outputs[0].sample_id
        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
                predictions)}
    return tf.estimator.EstimatorSpec(
        mode=mode,
        eval_metric_ops=None,
        predictions=predictions,
        loss=loss,
        training_hooks=hooks,
        export_outputs=export_outputs,
        train_op=train_op)


def _encoder(params, inputs, input_lengths):
    cells = [tf.contrib.rnn.GRUCell(params['hidden_size']) for _ in range(params['num_layers'])]
    mrnn = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
    _, enc_state = tf.nn.dynamic_rnn(mrnn, inputs, sequence_length=input_lengths, dtype=tf.float32)
    return enc_state


def _decoder(params, enc_state, enc_input, enc_input_lengths, helper):
    cells = [tf.contrib.rnn.GRUCell(params['hidden_size']) for _ in range(params['num_layers'])]
    mrnn = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        num_units=params['hidden_size'], memory=enc_input,
        memory_sequence_length=enc_input_lengths)
    attn_cell = tf.contrib.seq2seq.AttentionWrapper(
        mrnn, attention_mechanism, attention_layer_size=params['hidden_size'])
    output_layer = tf.layers.Dense(params['num_labels'])
    initial_state = attn_cell.zero_state(dtype=tf.float32, batch_size=params['batch_size'])
    initial_state = initial_state.clone(cell_state=enc_state)
    decoder = tf.contrib.seq2seq.BasicDecoder(attn_cell, helper, initial_state,output_layer=output_layer)
    dec_outputs = tf.contrib.seq2seq.dynamic_decode(
        decoder,
        output_time_major=False,
        impute_finished=True,
        maximum_iterations=params['max_target_seq_length'])

    return dec_outputs


class AOCR(tf.estimator.Estimator):
    def __init__(
            self,
            params=None,
            model_dir=None,
            config=None,
            warm_start_from=None
    ):
        def _model_fn(features, labels, mode, params, config):
            return _aocr_model_fn(
                features=features,
                labels=labels,
                mode=mode,
                params=params,
                config=config)

        super(AOCR, self).__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            config=config,
            params=params,
            warm_start_from=warm_start_from
        )


class IniInceptionHook(session_run_hook.SessionRunHook):
    def __init__(self, model_path):
        self._model_path = model_path
        self._ops = None

    def begin(self):
        if self._model_path is not None:
            inception_variables_dict = {
                var.op.name: var
                for var in slim.get_model_variables('InceptionV3')
            }
            self._init_fn_inception = slim.assign_from_checkpoint_fn(self._model_path, inception_variables_dict)

    def after_create_session(self, session, coord):
        if self._model_path is not None:
            logging.info('Do  Init Inception')
            self._init_fn_inception(session)

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return None

    def after_run(self, run_context, run_values):
        None
