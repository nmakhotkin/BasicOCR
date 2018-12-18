from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from aocr.cnn import CNN
import tensorflow as tf
import logging
import glob
import numpy as np

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
    datasets_files = []
    for tf_file in glob.iglob(params['data_set'] + '/*.tfrecord'):
        datasets_files.append(tf_file)

    def _input_fn():
        ds = tf.data.TFRecordDataset(datasets_files, buffer_size=256 * 1024 * 1024)

        def _emdeding(label):
            label = str(label, encoding='UTF-8')
            labels = []
            for c in label.lower():
                v = inv_charset.get(c, -1)
                if v > 0:
                    labels.append(v)
            labels.append(1)
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
            img = tf.image.decode_png(res['image/encoded'], channels=3)
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


def _aocr_model_fn(features, labels, mode, params=None, config=None):
    logging.info('Labels {}'.format(labels))
    start_tokens = tf.zeros([params['batch_size']], dtype=tf.int64)
    train_output0 = tf.concat([tf.expand_dims(start_tokens, 1), labels], 1)
    output_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(train_output0, 1)), 1)
    train_output = tf.reshape(train_output0,[params['batch_size'],-1,1])
    logging.info('train_output {}'.format(train_output))
    logging.info('output_lengths {}'.format(output_lengths))
    forward_only = (mode != tf.estimator.ModeKeys.TRAIN)
    global_step = tf.train.get_or_create_global_step()
    cnn_model = CNN(features, not forward_only)
    conv_output = cnn_model.tf_output()
    logging.info('Conv output {}'.format(conv_output))
    helper = tf.contrib.seq2seq.TrainingHelper(tf.cast(train_output,dtype=tf.float32), output_lengths)
    input_lengths = tf.zeros((params['batch_size']), dtype=tf.int64) + int(params['max_width'] / 4)
    logging.info('input_lengths {}'.format(input_lengths))
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        num_units=params['hidden_size'], memory=conv_output,
        memory_sequence_length=input_lengths)
    cell = tf.contrib.rnn.GRUCell(num_units=params['hidden_size'])
    attn_cell = tf.contrib.seq2seq.AttentionWrapper(
        cell, attention_mechanism, attention_layer_size=params['hidden_size'] / 2)
    out_cell = tf.contrib.rnn.OutputProjectionWrapper(
        attn_cell, params['num_labels']
    )
    decoder = tf.contrib.seq2seq.BasicDecoder(
        cell=out_cell, helper=helper,
        initial_state=out_cell.zero_state(
            dtype=tf.float32, batch_size=params['batch_size']))
    outputs = tf.contrib.seq2seq.dynamic_decode(
        decoder=decoder, output_time_major=False,
        impute_finished=True, maximum_iterations=params['max_target_seq_length']
    )
    train_outputs = outputs[0]
    logging.info('train_outputs.rnn_output: {}'.format(train_outputs.rnn_output))
    weights = tf.to_float(tf.not_equal(train_output0[:, :-1], 1))
    logging.info('weights: {}'.format(weights))
    loss = tf.contrib.seq2seq.sequence_loss(
        train_outputs.rnn_output, labels, weights=weights)

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

    return tf.estimator.EstimatorSpec(
        mode=mode,
        eval_metric_ops=None,
        predictions=None,
        loss=loss,
        training_hooks=None,
        export_outputs=None,
        train_op=train_op)


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
