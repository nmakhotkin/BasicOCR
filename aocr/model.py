from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from aocr.cnn import CNN
from aocr.seq2seq_model import Seq2SeqModel
import math
import tensorflow as tf
import logging


def _aocr_model_fn(features, labels, mode, params=None, config=None):
    forward_only = (mode != tf.estimator.ModeKeys.TRAIN)
    global_step = tf.train.get_or_create_global_step()
    max_resized_width = 1. * params['max_image_width'] / params['max_image_height'] * config['height']

    max_original_width = params['max_image_width']
    max_width = int(math.ceil(max_resized_width))

    encoder_size = int(math.ceil(1. * max_width / 4))
    decoder_size = params['max_prediction_length'] + 2
    buckets = [(encoder_size, decoder_size)]

    cnn_model = CNN(features, not forward_only)
    conv_output = cnn_model.tf_output()
    perm_conv_output = tf.transpose(conv_output, perm=[1, 0, 2])

    encoder_masks = []
    for i in xrange(params['encoder_size'] + 1):
        encoder_masks.append(
            tf.tile([[1.]], [params['batch_size'], 1])
        )

    decoder_inputs = []
    target_weights = []
    for i in xrange(decoder_size + 1):
        decoder_inputs.append(
            tf.tile([0], [params['batch_size']])
        )
        if i < decoder_size:
            target_weights.append(tf.tile([1.], [params['batch_size']]))
        else:
            target_weights.append(tf.tile([0.], [params['batch_size']]))
    attention_decoder_model = Seq2SeqModel(
        encoder_masks=encoder_masks,
        encoder_inputs_tensor=perm_conv_output,
        decoder_inputs=decoder_inputs,
        target_weights=target_weights,
        target_vocab_size=len(DataGen.CHARMAP),
        buckets=buckets,
        target_embedding_size=params['target_embedding_size'],
        attn_num_layers=params['attn_num_layers'],
        attn_num_hidden=params['attn_num_hidden'],
        forward_only=forward_only,
        use_gru=params['use_gru'])

    table = tf.contrib.lookup.MutableHashTable(
        key_dtype=tf.int64,
        value_dtype=tf.string,
        default_value="",
        checkpoint=True,
    )

    insert = table.insert(
        tf.constant(list(range(len(params['char_map']))), dtype=tf.int64),
        tf.constant(params['char_map']),
    )

    with tf.control_dependencies([insert]):
        num_feed = []
        prb_feed = []

        for line in xrange(len(attention_decoder_model.output)):
            guess = tf.argmax(attention_decoder_model.output[line], axis=1)
            proba = tf.reduce_max(
                tf.nn.softmax(attention_decoder_model.output[line]), axis=1)
            num_feed.append(guess)
            prb_feed.append(proba)

        # Join the predictions into a single output string.
        trans_output = tf.transpose(num_feed)
        trans_output = tf.map_fn(
            lambda m: tf.foldr(
                lambda a, x: tf.cond(
                    tf.equal(x, params['eos_id']),
                    lambda: '',
                    lambda: table.lookup(x) + a  # pylint: disable=undefined-variable
                ),
                m,
                initializer=''
            ),
            trans_output,
            dtype=tf.string
        )

        # Calculate the total probability of the output string.
        trans_outprb = tf.transpose(prb_feed)
        trans_outprb = tf.gather(trans_outprb, tf.range(tf.size(trans_output)))
        trans_outprb = tf.map_fn(
            lambda m: tf.foldr(
                lambda a, x: tf.multiply(tf.cast(x, tf.float64), a),
                m,
                initializer=tf.cast(1, tf.float64)
            ),
            trans_outprb,
            dtype=tf.float64
        )

        prediction = tf.cond(
            tf.equal(tf.shape(trans_output)[0], 1),
            lambda: trans_output[0],
            lambda: trans_output,
        )
        probability = tf.cond(
            tf.equal(tf.shape(trans_outprb)[0], 1),
            lambda: trans_outprb[0],
            lambda: trans_outprb,
        )

        prediction = tf.identity(prediction, name='prediction')
        probability = tf.identity(probability, name='probability')

        if forward_only:  # train
            updates = []
            summaries_by_bucket = []

            params = tf.trainable_variables()
            opt = tf.train.AdadeltaOptimizer(learning_rate=params['initial_learning_rate'])
            loss_op = attention_decoder_model.loss

            if params['reg_val'] > 0:
                reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                logging.info('Adding %s regularization losses', len(reg_losses))
                logging.debug('REGULARIZATION_LOSSES: %s', reg_losses)
                loss_op = params['reg_val'] * tf.reduce_sum(reg_losses) + loss_op

            gradients, params = list(zip(*opt.compute_gradients(loss_op, params)))
            if params['max_gradient_norm'] is not None:
                gradients, _ = tf.clip_by_global_norm(gradients, params['max_gradient_norm'])

            # Summaries for loss, variables, gradients, gradient norms and total gradient norm.
            summaries = [
                tf.summary.scalar("loss", loss_op),
                tf.summary.scalar("total_gradient_norm", tf.global_norm(gradients))
            ]
            all_summaries = tf.summary.merge(summaries)
            summaries_by_bucket.append(all_summaries)

            # update op - apply gradients
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                updates.append(opt.apply_gradients(list(zip(gradients, params)),global_step=global_step))

