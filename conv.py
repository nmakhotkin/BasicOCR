import argparse
import configparser
import json
import logging
import os

from mlboardclient.api import client
import tensorflow as tf

import aocr.model as aocr

mlboard = client.Client()
const = tf.saved_model.signature_constants


def parse_args():
    conf_parser = argparse.ArgumentParser(
        add_help=False
    )
    conf_parser.add_argument(
        '--checkpoint_dir',
        default=os.environ.get('TRAINING_DIR', 'training') + '/' + os.environ.get('BUILD_ID', '1'),
        help='Directory to save checkpoints and logs',
    )
    args, remaining_argv = conf_parser.parse_known_args()
    parser = argparse.ArgumentParser(
        parents=[conf_parser],
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    checkpoint_dir = args.checkpoint_dir
    logging.getLogger().setLevel('INFO')
    tf.logging.set_verbosity(tf.logging.INFO)
    parser.add_argument(
        '--limit_train',
        type=int,
        default=-1,
        help='Limit number files for train. For testing.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='Batch size.',
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=2e-4,
        help='Recommended learning_rate is 2e-4',
    )
    parser.add_argument(
        '--num_layers',
        type=int,
        default=1,
        help='Number RNN layers.',
    )
    parser.add_argument(
        '--rnn_type',
        default='BasicLSTM',
        choices=['BasicLSTM', 'CudnnLSTM', 'CudnnCompatibleLSTM'],
        help='Witch LSTM cell use for RNN',
    )
    parser.add_argument(
        '--max_target_seq_length',
        type=int,
        default=80,
        help='Maximum number of letters we allow in a single training (or test) example output',
    )
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=16,
        help='LSTM hidden size.',
    )
    parser.add_argument(
        '--lstm_direction_type',
        choices=['unidirectional', 'bidirectional'],
        default='unidirectional',
        help='Set LSTM direction type',
    )
    parser.add_argument(
        '--grad_clip',
        type=float,
        default=1,
        help='Norm for gradients clipping.',
    )
    parser.add_argument(
        '--output_keep_prob',
        type=float,
        default=0,
        help='Dropout value during training for output layer.',
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=1,
        help='Epoch to trian',
    )
    parser.add_argument(
        '--save_summary_steps',
        type=int,
        default=100,
        help="Log summary every 'save_summary_steps' steps",
    )
    parser.add_argument(
        '--save_checkpoints_secs',
        type=int,
        default=600,
        help="Save checkpoints every 'save_checkpoints_secs' secs.",
    )
    parser.add_argument(
        '--save_checkpoints_steps',
        type=int,
        default=1000,
        help="Save checkpoints every 'save_checkpoints_steps' steps",
    )
    parser.add_argument(
        '--max_width',
        type=int,
        default=150,
        help="Max image width",
    )
    parser.add_argument(
        '--keep_checkpoint_max',
        type=int,
        default=5,
        help='The maximum number of recent checkpoint files to keep.',
    )
    parser.add_argument(
        '--log_step_count_steps',
        type=int,
        default=100,
        help='The frequency, in number of global steps, that the global step/sec and the loss will be logged during training.',
    )
    parser.add_argument(
        '--data_set',
        default=None,
        help='Location of training files or evaluation files',
    )
    parser.add_argument(
        '--data_set_type',
        default='synth-crop',
        help='Dataset type',
    )
    parser.add_argument(
        '--conv',
        default='cnn',
        help='Encoder',
    )
    parser.add_argument(
        '--inception_checkpoint',
        default=None,
        help='inception_checkpoint',
    )

    parser.add_argument(
        '--warm_start_from',
        type=str,
        default=None,
        help='Warm start',
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.set_defaults(worker=False)
    group.set_defaults(evaluator=False)
    group.set_defaults(test=False)
    group.add_argument('--worker', dest='worker', action='store_true',
                       help='Start in Worker(training) mode.')
    group.add_argument('--evaluator', dest='evaluator', action='store_true',
                       help='Start in evaluation mode')
    group.add_argument('--test', dest='test', action='store_true',
                       help='Test mode')
    group.add_argument('--export', dest='export', action='store_true',
                       help='Export model')
    p_file = os.path.join(checkpoint_dir, 'parameters.ini')
    if tf.gfile.Exists(p_file):
        parameters = configparser.ConfigParser(allow_no_value=True)
        parameters.read(p_file)
        parser.set_defaults(**dict(parameters.items("PARAMETERS", raw=True)))
    args = parser.parse_args(remaining_argv)
    print('\n*************************\n')
    print(args)
    print('\n*************************\n')
    return checkpoint_dir, args


def _im2letter_model_fn(features, labels, mode, params=None, config=None):
    if isinstance(features, dict):
        features = features['images']

    batch_size = params['batch_size']
    features = tf.reshape(features, shape=(batch_size, -1, 3))
    labels = tf.reshape(labels, [batch_size, params['max_target_seq_length']])
    outputs = tf.layers.conv1d(
        features, filters=250, kernel_size=[32], strides=2, padding='SAME', activation=tf.nn.relu
    )
    outputs = tf.layers.conv1d(
        outputs, filters=250, kernel_size=[7], strides=1, padding='SAME', activation=tf.nn.relu
    )
    outputs = tf.layers.conv1d(
        outputs, filters=250, kernel_size=[7], strides=1, padding='SAME', activation=tf.nn.relu
    )
    outputs = tf.layers.conv1d(
        outputs, filters=250, kernel_size=[7], strides=1, padding='SAME', activation=tf.nn.relu
    )
    outputs = tf.layers.conv1d(
        outputs, filters=250, kernel_size=[7], strides=1, padding='SAME', activation=tf.nn.relu
    )
    outputs = tf.layers.conv1d(
        outputs, filters=250, kernel_size=[7], strides=1, padding='SAME', activation=tf.nn.relu
    )
    outputs = tf.layers.conv1d(
        outputs, filters=250, kernel_size=[7], strides=1, padding='SAME', activation=tf.nn.relu
    )
    outputs = tf.layers.conv1d(
        outputs, filters=250, kernel_size=[7], strides=1, padding='SAME', activation=tf.nn.relu
    )
    outputs = tf.layers.conv1d(
        outputs, filters=2000, kernel_size=[32], strides=1, padding='SAME', activation=tf.nn.relu
    )
    outputs = tf.layers.conv1d(
        outputs, filters=2000, kernel_size=[1], strides=1, padding='SAME', activation=tf.nn.relu
    )
    y_pred = tf.layers.conv1d(
        outputs, filters=params['num_labels'], kernel_size=[1], strides=1, padding='SAME'
    )

    hooks = []
    log_probs = tf.nn.log_softmax(y_pred, axis=1)
    if mode == tf.estimator.ModeKeys.TRAIN:
        predictions = None
        export_outputs = None
        opt = tf.train.AdamOptimizer(params['learning_rate'])

        # __import__('ipdb').set_trace()

        # log_probs = tf.transpose(log_probs, [1, 2])
        # log_probs = tf.transpose(log_probs, [0, 1])

        labels = tf.cast(labels, tf.int32)
        # lengths = tf.foldl(lambda x, y: x, labels)
        lengths = tf.map_fn(lambda x: x.shape[0], labels)

        zero = tf.constant(0, dtype=tf.int32)
        where = tf.not_equal(labels, zero)
        indices = tf.where(where)
        values = tf.gather_nd(labels, indices)
        sparse = tf.SparseTensor(indices, values, labels.shape)
        # loss = loss_ops.mean_squared_error(log_probs, labels)

        # CTC_Loss expects log_probs input shape
        # (batch_size, max_time, num_classes)
        ctc_loss = tf.nn.ctc_loss(
            sparse,
            log_probs,
            sequence_length=lengths,
            time_major=False,
            preprocess_collapse_repeated=False,
            ctc_merge_repeated=False,
        )
        loss = tf.reduce_mean(ctc_loss)
        train_op = opt.minimize(loss, global_step=tf.train.get_or_create_global_step())
    else:
        train_op = None
        loss = None
        predictions = tf.nn.top_k(outputs, k=1)
        sig_def = const.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        export_outputs = {
            sig_def: tf.estimator.export.PredictOutput(predictions)
        }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        eval_metric_ops=None,
        predictions=predictions,
        loss=loss,
        training_hooks=hooks,
        export_outputs=export_outputs,
        train_op=train_op
    )


class Image2Letter(tf.estimator.Estimator):
    def __init__(
            self,
            params=None,
            model_dir=None,
            config=None,
            warm_start_from=None
    ):
        def _model_fn(features, labels, mode, params, config):
            return _im2letter_model_fn(
                features=features,
                labels=labels,
                mode=mode,
                params=params,
                config=config
            )

        super(Image2Letter, self).__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            config=config,
            params=params,
            warm_start_from=warm_start_from
        )


def export(checkpoint_dir, params):
    conf = tf.estimator.RunConfig(
        model_dir=checkpoint_dir,
    )
    feature_placeholders = {
        'images': tf.placeholder(tf.float32, [params['batch_size'], 32, params['max_width'], 3], name='images'),
    }
    receiver = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_placeholders,
                                                                       default_batch_size=params['batch_size'])
    net = Image2Letter(
        params=params,
        model_dir=checkpoint_dir,
        config=conf,
    )
    export_path = net.export_savedmodel(
        checkpoint_dir,
        receiver,
    )
    export_path = export_path.decode("utf-8")
    client.update_task_info({'model_path': export_path})


def train(mode, checkpoint_dir, params):
    logging.info("start build model")
    logging.info("TF: {}".format(tf.__version__))

    save_summary_steps = params['save_summary_steps']
    save_checkpoints_secs = params['save_checkpoints_secs'] if params['save_checkpoints_steps'] is None else None
    save_checkpoints_steps = params['save_checkpoints_steps']

    conf = tf.estimator.RunConfig(
        model_dir=checkpoint_dir,
        save_summary_steps=save_summary_steps,
        save_checkpoints_secs=save_checkpoints_secs,
        save_checkpoints_steps=save_checkpoints_steps,
        keep_checkpoint_max=params['keep_checkpoint_max'],
        log_step_count_steps=params['log_step_count_steps'],
    )

    net = Image2Letter(
        params=params,
        model_dir=checkpoint_dir,
        config=conf,
        warm_start_from=params['warm_start_from']
    )
    logging.info("Start %s mode", mode)

    # params['max_target_seq_length'] = params['num_labels']

    if mode == 'train':
        input_fn = aocr.tf_input_fn(params, True)
        net.train(input_fn=input_fn)
    elif mode == 'eval':
        # train_fn = crnn.null_dataset()
        # train_spec = tf.estimator.TrainSpec(input_fn=train_fn)
        # eval_fn = crnn.eval_fn()
        # eval_spec = tf.estimator.EvalSpec(input_fn=eval_fn, steps=1, start_delay_secs=10, throttle_secs=10)
        # tf.estimator.train_and_evaluate(net, train_spec, eval_spec)
        logging.info("Not implemented")
    else:
        logging.info("Not implemented")


def main():
    checkpoint_dir, args = parse_args()
    logging.info('------------------')
    logging.info('TF VERSION: {}'.format(tf.__version__))
    logging.info('ARGS: {}'.format(args))
    logging.info('------------------')
    if args.worker:
        mode = 'train'
    elif args.test:
        mode = 'test'
    else:
        mode = 'eval'
        cluster = {'chief': ['fake_worker1:2222'],
                   'ps': ['fake_ps:2222'],
                   'worker': ['fake_worker2:2222']}
        os.environ['TF_CONFIG'] = json.dumps(
            {
                'cluster': cluster,
                'task': {'type': 'evaluator', 'index': 0}
            })

    _, charset = aocr.read_charset()
    logging.info("Charset: {}".format(charset))
    logging.info("NumClasses: {}".format(len(charset)))
    params = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'save_summary_steps': args.save_summary_steps,
        'save_checkpoints_steps': args.save_checkpoints_steps,
        'save_checkpoints_secs': args.save_checkpoints_secs,
        'keep_checkpoint_max': args.keep_checkpoint_max,
        'log_step_count_steps': args.log_step_count_steps,
        'data_set': args.data_set,
        'epoch': args.epoch,
        'limit_train': args.limit_train,
        'max_target_seq_length': args.max_target_seq_length,
        'num_labels': len(charset),
        'rnn_type': args.rnn_type,
        'beam_search_decoder': args.export,
        'grad_clip': args.grad_clip,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'output_keep_prob': args.output_keep_prob,
        'lstm_direction_type': args.lstm_direction_type,
        'charset': charset,
        'data_set_type': args.data_set_type,
        'max_width': args.max_width,
        'conv': args.conv,
        'warm_start_from': args.warm_start_from,
        'inception_checkpoint': args.inception_checkpoint,
        'normalize_length': True,
        'last_blank_label': True,
    }

    if not tf.gfile.Exists(checkpoint_dir):
        tf.gfile.MakeDirs(checkpoint_dir)

    if args.export:
        export(checkpoint_dir, params)
        return
    train(mode, checkpoint_dir, params)


if __name__ == '__main__':
    main()
