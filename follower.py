# Copyright 2020 The FedLearner Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding: utf-8
# pylint: disable=no-else-return, inconsistent-return-statements

import sys
import json
import logging
import tensorflow.compat.v1 as tf
import fedlearner.trainer as flt


ROLE = 'follower'

config = json.load(open('config.json', 'r'))
parser = flt.trainer_worker.create_argument_parser()
parser.add_argument('--batch-size', type=int, default=config['training_param']['batch_size'],
                    help='Training batch size.')
args = parser.parse_args()

if config['dataset'] == 'breast_vertical':
    input_len = 20
elif config['dataset'] == 'default_credit_vertical':
    input_len = 10
elif config['dataset'] == 'dvisit_vertical':
    input_len = 9
elif config['dataset'] == 'give_credit_vertical':
    input_len = 5
elif config['dataset'] == 'motor_vertical':
    input_len = 7
elif config['dataset'] == 'student_vertical':
    input_len = 7
elif config['dataset'] == 'vehicle_scale_vertical':
    input_len = 9
else:
    raise NotImplementedError('Dataset {} is not supported.'.format(config['dataset']))

if config['model'].startswith('mlp_'):
    sp = config['model'].split('_')
    if len(sp) != 2:
        raise NotImplementedError('Model {} is not supported.'.format(config['model']))
    mid = int(sp[1])
    if mid % 2:
        half = int((mid + 1) / 2)
    else:
        half = int(mid / 2)
else:
    raise NotImplementedError('Model {} is not supported.'.format(config['model']))

def input_fn(bridge, trainer_master):
    dataset = flt.data.DataBlockLoader(args.batch_size, ROLE,
        bridge, trainer_master).make_dataset()

    def parse_fn(example):
        feature_map = dict()
        feature_map["example_id"] = tf.FixedLenFeature([], tf.string)
        feature_map["x"] = tf.FixedLenFeature([input_len], tf.float32)
        features = tf.parse_example(example, features=feature_map)
        return features, dict(y=tf.constant(0))

    dataset = dataset.map(map_func=parse_fn,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

def serving_input_receiver_fn():
    feature_map = {
        "example_id": tf.FixedLenFeature([], tf.string),
        "x": tf.FixedLenFeature([input_len], tf.float32),
    }
    record_batch = tf.placeholder(dtype=tf.string, name='examples')
    features = tf.parse_example(record_batch, features=feature_map)
    return tf.estimator.export.ServingInputReceiver(features,
                                                    {'examples': record_batch})


def model_fn(model, features, labels, mode):
    x = features['x']

    w1f = tf.get_variable('w1f',
                          shape=[input_len, half],
                          dtype=tf.float32,
                          initializer=tf.random_uniform_initializer(
                              -0.01, 0.01))
    b1f = tf.get_variable('b1f',
                          shape=[half],
                          dtype=tf.float32,
                          initializer=tf.zeros_initializer())

    act1_f = tf.nn.relu(tf.nn.bias_add(tf.matmul(x, w1f), b1f))

    if mode == tf.estimator.ModeKeys.TRAIN:
        gact1_f = model.send('act1_f', act1_f, require_grad=True)
        optimizer = tf.train.MomentumOptimizer(config['training_param']['learning_rate'], **config['training_param']['optimizer_param'])
        train_op = model.minimize(
            optimizer,
            act1_f,
            grad_loss=gact1_f,
            global_step=tf.train.get_or_create_global_step())
        return model.make_spec(mode,
                               loss=tf.math.reduce_mean(act1_f),
                               train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        model.send('act1_f', act1_f, require_grad=False)
        fake_loss = tf.reduce_mean(act1_f)
        return model.make_spec(mode=mode, loss=fake_loss)

    # mode == tf.estimator.ModeKeys.PREDICT:
    return model.make_spec(mode=mode, predictions={'act1_f': act1_f})


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = flt.trainer_worker.create_argument_parser()
    flt.trainer_worker.train(
        ROLE, parser.parse_args(), input_fn,
        model_fn, serving_input_receiver_fn)
