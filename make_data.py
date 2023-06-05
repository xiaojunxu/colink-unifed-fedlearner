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

import os
import sys
import json
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.core.example.example_pb2 import Example
from tensorflow.core.example.feature_pb2 import FloatList, Features, Feature, \
                                                Int64List, BytesList
import flbenchmark.datasets

config = json.load(open(sys.argv[1], 'r'))

if config['dataset'] != 'give_credit_vertical' and sys.argv[2] == 'test':
    sys.exit()

flbd = flbenchmark.datasets.FLBDatasets('~/flbenchmark.working/data')

train_dataset, test_dataset = flbd.fateDatasets(config['dataset'])
train_data_base = os.path.abspath('~/flbenchmark.working/csv_data/'+config['dataset']+'_train')
test_data_base = os.path.abspath('~/flbenchmark.working/csv_data/'+config['dataset']+'_test')
flbenchmark.datasets.convert_to_csv(train_dataset, out_dir=train_data_base)
if test_dataset is not None:
    flbenchmark.datasets.convert_to_csv(test_dataset, out_dir=test_data_base)

current_dir = os.path.dirname(__file__)
shutil.rmtree(os.path.join(current_dir, 'data'), ignore_errors=True)
os.makedirs(os.path.join(current_dir, 'data/leader'))
os.makedirs(os.path.join(current_dir, 'data/follower'))

if config['dataset'] == 'breast_vertical':
    data_l = pd.read_csv('~/flbenchmark.working/csv_data/breast_vertical_train/breast_hetero_guest.csv', sep=',')
    xl = np.array(data_l.iloc[:, 2:]).astype(np.float32)
    y = np.array(data_l.y).astype(np.int64)
    data_f = pd.read_csv('~/flbenchmark.working/csv_data/breast_vertical_train/breast_hetero_host.csv', sep=',')
    xf = np.array(data_f.iloc[:, 1:]).astype(np.float32)
elif config['dataset'] == 'default_credit_vertical':
    data_l = pd.read_csv('~/flbenchmark.working/csv_data/default_credit_vertical_train/default_credit_hetero_guest.csv', sep=',')
    xl = np.array(data_l.iloc[:, 2:]).astype(np.float32)
    y = np.array(data_l.y).astype(np.int64)
    data_f = pd.read_csv('~/flbenchmark.working/csv_data/default_credit_vertical_train/default_credit_hetero_host.csv', sep=',')
    xf = np.array(data_f.iloc[:, 1:]).astype(np.float32)
elif config['dataset'] == 'dvisit_vertical':
    data_l = pd.read_csv('~/flbenchmark.working/csv_data/dvisit_vertical_train/dvisit_hetero_guest.csv', sep=',')
    xl = np.array(data_l.iloc[:, 2:]).astype(np.float32)
    y = np.array(data_l.y).astype(np.float32)
    data_f = pd.read_csv('~/flbenchmark.working/csv_data/dvisit_vertical_train/dvisit_hetero_host.csv', sep=',')
    xf = np.array(data_f.iloc[:, 1:]).astype(np.float32)
elif config['dataset'] == 'give_credit_vertical':
    data_l = pd.read_csv(f'~/flbenchmark.working/csv_data/give_credit_vertical_{sys.argv[2]}/give_credit_hetero_guest.csv', sep=',')
    xl = np.array(data_l.iloc[:, 2:]).astype(np.float32)
    y = np.array(data_l.y).astype(np.int64)
    data_f = pd.read_csv(f'~/flbenchmark.working/csv_data/give_credit_vertical_{sys.argv[2]}/give_credit_hetero_host.csv', sep=',')
    xf = np.array(data_f.iloc[:120000, 1:]).astype(np.float32)
elif config['dataset'] == 'motor_vertical':
    data_l = pd.read_csv('~/flbenchmark.working/csv_data/motor_vertical_train/motor_hetero_guest.csv', sep=',')
    xl = np.array(data_l.iloc[:, 2:]).astype(np.float32)
    y = np.array(data_l.y).astype(np.float32)
    data_f = pd.read_csv('~/flbenchmark.working/csv_data/motor_vertical_train/motor_hetero_host.csv', sep=',')
    xf = np.array(data_f.iloc[:, 1:]).astype(np.float32)
elif config['dataset'] == 'student_vertical':
    data_l = pd.read_csv('~/flbenchmark.working/csv_data/student_vertical_train/student_hetero_guest.csv', sep=',')
    xl = np.array(data_l.iloc[:, 2:]).astype(np.float32)
    y = np.array(data_l.y).astype(np.float32)
    data_f = pd.read_csv('~/flbenchmark.working/csv_data/student_vertical_train/student_hetero_host.csv', sep=',')
    xf = np.array(data_f.iloc[:, 1:]).astype(np.float32)
elif config['dataset'] == 'vehicle_scale_vertical':
    data_l = pd.read_csv('~/flbenchmark.working/csv_data/vehicle_scale_vertical_train/vehicle_scale_hetero_guest.csv', sep=',')
    xl = np.array(data_l.iloc[:, 2:]).astype(np.float32)
    y = np.array(data_l.y).astype(np.int64)
    data_f = pd.read_csv('~/flbenchmark.working/csv_data/vehicle_scale_vertical_train/vehicle_scale_hetero_host.csv', sep=',')
    xf = np.array(data_f.iloc[:, 1:]).astype(np.float32)
else:
    raise NotImplementedError('Dataset {} is not supported.'.format(config['dataset']))

assert xl.shape[0] == xf.shape[0]

def write_tfrecord_data(filename, data, header, dtypes):
    fout = tf.io.TFRecordWriter(filename)
    for i in range(data.shape[0]):
        example = tf.train.Example()
        for h, d, x in zip(header, dtypes, data[i]):
            if d == np.int64:
                example.features.feature[h].int64_list.value.append(x)
            else:
                example.features.feature[h].float_list.value.append(x)
        fout.write(example.SerializeToString())

def write_data(output_type, filename, X, y, role):
    #if role == 'leader':
    if 'leader' in role:
        data = np.concatenate((X, y), axis=1)
        N = data.shape[1] - 1
        header = ['f%05d'%i for i in range(N)] + ['label']
        dtypes = [np.float]*(N) + [np.int64]
    else: # role == 'follower':
        data = X
        N = data.shape[1]
        header = ['f%05d'%i for i in range(N)]
        dtypes = [np.float]*(N)

    data = np.asarray([tuple(i) for i in data], dtype=list(zip(header, dtypes)))
    if output_type == 'tfrecord':
        write_tfrecord_data(filename, data, header, dtypes)
    else:
        np.savetxt(
            filename,
            data,
            delimiter=',',
            header=','.join(header),
            fmt=['%d' if i == np.int64 else '%f' for i in dtypes],
            comments='')

if config['model'] == 'gbdt':
    output_type = 'tfrecord'
    write_data(
        output_type,
        'data/leader/leader_train.%s'%output_type,
        xl, y.reshape(-1, 1),
        'leader')
    write_data(
        output_type,
        'data/follower/follower_train.%s'%output_type,
        xf, y.reshape(-1, 1),
        'follower')
    if config['dataset'] == 'give_credit_vertical':
        data_l = pd.read_csv(f'~/flbenchmark.working/csv_data/give_credit_vertical_test/give_credit_hetero_guest.csv', sep=',')
        xlt = np.array(data_l.iloc[:, 2:]).astype(np.float32)
        yt = np.array(data_l.y).astype(np.int64)
        data_f = pd.read_csv(f'~/flbenchmark.working/csv_data/give_credit_vertical_test/give_credit_hetero_host.csv', sep=',')
        xft = np.array(data_f.iloc[:120000, 1:]).astype(np.float32)
        write_data(
            output_type,
            'data/leader/leader_test.%s'%output_type,
            xlt, yt.reshape(-1, 1),
            'leader')
        write_data(
            output_type,
            'data/follower/follower_test.%s'%output_type,
            xft, yt.reshape(-1, 1),
            'follower')
    else:
        write_data(
            output_type,
            'data/leader/leader_test.%s'%output_type,
            xl, y.reshape(-1, 1),
            'leader')
        write_data(
            output_type,
            'data/follower/follower_test.%s'%output_type,
            xf, y.reshape(-1, 1),
            'follower')
else:
    N = 1
    chunk_size = xl.shape[0] // N

    for i in range(N):
        filename_l = os.path.join(current_dir, 'data/leader/%02d.tfrecord'%i)
        filename_f = os.path.join(current_dir, 'data/follower/%02d.tfrecord'%i)
        fl = tf.io.TFRecordWriter(filename_l)
        ff = tf.io.TFRecordWriter(filename_f)

        for j in range(chunk_size):
            idx = i*chunk_size + j
            features_l = {}
            features_l['example_id'] = Feature(
                bytes_list=BytesList(value=[str(idx).encode('utf-8')]))
            if config['dataset'] == 'dvisit_vertical' or config['dataset'] == 'motor_vertical' or config['dataset'] == 'student_vertical':
                features_l['y'] = Feature(float_list=FloatList(value=[y[idx]]))
            else:
                features_l['y'] = Feature(int64_list=Int64List(value=[y[idx]]))
            features_l['x'] = Feature(float_list=FloatList(value=list(xl[idx])))
            fl.write(
                Example(features=Features(feature=features_l)).SerializeToString())

            features_f = {}
            features_f['example_id'] = Feature(
                bytes_list=BytesList(value=[str(idx).encode('utf-8')]))
            features_f['x'] = Feature(float_list=FloatList(value=list(xf[idx])))
            ff.write(
                Example(features=Features(feature=features_f)).SerializeToString())

        fl.close()
        ff.close()
