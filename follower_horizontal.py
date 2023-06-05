import flbenchmark.datasets
import time
import sys
import json
import os
import glob
import threading
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import ast
import numpy as np
import pandas as pd
import fedlearner.common.fl_logging as logging
from fedlearner.fedavg import train_from_keras_model

tf.enable_eager_execution()

config = json.load(open(sys.argv[1], 'r'))

flbd = flbenchmark.datasets.FLBDatasets('~/flbenchmark.working/data')

val_dataset = None
if config['dataset'] == 'reddit':
    train_dataset, test_dataset, val_dataset = flbd.leafDatasets(config['dataset'])
elif config['dataset'] == 'femnist':
    train_dataset, test_dataset = flbd.leafDatasets(config['dataset'])
else:
    train_dataset, test_dataset = flbd.fateDatasets(config['dataset'])
train_data_base = os.path.expanduser('~/flbenchmark.working/csv_data/'+config['dataset']+'_train')
test_data_base = os.path.expanduser('~/flbenchmark.working/csv_data/'+config['dataset']+'_test')
val_data_base = os.path.expanduser('~/flbenchmark.working/csv_data/'+config['dataset']+'_val')
flbenchmark.datasets.convert_to_csv(train_dataset, out_dir=train_data_base)
if test_dataset is not None:
    flbenchmark.datasets.convert_to_csv(test_dataset, out_dir=test_data_base)
if val_dataset is not None:
    flbenchmark.datasets.convert_to_csv(val_dataset, out_dir=val_data_base)

if config['dataset'] == 'reddit':
    # Dataset Pre-processing
    def load_data(split, use_first_k=None):
        use_first_k = None
        with open('~/flbenchmark.working/csv_data/reddit_%s/_main.json'%split) as inf:
            meta_info = json.load(inf)
            parties = meta_info['parties']
        if use_first_k is not None:
            parties = parties[:use_first_k]

        all_data = {pid: [] for pid in parties}
        for pid in parties:
            df = pd.read_csv('~/flbenchmark.working/csv_data/reddit_%s/%s.csv'%(split, pid))
            for _, row in df.iterrows():
                cur_frame = ast.literal_eval(row['x0'])
                cur_x = [tok for sent in cur_frame for tok in sent if tok != '<PAD>']
                all_data[pid].append(cur_x)
        return all_data

    def text_to_seq(tokenizer, data, max_length, trunc_type):
        seq_data = {}
        for pid, one_data in data.items():
            seq_data[pid] = pad_sequences(tokenizer.texts_to_sequences(one_data), maxlen=max_length, truncating=trunc_type)
        return seq_data

    train_data = load_data('train')
    test_data = load_data('test')
    all_users = list(train_data.keys())
    client_num = len(all_users)

    # Build vocab
    vocab_size = 10000
    embedding_dim = 160
    hidden_dim = 512
    oov_tok = '<OOV>'
    max_length = 25
    trunc_type= 'post'
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

    all_train_data = [seq for one_data in train_data.values() for seq in one_data]
    tokenizer.fit_on_texts(all_train_data)
    oov_idx = tokenizer.word_index['<OOV>']
    train_data_seq = text_to_seq(tokenizer, train_data, max_length, trunc_type)
    test_data_seq = text_to_seq(tokenizer, test_data, max_length, trunc_type)
    assert set(train_data_seq.keys()) == set(test_data_seq.keys())
    all_test_seq = np.array([seq for one_seq in test_data_seq.values() for seq in one_seq])
else:
    x = {}
    y = {}
    x["train"] = []
    x["test"] = []
    y["train"] = []
    y["test"] = []

    client_num = 0
    for dir_path in ["train", "test"]:
        if dir_path == 'test' and config['dataset'] == 'vehicle_scale_horizontal':
            break
        for data_path in glob.glob(os.path.expanduser(f'~/flbenchmark.working/csv_data/{config["dataset"]}_{dir_path}/*.csv')):
            data = pd.read_csv(data_path, sep=',')
            if config['dataset'] == 'femnist':
                if config['model'] == 'lenet':
                    x[dir_path].append(np.array(data.iloc[:, 1:]).reshape(-1, 28, 28, 1).astype(np.float32))
                else:
                    x[dir_path].append(np.array(data.iloc[:, 1:]).astype(np.float32))
            elif config['dataset'] == 'student_horizontal':
                x[dir_path].append(np.array(pd.concat([data.iloc[:, 9:], data.iloc[:, 1:8]], axis=1)).astype(np.float32))
            else:
                x[dir_path].append(np.array(data.iloc[:, 2:]).astype(np.float32))
            if config['dataset'] == 'student_horizontal':
                y[dir_path].append(np.array(data.y).astype(np.float32))
            else:
                y[dir_path].append(np.array(data.y).astype(np.int32))
            if dir_path == 'train':
                client_num += 1
    if config['dataset'] == 'vehicle_scale_horizontal':
        x_test = np.concatenate(x["train"], axis=0)
        y_test = np.concatenate(y["train"], axis=0)
    else:
        x_test = np.concatenate(x["test"], axis=0)
        y_test = np.concatenate(y["test"], axis=0)

if config['dataset'] == 'reddit':
    num_class = 0
    input_len = 0
    inplanes = 0
    type = 'classification'
elif config['dataset'] == 'femnist':
    num_class = 62
    input_len = 28
    inplanes = 1
    type = 'classification'
elif config['dataset'] == 'breast_horizontal':
    num_class = 2
    input_len = 30
    inplanes = 0
    type = 'classification'
elif config['dataset'] == 'default_credit_horizontal':
    num_class = 2
    input_len = 23
    inplanes = 0
    type = 'classification'
elif config['dataset'] == 'give_credit_horizontal':
    num_class = 2
    input_len = 10
    inplanes = 0
    type = 'classification'
elif config['dataset'] == 'student_horizontal':
    num_class = 1
    input_len = 13
    inplanes = 0
    type = 'regression'
elif config['dataset'] == 'vehicle_scale_horizontal':
    num_class = 4
    input_len = 18
    inplanes = 0
    type = 'classification'
else:
    raise NotImplementedError('Dataset {} is not supported.'.format(config['dataset']))

class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0]], 'REFLECT')

def create_model(num_class, input_len, type):
    if type == 'classification':
        activation = 'softmax'
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metric = 'acc'
    elif type == 'regression':
        activation = 'linear'
        loss = tf.keras.losses.MeanSquaredError()
        metric = 'mse'
    if config['model'] == 'linear_regression':
        if config['dataset'] == 'femnist':
            input_len = inplanes * input_len * input_len
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(num_class, activation=activation, input_shape=(input_len, )),
        ])
        model.compile(optimizer=tf.keras.optimizers.SGD(config['training_param']['learning_rate'], **config['training_param']['optimizer_param']),
                    loss=loss,
                    metrics=[metric])
    elif config['model'] == 'logistic_regression':
        if config['dataset'] == 'femnist':
            input_len = inplanes * input_len * input_len
        if type == 'classification':
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(num_class, activation='sigmoid', input_shape=(input_len, )),
                tf.keras.layers.Softmax()
            ])
        elif type == 'regression':
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(num_class, activation='sigmoid', input_shape=(input_len, )),
            ])
        model.compile(optimizer=tf.keras.optimizers.SGD(config['training_param']['learning_rate'], **config['training_param']['optimizer_param']),
                    loss=loss,
                    metrics=[metric])
    elif config['model'].startswith('mlp_'):
        if config['dataset'] == 'femnist':
            input_len = inplanes * input_len * input_len
        sp = config['model'].split('_')
        if len(sp) == 2:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(sp[1], activation='relu', input_shape=(input_len, )),
                tf.keras.layers.Dense(num_class, activation=activation),
            ])
            model.compile(optimizer=tf.keras.optimizers.SGD(config['training_param']['learning_rate'], **config['training_param']['optimizer_param']),
                        loss=loss,
                        metrics=[metric])
        elif len(sp) == 3:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(sp[1], activation='relu', input_shape=(input_len, )),
                tf.keras.layers.Dense(sp[2], activation='relu'),
                tf.keras.layers.Dense(num_class, activation=activation),
            ])
            model.compile(optimizer=tf.keras.optimizers.SGD(config['training_param']['learning_rate'], **config['training_param']['optimizer_param']),
                        loss=loss,
                        metrics=[metric])
        elif len(sp) == 4:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(sp[1], activation='relu', input_shape=(input_len, )),
                tf.keras.layers.Dense(sp[2], activation='relu'),
                tf.keras.layers.Dense(sp[3], activation='relu'),
                tf.keras.layers.Dense(num_class, activation=activation),
            ])
            model.compile(optimizer=tf.keras.optimizers.SGD(config['training_param']['learning_rate'], **config['training_param']['optimizer_param']),
                        loss=loss,
                        metrics=[metric])
    elif config['model'] == 'lenet':
        if config['dataset'] != 'femnist':
            raise NotImplementedError('Dataset {} is not supported for {}.'.format(config['dataset'], config['model']))
        model = tf.keras.Sequential([
            ReflectionPadding2D(padding=(2, 2), input_shape=(input_len, input_len, inplanes)),
            tf.keras.layers.Conv2D(6, 5, data_format="channels_last"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(16, 5),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(120, activation='relu'),
            tf.keras.layers.Dense(84, activation='relu'),
            tf.keras.layers.Dense(num_class, activation='softmax'),
        ])
        model.compile(optimizer=tf.keras.optimizers.SGD(config['training_param']['learning_rate'], **config['training_param']['optimizer_param']),
                    loss=loss,
                    metrics=[metric])
    elif config['model'] == 'lstm':
        if config['dataset'] != 'reddit':
            raise NotImplementedError('Dataset {} is not supported for {}.'.format(config['dataset'], config['model']))
        tf.keras.backend.clear_session()
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length-1),
            tf.keras.layers.LSTM(hidden_dim, return_sequences=True),
            tf.keras.layers.Dense(vocab_size, activation='softmax'),
        ])
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metric = tf.keras.metrics.SparseCategoricalAccuracy()
        def loss_without_oov(y_true, y_pred):
            # Check: ignore PAD token in y_true
            y_true_flat = tf.reshape(y_true, [-1])
            y_pred_flat = tf.reshape(y_pred, [-1, vocab_size])
            indices = tf.where( tf.logical_and( tf.not_equal(y_true_flat, oov_idx) , tf.not_equal(y_true_flat, 0) ) )
            indices = tf.reshape(indices, [-1])
            y_true_tgt = tf.gather(y_true_flat, indices)
            y_pred_tgt = tf.gather(y_pred_flat, indices)
            return loss(y_true_tgt, y_pred_tgt)
        def metric_without_oov(y_true, y_pred):
            # Check: ignore OOV and PAD token in y_true
            y_true_flat = tf.reshape(y_true, [-1])
            y_pred_flat = tf.reshape(y_pred, [-1, vocab_size])
            indices = tf.where( tf.logical_and( tf.not_equal(y_true_flat, oov_idx) , tf.not_equal(y_true_flat, 0) ) )
            indices = tf.reshape(indices, [-1])
            y_true_tgt = tf.gather(y_true_flat, indices)
            y_pred_tgt = tf.gather(y_pred_flat, indices)
            return metric(y_true_tgt, y_pred_tgt)
        model.compile(optimizer=tf.keras.optimizers.SGD(config['training_param']['learning_rate'], **config['training_param']['optimizer_param']),
                  loss=loss_without_oov,
                  metrics=[metric_without_oov])
    else:
        raise NotImplementedError('Model {} is not supported.'.format(config['model']))
    return model

_fl_cluster = {
    "leader": {
        "name": "leader",
        "address": f"{sys.argv[2]}:30150"
    },
    "followers": []
}
for i in range(client_num-1):
    _fl_cluster["followers"].append({
    "name": "follower_"+str(i),
    "address": f"{sys.argv[2]}:"+str(30151+i)
})
model = create_model(num_class, input_len, type)
i = int(sys.argv[3])

if config['dataset'] == 'reddit':
    train_from_keras_model(model,
                       all_test_seq[:,:-1],
                       all_test_seq[:,1:],
                       train_data_seq[all_users[i+1]][:,:-1],
                       train_data_seq[all_users[i+1]][:,1:],
                       batch_size=config['training_param']['batch_size'],
                       epochs=config['training_param']['epochs'],
                       fl_name=f"follower_{i}",
                       fl_cluster=_fl_cluster,
                       steps_per_sync=config['training_param']['steps_per_sync'])
else:
    train_from_keras_model(model,
                       x_test,
                       y_test,
                       x["train"][i+1],
                       y["train"][i+1],
                       batch_size=config['training_param']['batch_size'],
                       epochs=config['training_param']['epochs'],
                       fl_name=f"follower_{i}",
                       fl_cluster=_fl_cluster,
                       steps_per_sync=config['training_param']['steps_per_sync'])
