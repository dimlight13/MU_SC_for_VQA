import tensorflow as tf
import os 
import numpy as np
import h5py
import pickle
from keras.utils import to_categorical
from keras.metrics import CategoricalAccuracy
from keras.losses import CategoricalCrossentropy
from tqdm import tqdm
import random
import collections

_cse = CategoricalCrossentropy()
train_acc_metric = CategoricalAccuracy()
val_acc_metric = CategoricalAccuracy()

def save_weights_process(tx_model, rx_model):
    tx_model.save("./trained_model/tx_model.h5")
    np.save('./trained_model/rx_model_weights.npy', rx_model.get_weights())


class ClevrDataset(tf.keras.utils.Sequence):
    def __init__(self, data_dir, split='train'):
        with open(os.path.join(data_dir, '{}.pkl'.format(split)), 'rb') as f:
            self.data = pickle.load(f)

        self.data_dict = collections.defaultdict(list)
        for item in self.data:
            _, _, answer, _ = item
            self.data_dict[answer].append(item)

        self.shuffle()
        self.img = h5py.File(os.path.join(data_dir, 'image_{}.h5'.format(split)), 'r')
        self.max_question_length = 46  # max_len + 1
        self.num_classes = 28  # max_len + 1

    def shuffle(self):
        self.data = []
        for _, items in self.data_dict.items():
            random.shuffle(items)
            self.data += items
        
        self.data = self.data[:10000]

    def get_max_answer(self):
        max_answer = 0
        for index in range(len(self.data)):
            _, _, answer, _ = self.data[index]
            if answer > max_answer:
                max_answer = answer
        return max_answer + 1

    def get_max_question_length(self):
        max_len = 0
        for index in range(len(self.data)):
            _, question, _, _ = self.data[index]
            if len(question) > max_len:
                max_len = len(question)
        return max_len + 1

    def pad_question(self, question):
        if len(question) >= self.max_question_length:
            question_padded = question[:self.max_question_length]
        else:
            question_padded = question + [0] * (self.max_question_length - len(question))
        return question_padded

    def __getitem__(self, index):
        imgfile, question, answer, family = self.data[index]
        id = int(imgfile.rsplit('_', 1)[1][:-4])
        img = tf.convert_to_tensor(self.img[f'image_{id}'][()] / 255., dtype=tf.float32)

        # Padding
        question_padded = self.pad_question(question)
        return img, question_padded, to_categorical(answer, num_classes=self.num_classes)

    def __len__(self):
        return len(self.data)
    

def load_dataset(data_dir='data/', name='train'):
    dataset = ClevrDataset(data_dir=data_dir, split=name)
    return dataset

@tf.function
def train_step(inputs, labels, sc_model, opt):
    with tf.GradientTape() as tape:
        predictions = sc_model(inputs, training=True)
        loss = _cse(labels, predictions)

    gradients = tape.gradient(loss, sc_model.trainable_variables)
    opt.apply_gradients(zip(gradients, sc_model.trainable_variables))
    train_acc_metric.update_state(labels, predictions)
    return loss