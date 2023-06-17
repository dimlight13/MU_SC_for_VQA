
import argparse
import numpy as np
from keras.models import load_model
from models import Chan_Model, build_receiver_model, perfect_channel_estimation
from utils import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

parser = argparse.ArgumentParser(description='Multi-modal VQA with Semantic Communication')
parser.add_argument('--data_dir', type=str, default='data/')
parser.add_argument('--train_model_dir', type=str, default='./trained_model/')
parser.add_argument('--snr', type=int, default=10)
parser.add_argument('--channel_type', type=str, default='rayleigh', help="awgn or rician or rayleigh")
args = parser.parse_args()


if __name__ == "__main__":
    channel_type = args.channel_type
    train_model_dir = args.train_model_dir
    data_dir = args.data_dir
    snr = args.snr

    tx_model = load_model(train_model_dir + "tx_model.h5")
    rx_model = build_receiver_model()
    weights_load = np.load(train_model_dir + 'rx_model_weights.npy', allow_pickle=True)
    rx_model.set_weights(weights_load)
    chan_layer = Chan_Model('Channel')

    test_dataset = load_dataset(data_dir, 'val')

    dataloader = tf.data.Dataset.from_generator(lambda: test_dataset,
                    output_signature=(
                    tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(46), dtype=tf.int32),
                    tf.TensorSpec(shape=(28), dtype=tf.int32),))
    dataloader = dataloader.shuffle(buffer_size=1024, seed=42).batch(256)

    # Unpack train_dataset and val_dataset
    X_test, ques_test, ans_test = next(iter(dataloader))

    img_chan_output, txt_chan_output = tx_model.predict([X_test, ques_test], verbose=0)

    # Channel
    img_tx_chan = chan_layer(img_chan_output, snr, channel_type=channel_type)
    txt_tx_chan = chan_layer(txt_chan_output, snr, channel_type=channel_type)

    cls_output = rx_model.predict([img_chan_output, txt_chan_output], verbose=0)

    y_true = np.argmax(ans_test, axis=1)
    y_pred = np.argmax(cls_output, axis=1)
    acc = accuracy_score(y_true, y_pred)

    print("channel_type = {}, snr_db = {}, acc = {} %".format(channel_type, snr, acc * 100))

    import matplotlib.pyplot as plt

    conf_mat = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 6))
    df_mat = pd.DataFrame(conf_mat)
    sn.set(font_scale=0.7)
    sn.heatmap(df_mat, annot=True, annot_kws={"size": 8}, cmap='Blues')
    plt.savefig("results/SC_vqa_{}_{}_dB.png".format(channel_type, snr), dpi=300)
    plt.close()