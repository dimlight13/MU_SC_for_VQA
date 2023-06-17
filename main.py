import argparse
from keras.optimizers import Adam
from models import build_transmitter_model, build_receiver_model
from keras import Model
from utils import *

parser = argparse.ArgumentParser(description='Multi-modal VQA with Semantic Communication')
parser.add_argument('--data_dir', type=str, default='data/', required=True)
parser.add_argument('--batch_size', type=int, default=32, required=True)
parser.add_argument('--num_epochs', type=int, default=10, required=True)
parser.add_argument('--learning_rate', type=float, default=1e-4)
args = parser.parse_args()


if __name__ == "__main__":
    opt = Adam(learning_rate=args.learning_rate)
    batch_size = args.batch_size
    n_epochs = args.num_epochs
    tx_model = build_transmitter_model()
    rx_model = build_receiver_model()

    data_dir = args.data_dir
    tx_output = tx_model.output

    cls_output = rx_model(tx_output)
    sc_model = Model(inputs=tx_model.input, outputs=cls_output, name="sc_model")
    sc_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])

    train_dataset = load_dataset(data_dir=data_dir, name='train')
    val_dataset = load_dataset(data_dir=data_dir, name='val')

    train_dataloader = tf.data.Dataset.from_generator(lambda: train_dataset,
                    output_signature=(
                    tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(46), dtype=tf.int32),
                    tf.TensorSpec(shape=(28), dtype=tf.int32),))
    train_dataloader = train_dataloader.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_dataloader = tf.data.Dataset.from_generator(lambda: val_dataset,
                    output_signature=(
                    tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(46), dtype=tf.int32),
                    tf.TensorSpec(shape=(28), dtype=tf.int32),))
    val_dataloader = val_dataloader.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    sc_model.fit(train_dataloader.map(lambda x, y, z: ((x, y), z)), \
                validation_data=(val_dataloader.map(lambda x, y, z: ((x, y), z))), epochs=n_epochs)

    sc_model_weights = sc_model.get_weights()
    tx_model_weights = sc_model_weights[:len(tx_model.weights)]
    rx_model_weights = sc_model_weights[len(tx_model.weights):]

    tx_model.set_weights(tx_model_weights)
    rx_model.set_weights(rx_model_weights)

    save_weights_process(tx_model, rx_model)