
from keras.applications.resnet import ResNet101
from keras.models import Model
from keras.layers import Conv2D, Reshape, Lambda, Input, Embedding, Bidirectional, LSTM, Dense, Flatten, LayerNormalization
import tensorflow as tf
from mac_model import MACNetwork

# load ResNet-101 
base_model = ResNet101(weights='imagenet', include_top=False)

# extract first 30 blocks
resnet_30_blocks = Model(inputs=base_model.input, outputs=base_model.get_layer('conv4_block23_out').output)

# define custom lambda to calculate L2-norm
def complex_l2_norm(x):
    squared_norm = tf.reduce_sum(tf.square(tf.abs(x)), axis=-1, keepdims=True)
    real_norm = tf.sqrt(squared_norm + 1e-10)  # Add a small epsilon to avoid division by zero
    real_norm = tf.cast(real_norm, tf.complex64)  # Cast real_norm to complex64 to match dtype of x
    return x / real_norm

def real_to_complex(x, half_len):
    real = x[..., :half_len]  # Real part
    imag = x[..., half_len:]  # Imaginary part
    return tf.complex(real, imag)

def complex_to_real(x):
    real_part = tf.cast(tf.math.real(x), dtype=tf.float32)
    imag_part = tf.cast(tf.math.imag(x), dtype=tf.float32)
    return tf.concat([real_part, imag_part], axis=-1)

# Feature extraction def function 
def build_transmitter_model():
    input_text_num = 46
    image_shape = (224, 224, 3)
    text_shape = (input_text_num)

    img_inputs = Input(shape=image_shape)
    txt_inputs = Input(shape=text_shape)

    # Image semantic encoding
    img_features = resnet_30_blocks(img_inputs) # (1, 14, 14, 1024)

    # Image channel encoding
    img_features = Conv2D(512, (3, 3), padding='same', activation='elu')(img_features)
    img_features = Conv2D(128, (3, 3), padding='same', activation='elu')(img_features)

    img_features = real_to_complex(img_features, 64)

    # Reshape layer & normalization process
    img_features = Flatten()(img_features)
    img_outputs = Lambda(complex_l2_norm)(img_features)

    # Text embedding process
    emb_features = Embedding(100, 300, input_length=input_text_num)(txt_inputs)
    
    # Text semantic encoding
    txt_features = Bidirectional(LSTM(units=512, activation='tanh', return_sequences=True))(emb_features)

    # Text channel encoding
    txt_features = Dense(256, activation='relu')(txt_features)
    txt_features = Dense(256, activation='relu')(txt_features)

    txt_features = real_to_complex(txt_features, 128)
    txt_features = Flatten()(txt_features)
    txt_outputs = Lambda(complex_l2_norm)(txt_features)

    model = Model(inputs=[img_inputs, txt_inputs], outputs=[img_outputs, txt_outputs], name="tx_model")
    # model.summary()
    return model

def build_receiver_model():
    image_shape = (12544,)
    text_shape = (5888,)

    img_inputs = Input(shape=image_shape)
    txt_inputs = Input(shape=text_shape)

    # Reshape layer
    img_features = complex_to_real(img_inputs)
    img_features = Reshape((14, 14, 128))(img_features)
    img_features = Conv2D(256, (3, 3), padding='same', activation='elu')(img_features)
    img_features = Conv2D(512, (3, 3), padding='same', activation='elu')(img_features) # img_features shape = (None, 14, 14, 512)

    txt_features = complex_to_real(txt_inputs)
    txt_features = Reshape((46, 256))(txt_features)

    # Text channel encoding
    txt_features = Dense(256, activation='relu')(txt_features)
    txt_features = Dense(256, activation='relu')(txt_features)
    txt_features = Dense(512, activation='relu')(txt_features)
    txt_features = LayerNormalization(axis=1)(txt_features) # txt_features shape = (None, 46, 512)
    
    # extract features with MACNetwork
    mac_model = MACNetwork()
    mac_output = mac_model([img_features, txt_features])
    
    # Output layer
    cls_output = Dense(28, activation='softmax')(mac_output)
    model = Model(inputs=[img_inputs, txt_inputs], outputs=cls_output, name="rx_model")
    # model.summary()
    return model

def perfect_channel_estimation(Y, H):
    # Calculate the channel estimate
    Hermitian_matrix = tf.transpose(tf.linalg.adjoint(H))
    X_hat = tf.linalg.solve(tf.matmul(Hermitian_matrix, H), tf.matmul(Hermitian_matrix, tf.transpose(Y)))
    X_hat = tf.transpose(X_hat)
    return X_hat

# Channel model
class Chan_Model(object):  
    def __init__(self, name):
        self.name = name
    
    def __call__(self, _input, snr_db, channel_type='awgn'):
        snr = tf.pow(10.0, snr_db / 10.0)  # dB to linear scale
        std = tf.sqrt(1.0 / (2.0 * snr))  # Calculate standard deviation from SNR
        std_value = tf.cast(std / tf.sqrt(tf.cast(2, dtype=tf.float32)), dtype=tf.float32)

        if channel_type == 'awgn':
            noise_real = tf.random.normal(tf.shape(_input), mean=0.0, stddev=std_value, dtype=tf.float32)
            noise_imag = tf.random.normal(tf.shape(_input), mean=0.0, stddev=std_value, dtype=tf.float32)
            noise = tf.complex(noise_real, noise_imag)
            output = _input + noise
            channel_matrix = tf.eye(tf.shape(_input)[-1], dtype=tf.complex64)
        elif channel_type == 'rayleigh':
            # Rayleigh fading
            h_real = tf.random.normal(tf.shape(_input), mean=0.0, stddev=std_value, dtype=tf.float32)
            h_imag = tf.random.normal(tf.shape(_input), mean=0.0, stddev=std_value, dtype=tf.float32)
            H = tf.complex(h_real, h_imag)
            noise = tf.complex(tf.random.normal(tf.shape(_input), mean=0.0, stddev=std_value, dtype=tf.float32),
                               tf.random.normal(tf.shape(_input), mean=0.0, stddev=std_value, dtype=tf.float32))
            output = H * _input + noise
            channel_matrix = tf.matmul(tf.transpose(H,perm=[1,0]), tf.math.conj(H))  # Outer product
        elif channel_type == 'rician':
            # Rician fading
            K_factor = 1.0  # LOS power to Non-LOS power. Change accordingly.
            h_direct = tf.complex(tf.sqrt(K_factor) * std, 0.0)  # LOS Component
            h_real = tf.random.normal(tf.shape(_input), mean=0.0, stddev=std_value, dtype=tf.float32)  # NLOS Component
            h_imag = tf.random.normal(tf.shape(_input), mean=0.0, stddev=std_value, dtype=tf.float32)  # NLOS Component
            h_nlos = tf.complex(h_real, h_imag)
            H = h_direct + h_nlos
            output = H * _input
            noise_real = tf.random.normal(tf.shape(_input), mean=0.0, stddev=std_value, dtype=tf.float32)
            noise_imag = tf.random.normal(tf.shape(_input), mean=0.0, stddev=std_value, dtype=tf.float32)
            noise = tf.complex(noise_real, noise_imag)
            output += noise
            channel_matrix = tf.matmul(tf.transpose(H,perm=[1,0]), tf.math.conj(H))  # Outer product
        else:
            raise ValueError("Invalid channel type. Choose 'awgn', 'rayleigh', or 'rician'.")
        
        return output