from keras.layers import Dense, Embedding, LSTM, Conv2D, Reshape, Dropout, Concatenate
from keras.initializers import GlorotNormal, Zeros
from keras.activations import softmax
import tensorflow as tf
from keras.backend import permute_dimensions

class ControlUnit(tf.keras.layers.Layer):
    def __init__(self, dim, max_step=12):
        super(ControlUnit, self).__init__()

        self.position_aware = [Dense(dim, kernel_initializer=GlorotNormal(), bias_initializer=Zeros()) for _ in range(max_step)]

        self.control_question = Dense(dim, kernel_initializer=GlorotNormal(), bias_initializer=Zeros())
        self.attn = Dense(1, kernel_initializer=GlorotNormal(), bias_initializer=Zeros())
        self.concat = Concatenate()

        self.dim = dim
        self.max_step = max_step

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "max_step": self.max_step,
        })
        return config

    def call(self, step, context, question, control):
        position_aware = tf.switch_case(
            step, 
            branch_fns=[lambda: self.position_aware[i](question) for i in range(self.max_step)]
        )

        control_question = self.concat([control, position_aware])
        # control_question = self.concat([tf.expand_dims(control, -1), tf.expand_dims(position_aware, -1)])
        control_question = self.control_question(control_question)
        control_question = tf.expand_dims(control_question, -1)

        context_tr = tf.transpose(context, perm=[0, 2, 1])
        context_prod = control_question * context_tr
        attn_weight = self.attn(context_prod)

        attn = softmax(attn_weight, 1)

        next_control = tf.reduce_sum(attn * context_tr, -1)

        return next_control

class ReadUnit(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(ReadUnit, self).__init__()

        self.mem = Dense(dim, kernel_initializer=GlorotNormal(), bias_initializer=Zeros())
        self.concat = Dense(dim, kernel_initializer=GlorotNormal(), bias_initializer=Zeros())
        self.attn = Dense(1, kernel_initializer=GlorotNormal(), bias_initializer=Zeros())
        self.dim = dim

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
        })
        return config

    def call(self, memory, know, control):
        mem = tf.expand_dims(self.mem(memory[-1]), -1)

        concat = self.concat(tf.transpose(tf.concat([mem * know, know], 1), perm=[0, 2, 1]))
        attn = concat * tf.expand_dims(control[-1], 1)
        attn = tf.squeeze(self.attn(attn), -1)
        attn = tf.expand_dims(softmax(attn, 1), 1)
        read = tf.reduce_sum(attn * know, 1)
        
        return read


class WriteUnit(tf.keras.layers.Layer):
    def __init__(self, dim, self_attention=False, memory_gate=False):
        super(WriteUnit, self).__init__()
        self.dim = dim

        self.concat = Dense(dim, kernel_initializer=GlorotNormal(), bias_initializer=Zeros())

        self.linear = Dense(1)

        if self_attention:
            self.attn = Dense(1, kernel_initializer=GlorotNormal(), bias_initializer=Zeros())
            self.mem = Dense(dim, kernel_initializer=GlorotNormal(), bias_initializer=Zeros())

        if memory_gate:
            self.control = Dense(1, kernel_initializer=GlorotNormal(), bias_initializer=Zeros())

        self.self_attention = self_attention
        self.memory_gate = memory_gate

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "self_attention": self.self_attention,
            "memory_gate": self.memory_gate,
        })
        return config

    def call(self, memories, retrieved, controls):
        prev_mem = memories[-1]
        
        # retrieved = self.linear(tf.expand_dims(retrieved, -1))
        # retrieved = Reshape((self.dim,))(retrieved)
        concat = self.concat(tf.concat([retrieved, prev_mem], -1))
        next_mem = concat

        if self.self_attention:
            controls_cat = tf.stack(controls[:-1], 2)
            attn = tf.expand_dims(controls[-1], 2) * controls_cat
            attn = self.attn(permute_dimensions(attn, pattern=[0, 2, 1]))
            attn = permute_dimensions(softmax(attn, 1), pattern=[0, 2, 1])

            memories_cat = tf.stack(memories, 2)
            attn_mem = tf.reduce_sum(attn * memories_cat, 2)
            next_mem = self.mem(attn_mem) + concat

        if self.memory_gate:
            control = self.control(controls[-1])
            gate = tf.sigmoid(control)
            next_mem = gate * prev_mem + (1 - gate) * next_mem

        return next_mem

class MACUnit(tf.keras.layers.Layer):
    def __init__(self, dim, max_step=12, self_attention=False, memory_gate=False, dropout=0.15):
        super(MACUnit, self).__init__()

        self.self_attention = self_attention
        self.memory_gate = memory_gate

        self.control = ControlUnit(dim, max_step)
        self.read = ReadUnit(dim)
        self.write = WriteUnit(dim, self_attention, memory_gate)

        self.mem_0 = self.add_weight(shape=(1, dim), initializer=Zeros())
        self.control_0 = self.add_weight(shape=(1, dim), initializer=Zeros())

        self.dim = dim
        self.max_step = max_step
        self.dropout = dropout

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "self_attention": self.self_attention,
            "memory_gate": self.memory_gate,
            "dropout": self.dropout,
        })
        return config

    def call(self, context, question, knowledge):
        b_size = tf.shape(question)[0]

        control = tf.tile(self.control_0, [b_size, 1])
        memory = tf.tile(self.mem_0, [b_size, 1])

        controls = [control]
        memories = [memory]

        for i in range(self.max_step):
            control = self.control(i, context, question, control)
            controls.append(control)

            read = self.read(memories, knowledge, controls)
            
            memory = self.write(memories, read, controls)
            memories.append(memory)

        return memory


class MACNetwork(tf.keras.layers.Layer):
    def __init__(self, n_vocab=100, dim=128, embed_hidden=300, max_step=4, self_attention=True, memory_gate=False, n_classes=28, dropout=0.15):
        super(MACNetwork, self).__init__()

        self.n_vocab = n_vocab
        self.embed_hidden =embed_hidden
        self.n_classes =n_classes
        self.memory_gate = memory_gate
        self.self_attention = self_attention

        self.conv = Conv2D(dim, (3, 3), padding='same', activation='elu', kernel_initializer=GlorotNormal())
        self.conv1 = Conv2D(dim, (3, 3), padding='same', activation='elu', kernel_initializer=GlorotNormal())

        self.embed = Embedding(n_vocab, embed_hidden, mask_zero=True)
        self.lstm = LSTM(embed_hidden, return_sequences=True, return_state=True, kernel_initializer=GlorotNormal())
        self.lstm_proj = Dense(dim, kernel_initializer=GlorotNormal())

        self.mac = MACUnit(dim, max_step, self_attention, memory_gate, dropout)

        self.classifier = Dense(dim, activation='elu', kernel_initializer=GlorotNormal())
        # self.classifier1 = Dense(n_classes, activation='softmax', kernel_initializer=HeNormal())
    
        self.dropout = Dropout(0.15)

        self.max_step = max_step
        self.dim = dim

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_vocab": self.n_vocab,
            "dim": self.dim,
            "embed_hidden": self.embed_hidden,
            "max_step": self.max_step,
            "self_attention": self.self_attention,
            "memory_gate": self.memory_gate,
            "n_classes": self.n_classes,
            "dropout": self.dropout,
        })
        return config

    def call(self, x):
        image, question = x

        img = self.conv1(self.conv(image))
        img = Reshape([self.dim, -1])(img)

        lstm_out, h, _ = self.lstm(question)
        lstm_out = self.lstm_proj(lstm_out)
        
        memory = self.mac(lstm_out, h, img)

        # raise ValueError
        out = tf.concat([memory, h], 1)
        out = self.dropout(out)
        out = self.classifier(out)
        return out