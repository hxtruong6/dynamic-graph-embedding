import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras import Model, regularizers, initializers
import numpy as np
import json

from src.utils.net2net import net2wider, net2deeper


# https://github.com/paulpjoby/DynGEM
class PartCoder(Layer):
    def __init__(self, input_dim, output_dim=2, hidden_dims=None, l1=0.01, l2=0.01, seed=6):
        super(PartCoder, self).__init__()
        self.l1 = l1
        self.l2 = l2
        self.seed = seed
        # self.layers = NoDependency([])
        # self.__dict__['layers'] = []
        self.layers = []

        _input_dim = input_dim
        for i, dim in enumerate(hidden_dims):
            layer = Dense(
                units=dim,
                activation=tf.nn.relu,
                kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2),
                kernel_initializer=initializers.GlorotNormal(seed=self.seed),
                bias_initializer=initializers.Zeros()
            )
            layer.build(input_shape=(None, _input_dim))
            _input_dim = dim
            self.layers.append(layer)

        # Final, adding output_layer (latent/reconstruction layer)
        layer = Dense(
            units=output_dim,
            activation=tf.nn.sigmoid,
            kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2),
            kernel_initializer=initializers.GlorotNormal(seed=6),
            bias_initializer=initializers.Zeros()
        )
        layer.build(input_shape=(None, _input_dim))
        self.layers.append(layer)

    def wider(self, added_size=1, pos_layer=None):
        layers_size = len(self.layers)
        if layers_size < 2:
            raise ValueError("Number of layer must be greater than 2.")
        if pos_layer is None:
            pos_layer = max(layers_size - 2, 0)
        elif pos_layer >= layers_size - 1 or pos_layer < 0:
            raise ValueError(
                f"pos_layer is expected less than length of layers (pos_layer in [0, layers_size-2])")

        # TODO: get biggest value to divide for new weights
        weights, bias = self.layers[pos_layer].get_weights()
        weights_next_layer, bias_next_layer = self.layers[pos_layer + 1].get_weights()

        new_weights, new_bias, new_weights_next_layer = net2wider(weights, bias, weights_next_layer, added_size)

        src_units, des_units = weights.shape[0], weights.shape[1] + added_size
        next_des_units = weights_next_layer.shape[1]

        wider_layer = Dense(
            units=des_units,
            activation=tf.nn.relu,
            kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2)
        )

        # input_shape = (batch_size, input_features).
        # input_features = number of units in layer = length(layer) = output of previous layer
        wider_layer.build(input_shape=(None, src_units))
        wider_layer.set_weights([new_weights, new_bias])

        next_layer = Dense(
            units=next_des_units,
            activation=tf.nn.relu,
            kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2)
        )
        next_layer.build(input_shape=(None, des_units))
        next_layer.set_weights([new_weights_next_layer, bias_next_layer])

        self.layers[pos_layer] = wider_layer
        self.layers[pos_layer + 1] = next_layer

    def deeper(self, pos_layer=None):
        layers_size = len(self.layers)
        if pos_layer is None:
            pos_layer = max(layers_size - 2, 0)
        elif pos_layer >= layers_size - 1 or pos_layer < 0:
            raise ValueError(
                f"pos_layer is expected less than length of layers (pos_layer in [0, layers_size-2]).")

        weights, bias = self.layers[pos_layer].get_weights()
        new_weights, new_bias = net2deeper(weights)
        des_units = weights.shape[1]
        # TODO: add initial kernel
        layer = Dense(
            units=des_units,
            activation=tf.nn.relu,
            kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2),
        )
        layer.build(input_shape=(None, des_units))
        layer.set_weights([new_weights, new_bias])

        self.layers.insert(pos_layer + 1, layer)

    def set_dump_weight(self, dum_weight=None):
        for i in range(len(self.layers)):
            w, b = self.layers[i].get_weights()

            for u in range(w.shape[0]):
                for v in range(w.shape[1]):
                    if dum_weight is None:
                        w[u][v] = u * w.shape[1] + v
                    else:
                        w[u][v] = dum_weight
            for v in range(b.shape[0]):
                b[v] = v
                if dum_weight is None:
                    b[v] = v
                else:
                    b[v] = dum_weight

            self.layers[i].set_weights([w, b])

    def call(self, inputs):
        z = inputs
        for layer in self.layers:
            z = layer(z)

        return z

    def info(self, show_weight=False, show_config=False):
        print(f"{self.name}\n----------")
        print(f"Number of layers: {len(self.layers)}")
        for i, layer in enumerate(self.layers):
            print(f"Layer {i + 1}\n\t Name={layer.name}\n\t Shape ={layer.get_weights()[0].shape}")
            if show_weight:
                print(f"\t Weight= {layer.get_weights()}")
            if show_config:
                print(f"Config: {json.dumps(layer.get_config(), sort_keys=True, indent=4)}")

    def get_length_layers(self):
        return len(self.layers)

    def begin_insert_layer(self, layer_dim):
        # `self.layers[0].get_weights()` -> [weights, bias]
        next_units = self.layers[0].get_weights()[0].shape[0]
        layer = Dense(
            units=next_units,
            activation=tf.nn.relu,
            kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2),
            kernel_initializer=initializers.GlorotNormal(seed=self.seed),
            bias_initializer=initializers.Zeros()
        )
        layer.build(input_shape=(None, layer_dim))
        self.layers.insert(0, layer)

    def last_insert_layer(self, layer_dim):
        prev_weights, prev_bias = self.layers[len(self.layers) - 1].get_weights()
        prev_units = prev_weights.shape[1]

        replace_prev_layer = Dense(
            units=prev_units,
            activation=tf.nn.relu,
            kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2),
        )
        replace_prev_layer.build(input_shape=(None, prev_weights.shape[0]))
        replace_prev_layer.set_weights([prev_weights, prev_bias])

        added_layer = Dense(
            units=layer_dim,
            activation=tf.nn.sigmoid,
            kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2),
            kernel_initializer=initializers.GlorotNormal(seed=self.seed),
            bias_initializer=initializers.Zeros()
        )
        added_layer.build(input_shape=(None, prev_units))

        del self.layers[len(self.layers) - 1]
        self.layers.append(replace_prev_layer)
        self.layers.append(added_layer)

    def get_layers_size(self):
        layers_size = []
        for layer in self.layers:
            weights, _ = layer.get_weights()
            layers_size.append(weights.shape)
        # print("layer_size: ", layers_size)
        return layers_size

    def get_weights(self):
        '''

        :return: [[weights, bias],[],...]
        '''
        layer_weights = []
        for layer in self.layers:
            layer_weights.append(layer.get_weights())
        return layer_weights

    def set_weights(self, weights):
        '''

        :param weights: [[weights, bias],[],...]
        :return:
        '''
        for i in range(0, len(self.layers)):
            # self.layers[i].build(input_shape=(None, weights[i][0].shape[1]))
            if not self.layers[i].get_weights():
                self.layers[i].build(input_shape=(None, len(weights[i][0])))
            self.layers[i].set_weights(weights[i])


class Autoencoder(Model):
    def __init__(self, input_dim, embedding_dim, hidden_dims=None, v1=0.01, v2=0.01):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        if hidden_dims is None:
            hidden_dims = [512, 128]

        self.hidden_dims = hidden_dims
        self.l1 = v1
        self.l2 = v2

        self.encoder = PartCoder(input_dim=input_dim, output_dim=embedding_dim, hidden_dims=hidden_dims, l1=self.l1,
                                 l2=self.l2)
        self.decoder = PartCoder(input_dim=embedding_dim, output_dim=input_dim, hidden_dims=hidden_dims[::-1],
                                 l1=self.l1,
                                 l2=self.l2)

    def wider(self, added_size=1, pos_layer=None):
        if pos_layer is None:
            pos_layer = self.encoder.get_length_layers() - 2

        self.encoder.wider(added_size=added_size, pos_layer=pos_layer)
        self.decoder.wider(added_size=added_size, pos_layer=self.decoder.get_length_layers() - pos_layer - 2)

    def deeper(self, pos_layer=None):
        if pos_layer is None:
            pos_layer = self.encoder.get_length_layers() - 2

        self.encoder.deeper(pos_layer=pos_layer)
        self.decoder.deeper(pos_layer=self.decoder.get_length_layers() - pos_layer - 2)

    def call(self, inputs):
        Y = self.encoder(inputs)
        X_hat = self.decoder(Y)
        return X_hat, Y

    def get_embedding(self, inputs):
        return self.encoder(inputs)

    def get_reconstruction(self, inputs):
        return self.decoder(self.encoder(inputs))

    def info(self, show_weight=False, show_config=False):
        self.encoder.info(show_weight, show_config)
        self.decoder.info(show_weight, show_config)

    def expand_first_layer(self, layer_dim):
        self.input_dim = layer_dim
        self.encoder.begin_insert_layer(layer_dim=layer_dim)
        self.decoder.last_insert_layer(layer_dim=layer_dim)
        self.hidden_dims = self.get_hidden_dims()

    def get_layers_size(self):
        '''
        Return size of the encoder layers part. Suppose layers of decoder has same size as the encoder
        :return: layers size of encoder part
        '''
        return self.encoder.get_layers_size()

    def get_input_dim(self):
        return self.input_dim

    def set_dum_weight(self, dum_weight):
        self.encoder.set_dump_weight(dum_weight)
        self.decoder.set_dump_weight(dum_weight)

    def get_weights_model(self):
        '''
        Return a list of layer weights in the total of model
        :return: [[encoder_layer_weights],[decoder_layer_weights]]
        '''
        return [self.encoder.get_weights(), self.decoder.get_weights()]

    def set_weights_model(self, weights):
        encoder_weights, decoder_weights = weights
        self.encoder.set_weights(encoder_weights)
        self.decoder.set_weights(decoder_weights)

    def get_config_layer(self):
        config_layer = {
            "input_dim": self.input_dim,
            "embedding_dim": self.embedding_dim,
            "hidden_dims": self.hidden_dims,
            "l1": self.l1,
            "l2": self.l2
        }
        return config_layer

    def get_hidden_dims(self):
        hidden_dims = []
        for i, (l1, l2) in enumerate(self.get_layers_size()):
            if i == 0:
                continue
            hidden_dims.append(l1)
        return hidden_dims


if __name__ == "__main__":
    # print("\n#######\nEncoder")
    # # Suppose: 4 -> 3-> 5 -> 2
    # encoder = PartCoder(output_dim=2, hidden_dims=[3, 5])
    # x = tf.ones((3, 4))
    # y = encoder(x)
    # # print("y=", y)
    # # encoder.info(show_weight=True, show_config=False)
    # encoder.deeper()
    # y = encoder(x)
    # # print("y=", y)
    # print("After deeper")
    # encoder.info(show_weight=True, show_config=False)
    #
    # # ----------- Decoder -----------
    # print("\n####\nDecoder")
    # # Suppose: 2 -> 5 -> 3 -> 4
    #
    # decoder = PartCoder(output_dim=4, hidden_dims=[5, 3])
    # x = tf.ones((3, 2))
    # y = decoder(x)
    # # print("y=", y)
    # # encoder.info(show_weight=True, show_config=False)
    # decoder.deeper()
    # y = decoder(x)
    # # print("y=", y)
    # print("After deeper")
    # decoder.info(show_weight=True, show_config=False)

    # print("\n#######\nWider encoder")
    # # Suppose: 2 -> 3 -> 2
    # encoder = PartCoder(output_dim=2, hidden_dims=[3, 4, 1])
    # x = tf.ones((3, 2))
    #
    # print("[Original] y=", encoder(x))
    # encoder.info(show_weight=True, show_config=False)
    # print("[Original_1] y=", encoder(x))
    #
    # encoder.set_dump_weight()
    # print("[Dump] y=", encoder(x))
    # encoder.info(show_weight=True, show_config=False)
    #
    # encoder.wider(added_size=4)
    # print("After wider")
    # print("[Wider] y=", encoder(x))
    # encoder.info(show_weight=True, show_config=False)
    #
    # encoder.deeper()
    # print("\n###### Deeper ")
    # print("[Deeper] y=", encoder(x))
    # encoder.info(show_weight=True, show_config=False)
    #
    # encoder.wider()
    # print("\n##### Wider")
    # print("[Wider] y=", encoder(x))
    # encoder.info(show_weight=True, show_config=False)

    # ------ Test autoencoder ---------
    # ae = Autoencoder(input_dim=4, embedding_dim=2, hidden_dims=[3])
    # X = np.random.rand(1, 4).astype(np.float32)
    # X_hat, Y = ae(X)
    # X_ = np.random.rand(5, 4).astype(np.float32)
    # print(ae.get_embedding(inputs=X_))

    # ---------------- Expand first layer AE -----------
    ae = Autoencoder(input_dim=4, embedding_dim=2, hidden_dims=[3])
    X = np.random.rand(1, 4).astype(np.float32)
    X_hat, Y = ae(X)

    # print("Before expand:")
    # ae.info(show_weight=True)

    ae.expand_first_layer(layer_dim=6)
    X_2 = np.random.rand(1, 6).astype(np.float32)
    X_hat, Y = ae(X_2)
    # print("After expand:")
    # ae.info(show_weight=True)
    print(ae.get_layers_size())

    # ------------------ Test wider deeper ------------
    # ae.info()
    # print("##### ----> Modify")
    # ae.wider(added_size=2)
    # ae.deeper()
    # ae.info()
