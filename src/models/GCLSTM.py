"""
GCLSTM.
Questo modello può essere lanciato anche nelle sue versioni ablate:
GCLSTM-noconv: Conv2D disattivata completamente
GCLSTM-closeness: usa le closeness anzichè le correlazioni come matrice di adiacenza
GCLSTM-ones: usa una matrice di adiacenza con tutti 1
GCLSTM-nosummary: usa solo la prima feature statistica anzichè tutte e 7
GCLSTM-noskip: salta la skip connection
"""

import sys
import scipy
import numpy as np
import tensorflow as tf

sys.path.append("../lib")
from lib.spektral_utilities import *
from lib.spektral_gcn import GraphConv


class GCLSTM(tf.keras.Model):
    def __init__(self, H, P, adj, N, node=0, ablation=None, **kwargs):
        super().__init__()

        self.w = None
        if kwargs.get("summary_path", False):
            self.w = tf.summary.create_file_writer(kwargs.get("summary_path"))

        self.H = H
        self.adj = adj
        self.N = N
        self.node = node

        self.gcn1 = GraphConv(32, activation="relu", name="gcn1")
        self.gcn2 = GraphConv(16, activation="relu", name="gcn2")

        self.conv1 = tf.keras.layers.Conv1D(
            4, (3), padding="same", data_format="channels_first", name="conv1"
        )  # 1d
        self.conv2 = tf.keras.layers.Conv1D(
            4, (3), padding="same", data_format="channels_first", name="conv2"
        )  # 1d

        self.bn = tf.keras.layers.BatchNormalization()

        self.pooling = tf.keras.layers.AveragePooling1D(
            (2), padding="same", data_format="channels_first", name="pooling"
        )  # 1d

        self.lstm1 = tf.keras.layers.LSTM(128, return_sequences=True, name="lstm1")
        self.lstm2 = tf.keras.layers.LSTM(128, name="lstm2")

        self.dropout = tf.keras.layers.Dropout(0.5, name="dropout")
        self.out = tf.keras.layers.Dense(P, name="out")

        self.ablation = ablation
        self.gcnmaps = tf.Variable(
            initial_value=tf.zeros([0, N, 16]),
            trainable=False,
            shape=[None, N, 16],
            name="gcnmaps",
        )
        self.convmaps = tf.Variable(
            initial_value=tf.zeros([0, 4, N, 16]),
            trainable=False,
            shape=[None, 4, N, 16],
            name="convmaps",
        )
        self.summaries = tf.Variable(
            initial_value=tf.zeros([0, N, 1 if self.ablation == "nosummary" else 7]),
            trainable=False,
            shape=[None, N, 1 if self.ablation == "nosummary" else 7],
            name="summaries",
        )

    def get_interpretation(self):
        """Returns a tuple containing two numpy arrays: GCN outputs (D,N,16) and Conv2D feature maps (D,4,N,16), where D is the number of test dates."""
        return self.gcnmaps, self.convmaps, self.summaries

    def set_adj(self, adj):
        self.adj = adj

    def set_node(self, n):
        assert n < len(self.adj)
        self.node = n
        print(f"Node {n} set for training in {__name__}.")

    @staticmethod
    def compute_node_features(inputs):
        B, H, N, F = inputs.shape
        return tf.stack(
            [
                tf.math.reduce_mean(inputs[..., 0], axis=(1)),
                tf.math.reduce_mean(inputs[:, H // 2 :, :, 0], axis=(1)),
                tf.math.reduce_std(inputs[..., 0], axis=(1)),
                tf.math.reduce_std(inputs[:, H // 2 :, :, 0], axis=(1)),
                tf.numpy_function(
                    lambda x: np.nan_to_num(
                        np.array(
                            [
                                [
                                    scipy.stats.skew(
                                        x[b, :, n, 0], axis=None, nan_policy="raise"
                                    )
                                    for n in range(N)
                                ]
                                for b in range(B)
                            ],
                            dtype=np.float32,
                        )
                    ),
                    [inputs],
                    tf.float32,
                ),
                tf.numpy_function(
                    lambda x: np.nan_to_num(
                        np.array(
                            [
                                [
                                    scipy.stats.kurtosis(
                                        x[b, :, n, 0], axis=None, nan_policy="raise"
                                    )
                                    for n in range(N)
                                ]
                                for b in range(B)
                            ],
                            dtype=np.float32,
                        )
                    ),
                    [inputs],
                    tf.float32,
                ),
                tf.numpy_function(
                    lambda x: np.nan_to_num(
                        np.array(
                            [
                                [
                                    np.polyfit(np.arange(H), x[b, :, n, 0], 1)[0]
                                    for n in range(N)
                                ]
                                for b in range(B)
                            ],
                            dtype=np.float32,
                        )
                    ),
                    [inputs],
                    tf.float32,
                ),  # coefficient of line that "interpolates" the target across steps, for each node and sequence
            ],
            axis=2,
        )

    def call(self, inputs, training=False):

        node_features = self.compute_node_features(inputs)  # [B,N,7]

        if self.ablation == "nosummary":
            node_features = node_features[..., :1]

        if not training:  # save summaries for interpretation
            self.summaries.assign(tf.concat([self.summaries, node_features], axis=0))

        gcn_out = self.gcn1([node_features, self.adj])
        gcn_out = self.gcn2([gcn_out, self.adj])  # [B,N,16]

        noconv_gcn_out = tf.identity(gcn_out)
        noconv_gcn_out = tf.keras.layers.Flatten(data_format="channels_last")(
            noconv_gcn_out
        )

        if not training:  # save gcn maps for interpretation
            self.gcnmaps.assign(tf.concat([self.gcnmaps, gcn_out], axis=0))

        gcn_out = self.conv1(gcn_out)
        gcn_out = self.pooling(gcn_out)
        first_conv_out = tf.identity(gcn_out)
        gcn_out = tf.nn.relu(gcn_out)
        gcn_out = self.conv2(gcn_out)

        # add & norm
        if not self.ablation == "noskip":
            gcn_out = gcn_out + first_conv_out
            gcn_out = self.bn(gcn_out)

        # flatten
        gcn_out = tf.reshape(gcn_out, (-1))

        lstm_out = self.lstm1(inputs[:, :, self.node, :])
        lstm_out = self.lstm2(lstm_out)  # [B,128]

        if not self.ablation == "noconv":
            out = tf.concat((gcn_out, lstm_out), axis=-1)  # [B,192]
        else:
            out = tf.concat((noconv_gcn_out, lstm_out), axis=-1)

        out = self.dropout(out)
        out = self.out(out)  # [B,P]

        return out

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)

            y = y[:, :, self.node]  # Get only the active node
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        # Info per TensorBoard
        if self.w:
            with self.w.as_default():
                tf.summary.histogram("y", y)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)

        y = y[:, :, self.node]  # Get only the active node

        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}


class GCLSTM_oneclass(tf.keras.Model):
    def __init__(self, h, p, n, adj=None):
        super(GCLSTM_oneclass, self).__init__()

        print(f"Initializing {__name__}.")

        self.h = h
        self.p = p
        self.n = n
        self.real_f = [31, 8, 3, 9]  # @TODO renderlo non hard-coded
        self.adj = adj
        self.node = 0

        self.mappers = [
            tf.keras.layers.Dense(
                32, activation="relu", input_shape=(h, self.real_f[node])
            )
            for node in range(n)
        ]

        self.encoder = tf.keras.layers.Dense(8, activation="relu")
        self.decoder = tf.keras.layers.Dense(32, activation="relu")

        self.demappers = [
            tf.keras.layers.Dense(self.real_f[node], activation="relu")
            for node in range(n)
        ]

    def set_adj(self, adj):
        self.adj = adj

    def set_node(self, node):
        self.node = node
        print(f"Node {node} set for training in {__name__}.")

    def call(self, inputs):
        print(
            "\n\n\nInputs: " + str(inputs.shape)
        )  # [B,H,N,F], F corrisponde al max_n(F), ovvero al numero di feature del nodo che ne ha di più. I NaN sono usati come padding.
        B, H, N, F = inputs.shape
        """
        if self.real_F is None:
            print('Computing the real F...')
            real_F = []

            for n in range(N):  # Controllo feature "vere", in base al padding dei NaN
                first_nan_idx = tf.where(tf.math.is_nan(tf.convert_to_tensor(inputs[:, :, n], dtype=tf.float32)))[0][-1]
                f = first_nan_idx if first_nan_idx is not None else F
                real_F.append(f)
            self.real_F = tf.Variable(real_F, trainable=False)
            print(f'real_F[0]: {real_F[0]}')
        """
        print(f"Inputs contains nan?: {tf.math.reduce_any(tf.math.is_nan(inputs))}")

        common_space_mapping = [
            self.mappers[n](inputs[:, :, n, : self.real_f[n]]) for n in range(N)
        ]
        common_space_mapping = tf.stack(common_space_mapping, axis=2)  # [B,H,N,32]
        print(f"common_space_mapping.shape: {common_space_mapping.shape}")

        encoded = self.encoder(common_space_mapping)  # [B,H,N,8]
        print(f"encoded.shape: {encoded.shape}")

        decoded = self.decoder(encoded)  # [B,H,N,32]
        print(f"decoded.shape: {decoded.shape}")

        demapped = [self.demappers[n](decoded[:, :, n]) for n in range(N)]
        demapped = [
            tf.pad(d, paddings=[[0, 0], [0, 0], [0, F - d.shape[-1]]]) for d in demapped
        ]
        demapped = tf.stack(demapped, axis=2)
        print(f"demapped.shape: {demapped.shape}")

        print(f"Output contains nan?: {tf.math.reduce_any(tf.math.is_nan(demapped))}")

        return demapped

    def train_step(self, data):
        x, y = data
        x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)  # Sostituisco NaN con zeri

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.compiled_loss(
                x, y_pred, regularization_losses=self.losses
            )  # vedo distanza tra input e output

        # Compute gradients and update weights
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(x, y_pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
