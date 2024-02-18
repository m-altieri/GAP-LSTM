import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class Logbook:
    def __init__(self):
        super().__init__()
        # self.metrics is a dict of lists of lists:
        # {'m1': [[...], [...], ..., [...]], ..., 'mn': [[...], [...], ..., [...]]}
        self.metrics = {}
        self.figure = plt.figure()

    def register(self, name, value):
        if name not in self.metrics:
            self.metrics[name] = [[]]

        self.metrics[name][-1].append(value)

    def get(self, name, overall=False):
        return self.metrics[name] if overall else self.metrics[name][-1]

    def count(self, name, overall=False):
        return np.size(self.metrics[name]) if overall else len(self.metrics[name][-1])

    def new(self):
        for name in self.metrics:
            self.metrics[name].append([])

    def save_plot(self, path, names=None):
        for metric in self.metrics:
            if names and metric not in names:
                continue
            plt.figure(metric)
            plt.plot(
                np.arange(self.count(metric)),
                np.array(self.get(metric)),
                linewidth=0.1,
                color="black",
                alpha=0.25,
            )
            plt.xticks(np.arange(0, self.count(metric), 3), fontsize=24)
            plt.yticks(fontsize=24)
            plt.xlabel("Timesteps", fontsize=24)
            plt.ylabel("Spatial Autocorrelation", fontsize=24)
            plt.gcf().subplots_adjust(left=0.20, bottom=0.20)
            plt.savefig(os.path.join(path, metric + ".png"))
            plt.savefig(os.path.join(path, metric + ".pdf"))


def morans_I(x, adj):
    x_h = tf.identity(x)  # (B,N,F)
    b, n, f = x_h.shape  # x: (B,N,F)
    epsilon = 1e-3
    x_mean = tf.reduce_mean(x_h, axis=1)  # (B,F)
    numerator = 0.0
    denominator = 0.0

    normalizer = tf.cast(n / tf.math.reduce_sum(adj), tf.float32)  # ()
    for node_i in range(n):
        denominator += tf.cast(
            (x_h[:, node_i] - x_mean) ** 2, tf.float32
        )  # (B,F), (B,F) -> (B,F)

        for node_j in range(n):
            # (), ((B,F), (B,F)) -> (B,F)
            numerator += tf.math.multiply(
                tf.cast(adj[node_i, node_j], tf.float32),
                tf.cast(
                    tf.math.multiply(
                        tf.cast((x_h[:, node_i] - x_mean), tf.float32),
                        tf.cast((x_h[:, node_j] - x_mean), tf.float32),
                    ),
                    tf.float32,
                ),
            )

    I = tf.math.multiply(
        normalizer,
        tf.cast(tf.math.divide(numerator, denominator + epsilon), tf.float32),
    )

    I = tf.math.reduce_mean(I)
    if np.isnan(I):
        print("ERROR: division by zero.")
        print(f"Normalizer: {normalizer:.3f}")
        print(f"Numerator: {numerator}")
        print(f"Denominator: {denominator}")
        sys.exit(1)

    return I
