import os
import sys
import argparse
import numpy as np

np.random.seed(42)
import matplotlib.pyplot as plt


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.set_visible_devices(physical_devices[:1], "GPU")
logical_devices = tf.config.list_logical_devices("GPU")
assert len(logical_devices) == len(physical_devices) - 1

sys.path.insert(1, os.path.join(sys.path[0], ".."))
import utils.sequence
from models.LSTM import LSTM
from models.SVD_LSTM import SVD_LSTM
from models.CNN_LSTM import CNN_LSTM
from models.GCN_LSTM import GCN_LSTM
from models.GAP_LSTM import GAP_LSTM


# DATASET CONFIGURATION
dataset_config = {
    "lightsource": {
        "timesteps": 19,
        "nodes": 7,
        "features": 11,
        "test_dates": 36,
    },
    "pv-italy": {
        "timesteps": 19,
        "nodes": 17,
        "features": 12,
        "test_dates": 85,
    },
    "wind-nrel": {
        "timesteps": 24,
        "nodes": 5,
        "features": 8,
        "test_dates": 73,
    },
    "beijing-multisite-airquality": {
        "timesteps": 6,
        "nodes": 12,
        "features": 11,
        "test_dates": 146,
    },
}


def create_timeseries(dataset, dataset_name, test_indexes):
    horizon = dataset_config[dataset_name]["timesteps"]
    X, Y = utils.sequence.obs2seqs(dataset, horizon, horizon, horizon)
    test_indexes = test_indexes - horizon // horizon  # l'indice è da quando parte
    trainX = X
    trainY = Y
    testX = X[test_indexes]
    testY = Y[test_indexes]

    # dalla Y prendo solo la label
    trainY = trainY[..., 0]
    testY = testY[..., 0]

    return (trainX, trainY, testX, testY, test_indexes)


def create_adj(adj_path=None):
    adj = np.load(adj_path).astype(np.float32)
    D = np.zeros_like(adj)
    for row in range(len(D)):
        D[row, row] = np.sum(adj[row])  # Degree matrix (D_ii = sum(adj[i,:]))
    sqinv_D = np.sqrt(np.linalg.inv(D))  # Calcola l'inversa e la splitta in due radici
    adj = np.matmul(sqinv_D, np.matmul(adj, sqinv_D))

    if np.isnan(adj).any() or np.isinf(adj).any():
        print(f"Adjacency matrix is nan or infinite: \n{adj}")
        sys.exit(1)
    return adj


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("model")
    argparser.add_argument("dataset")
    argparser.add_argument(
        "-w",
        "--weights-seq",
        type=int,
        help="Test sequence to retrieve the weights from (zero-based). In unspecified, they will be retrieved from the last test sequence of the dataset (end of training).",
    )
    argparser.add_argument(
        "-t",
        "--test-seq",
        type=int,
        help="Test sequence to use (zero-based).",
    )
    args = argparser.parse_args()
    dataset_name = args.dataset
    model_name = args.model

    # LOAD DATA
    dataset = np.load(
        os.path.join("../..", "data", dataset_name, dataset_name + ".npz")
    )["data"]
    adj = create_adj(
        os.path.join("../../data", dataset_name, f"closeness-{dataset_name}.npy")
    )
    test_indexes = np.load(
        os.path.join("../..", "data", dataset_name, dataset_name + "_0.1.npy")
    )
    trainX, trainY, testX, testY, test_indexes = create_timeseries(
        dataset, dataset_name, test_indexes
    )

    # MODEL CONFIGURATION
    model_config = {
        "LSTM": {
            "class": LSTM,
            "args": [
                dataset_config[args.dataset]["nodes"],
                dataset_config[args.dataset]["features"],
                dataset_config[args.dataset]["timesteps"],
            ],
            "kwargs": {},
        },
        "GRU": {
            "class": LSTM,
            "args": [
                dataset_config[args.dataset]["nodes"],
                dataset_config[args.dataset]["features"],
                dataset_config[args.dataset]["timesteps"],
            ],
            "kwargs": {"is_GRU": True},
        },
        "SVD-LSTM": {
            "class": SVD_LSTM,
            "args": [
                dataset_config[args.dataset]["nodes"],
                dataset_config[args.dataset]["features"],
                dataset_config[args.dataset]["timesteps"],
            ],
            "kwargs": {},
        },
        "CNN-LSTM": {
            "class": CNN_LSTM,
            "args": [
                dataset_config[args.dataset]["nodes"],
                dataset_config[args.dataset]["features"],
                dataset_config[args.dataset]["timesteps"],
            ],
            "kwargs": {},
        },
        "GCN-LSTM": {
            "class": GCN_LSTM,
            "args": [
                dataset_config[args.dataset]["nodes"],
                dataset_config[args.dataset]["features"],
                dataset_config[args.dataset]["timesteps"],
                adj,
            ],
            "kwargs": {},
        },
        "GAP-LSTM": {
            "class": GAP_LSTM,
            "args": [
                dataset_config[args.dataset]["timesteps"],
                dataset_config[args.dataset]["timesteps"],
                adj,
                dataset_config[args.dataset]["nodes"],
                dataset_config[args.dataset]["features"],
            ],
            "kwargs": {},
        },
    }

    # BUILD MODEL
    model = model_config[model_name]["class"](
        *model_config[args.model].get("args"),
        **model_config[args.model].get("kwargs"),
    )
    model.compile(run_eagerly=True)

    # Load model weights
    if args.weights_seq is None:
        args.weights_seq = dataset_config[dataset_name]["test_dates"] - 1

    model_weights_path = (
        f"../../saved_models/{model_name}-{dataset_name}-{args.weights_seq}.h5"
    )
    model(trainX[:1])
    model.load_weights(model_weights_path)

    # Predict
    if args.test_seq is None:
        args.test_seq = dataset_config[dataset_name]["test_dates"] - 1

    preds = model.predict(testX[args.test_seq : args.test_seq + 1])[0]
    truth = testY[args.test_seq]

    plt.plot(preds, color="red", linewidth=1, alpha=0.4)
    plt.plot(truth, color="blue", linewidth=1, alpha=0.4)
    plt.plot(np.mean(preds, axis=-1), color="red", linewidth=4)
    plt.plot(np.mean(truth, axis=-1), color="blue", linewidth=4)
    plt.plot([], color="red", label="Predictions")
    plt.plot([], color="blue", label="Ground truth")
    plt.xlabel("Timesteps", fontsize=20)
    plt.ylabel("Target", fontsize=20)
    plt.xticks(
        np.arange(0, dataset_config[dataset_name]["timesteps"], 3),
        np.arange(0, dataset_config[dataset_name]["timesteps"], 3) + 1,
        fontsize=20,
    )
    plt.yticks(fontsize=20)
    plt.legend(fontsize=18)
    plt.subplots_adjust(left=0.2, bottom=0.2)

    if not os.path.exists("../../pred_curves"):
        os.makedirs("../../pred_curves")
    plt.savefig(
        f"../../pred_curves/{args.model}-{args.dataset}-w{args.weights_seq}-t{args.test_seq}.png"
    )
    plt.savefig(
        f"../../pred_curves/{args.model}-{args.dataset}-w{args.weights_seq}-t{args.test_seq}.pdf"
    )


if __name__ == "__main__":
    main()
