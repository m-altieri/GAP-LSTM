import os
import sys
import time
import json
import math
import pickle
import datetime
import argparse
import utils.math
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from colorama import Fore, Style
from sklearn.utils import shuffle


sys.path.append("./models")
sys.path.append("/lustrehome/altieri/research/src/models")

import utils.logging
import utils.sequence
from utils.dicts import dict_union

from models.VAR import VAR
from models.LSTM import LSTM
from models.ARIMA import ARIMA
from models.GCLSTM import GCLSTM
from models.Prophet import Prophet
from models.CNN_LSTM import CNN_LSTM
from models.SVD_LSTM import SVD_LSTM
from models.GCN_LSTM import GCN_LSTM
from models.GAP_LSTM import GAP_LSTM


def parse_args():
    argparser = argparse.ArgumentParser(
        description="Run the experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument("model", action="store", help="select the model to run")
    argparser.add_argument(
        "dataset", action="store", help="select the dataset to run the model on"
    )
    argparser.add_argument(
        "-c",
        "--conf",
        action="store",
        help="select the model configuration to run; not setting it has the same effect of setting -a",
    )
    argparser.add_argument(
        "-a",
        "--all",
        action="store_const",
        const=True,
        help="run all configurations; if this flag is set, -c is ignored",
    )
    argparser.add_argument(
        "-r", "--run-name", action="store", help="set the name of the output folder"
    )
    argparser.add_argument(
        "-m",
        "--mode",
        action="store",
        choices=["landmark", "sliding"],
        default="landmark",
        help="select the evaluation mode; default is landmark",
    )
    argparser.add_argument(
        "-w",
        "--wsize",
        action="store",
        type=int,
        choices=[30, 60, 90],
        default=30,
        help="select the window size; if -m is landmark, this is ignored",
    )
    argparser.add_argument(
        "-n",
        "--starting-node",
        action="store",
        default=0,
        type=int,
        help="choose the node you want to start training from, and skip the previous ones. This only affects single-node models.",
    )
    argparser.add_argument(
        "-d",
        "--prediction-seqs",
        action="store",
        choices=["test", "val"],
        default="test",
        help="choose the file containing prediction sequence indexes that you want to use",
    )
    argparser.add_argument(
        "-e",
        "--epochs",
        action="store",
        type=int,
        default=5,
        help="override the number of epochs in the config file",
    )
    argparser.add_argument(
        "-b",
        "--batch-size",
        action="store",
        type=int,
        default=32,
        help="override the batch size in the config file",
    )
    argparser.add_argument(
        "-l",
        "--learning-rate",
        action="store",
        type=float,
        default=1e-3,
        help="override the learning rate in the config file",
    )
    argparser.add_argument(
        "--recas",
        action="store_true",
        help="set the correct path for execution on recas",
    )
    argparser.add_argument(
        "--collect-only",
        action="store_true",
        help="skip to results collection",
    )
    argparser.add_argument(
        "--interpretable",
        action="store_true",
        help="attempt to retrieve interpretable information from the model",
    )
    argparser.add_argument(
        "--skip-finished",
        action="store_true",
        help="skip configurations whose results have already been written. Only has effect if -a is set",
    )

    # @TODO molto rudimentale; per ora semi-funziona solo per GCLSTM
    argparser.add_argument(
        "--tensorboard",
        action="store_true",
        help="log information to tensorboard",
    )
    argparser.add_argument(
        "--distributed",
        action="store_true",
        help="run the experiment in distributed mode. use only with horovodrun.",
    )
    argparser.add_argument(
        "--adj",
        action="store",
        choices=["ones", "closeness", "corr"],
        default="closeness",
    )
    argparser.add_argument("--replicate-nodes", action="store", type=int)
    argparser.add_argument("--ablation", action="store")
    argparser.add_argument(
        "-s",
        action="store_true",
        help="Run a single experiment. Don't take params from files",
    )
    argparser.add_argument(
        "-g",
        "--gnn-type",
        choices=["spektral", "weighted", "gat"],
        help="Define the GNN type",
    )
    argparser.add_argument("--early-stopping", action="store_true")
    argparser.add_argument("--exp-name")

    args = argparser.parse_args()

    if args.exp_name is None:
        args.exp_name = f'undefined-{args.model}-{args.dataset}-{datetime.datetime.now().strftime("%H%M%S")}'

    return args


def gpu_config(distributed):
    for device in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)

    if distributed:
        from paramiko import SSHClient
        from scp import SCPClient
        import horovod.tensorflow as hvd

        hvd.init()
        tf.config.set_visible_devices(
            tf.config.experimental.list_physical_devices("GPU")[hvd.local_rank()], "GPU"
        )
        print(
            f"Running in distributed mode: process {hvd.rank()} is using device {tf.config.list_logical_devices('GPU')}"
        )


def create_timeseries(
    timeseries_path, test_file_path, experiment_params, model_params, dataset_params
):
    print("Loading data...")

    data = np.load(timeseries_path, allow_pickle=True)["data"]
    if (replicate_nodes := experiment_params.get("replicate_nodes")) > 0:
        data = np.repeat(data, replicate_nodes, axis=1)
        print(f"Replicating nodes {replicate_nodes} times.")
    if experiment_params["mini_dataset"]:
        data = data[
            : len(data) / dataset_params["h"] // 10 * dataset_params["h"],
            :,
            :,
        ]

    print("Data loaded with shape " + str(data.shape) + ".")

    # Splitta le timeseries in tanti blocchi di lunghezza h + p
    print("Converting timeseries into windows of history steps and prediction steps...")
    X, Y = utils.sequence.obs2seqs(
        data,
        dataset_params["h"],
        dataset_params["p"],
        dataset_params["stride"],
    )

    sequences, _, _, _ = X.shape  # (S,T,N,F)
    print(
        f"Conversion complete: {sequences} sequences produced from {data.shape[0]} observations."
    )

    # Calcolo dimensione di train, val e test set
    print(
        f"Calculating training, validation and test splits {experiment_params['training_set_ratio']}, {experiment_params['validation_set_ratio']}, {experiment_params['test_set_ratio']})..."
    )
    training_seqs, val_seqs, test_seqs = utils.math.divide_into_3_integers(
        sequences,
        experiment_params["training_set_ratio"],
        experiment_params["validation_set_ratio"],
        experiment_params["test_set_ratio"],
    )

    if experiment_params["test_seqs_source"] == "random":
        # Split casuale delle sequenze in train, val e test. Qui vengono solo selezionate le sequenze perchè mi devo salvare gli indici, lo split vero viene dopo.
        print("Shuffling data with seed 42.")
        indexes = np.arange(sequences)
        rnd_above_buffer_perm = shuffle(
            indexes[experiment_params["buffer"] :], random_state=42
        )
        # val_indexes = rnd_above_buffer_perm[:val_seqs]
        test_indexes = rnd_above_buffer_perm[val_seqs : val_seqs + test_seqs]
        perm = shuffle(indexes, random_state=42)
        # train_indexes = np.setdiff1d(
        #     perm, np.concatenate((val_indexes, test_indexes)), assume_unique=True
        # )
        train_indexes = np.setdiff1d(perm, test_indexes, assume_unique=True)

    elif experiment_params["test_seqs_source"] == "file":
        test_indexes = (
            np.load(test_file_path) - dataset_params["h"] // dataset_params["stride"]
        )  # l'indice è da quando parte

        test_indexes = [index for index in test_indexes if index < sequences]
        train_indexes = np.setdiff1d(
            np.arange(sequences), test_indexes, assume_unique=True
        )
        # val_indexes = []

    # Se ALLOW_PARTIAL_BATCHES == False, tronca gli ultimi indici per far sì che il totale sia un multiplo di BATCH_SIZE
    if not experiment_params["allow_partial_batches"]:
        print(
            f"Making number of samples a multiple of batch size {model_params['batch_size']}..."
        )
        train_indexes = train_indexes[
            : len(train_indexes)
            // model_params["batch_size"]
            * model_params["batch_size"]
        ]
        # val_indexes = val_indexes[
        #     : len(val_indexes)
        #     // model_params["batch_size"]
        #     * model_params["batch_size"]
        # ]
        test_indexes = test_indexes[
            : len(test_indexes)
            // model_params["batch_size"]
            * model_params["batch_size"]
        ]
        print("Number of samples is a multiple of batch size")
    else:
        print("Skipped truncation of partial batches.")

    # Riordina gli indici
    train_indexes.sort()
    # val_indexes.sort()
    test_indexes.sort()

    # Uso tutte le date per il training, anche quelle di test (dopo che le ho usate per predire)
    trainX = X
    trainY = Y
    # valX = X[val_indexes if experiment_params["mode"] == "default" else []]
    # valY = Y[val_indexes if experiment_params["mode"] == "default" else []]
    testX = X[test_indexes]
    testY = Y[test_indexes]

    # Dall'Y prendo solo la label
    trainY = trainY[..., 0]
    # valY = valY[..., 0]
    testY = testY[..., 0]

    print(
        "Loading and preprocessing complete with shapes:"
        + f"\n\tTrainX shape: {trainX.shape}"
        + f"\n\tTrainY shape: {trainY.shape}"
        # + f"\n\tValX shape: {valX.shape}"
        # + f"\n\tValY shape: {valY.shape}"
        + f"\n\tTestX shape: {testX.shape}"
        + f"\n\tTestY shape: {testY.shape}"
    )

    return (trainX, trainY, testX, testY, test_indexes)


def create_adj(adj_path=None, sequence=None, show=False, save=False, **kwargs):
    """
    Create adjacency matrix
    -----------------------
    Parameters:
    adj_path (String, optional): if specified, the matrix will be loaded from file, from the specified path.
    The adjacency matrix loaded this way is assumed to already have self-loops (values on the diagonal should be the highest).
    sequence (Tensor, optional): if specified, the matrix will be calculated using the correlation of the first feature between nodes.
    It must be a tensor of shape [B,T,N,F].
    Exactly one between adj_path and sequence must be specified.
    The adjacency matrix is normalized but the Laplacian is not used.
    """
    assert (adj_path is None) + (
        sequence is None
    ) == 1  # One must be specified, the other must be None

    if adj_path:
        print("Loading adjacency matrix from " + adj_path + "...")
        adj = np.load(adj_path).astype(np.float32)
        if (replicate_nodes := kwargs.get("replicate_nodes", False)) > 0:
            adj = np.tile(adj, (replicate_nodes, replicate_nodes))
    elif sequence is not None:
        print(
            "Computing adjacency matrix from sequence of shape {}...".format(
                np.shape(sequence)
            )
        )
        B, T, N, F = np.shape(sequence)
        sequence = sequence[..., 0]  # [B,T,N]
        adj = np.corrcoef(
            np.reshape(sequence, (B * T, N)), rowvar=False
        )  # Variables are nodes, which are on the columns
    else:
        raise ValueError("Exactly one between adj_path and sequence must be specified.")

    D = np.zeros_like(adj)
    for row in range(len(D)):
        D[row, row] = np.sum(adj[row])  # Degree matrix (D_ii = sum(adj[i,:]))
    sqinv_D = np.sqrt(np.linalg.inv(D))  # Calcola l'inversa e la splitta in due radici
    adj = np.matmul(sqinv_D, np.matmul(adj, sqinv_D))

    if show:
        plt.figure(figsize=(12, 12))
        sns.heatmap(adj)
    if save:
        plt.savefig(
            os.path.join(
                self.path,
                "heatmaps",
                "corr-" + datetime.datetime.now().strftime("%H%M%S") + ".png",
            )
        )
    print("Adjacency matrix loaded with shape " + str(adj.shape) + ".")

    # Checking for NaN or infinite values
    if np.isnan(adj).any():
        adj = np.nan_to_num(adj, nan=0)
        print("Adjacency matrix contains NaN values. They have been replaced with 0.")
    if np.isnan(adj).any() or np.isinf(adj).any():
        print(
            "Adjacency matrix is nan or infinite:\n"
            + str(adj)
            + "\nTerminating due to critical error."
        )
        sys.exit(1)

    return adj


def build_model(experiment_params, model_params, dataset_params, adj=None):
    print(
        f"Starting model construction for model type {(model_type := model_params['name'])}...."
    )
    h, p, n, f = (dataset_params[k] for k in ["h", "p", "n", "f"])

    model = None
    model = {
        "GAP-LSTM": GAP_LSTM(
            dict_union(experiment_params, dataset_params, model_params), adj
        ),
        "GCLSTM": GCLSTM(
            h,
            p,
            adj,
            n,
            summary=experiment_params.get("summary"),
            ablation=experiment_params.get("ablation"),
        ),
        "ARIMA": ARIMA(),
        "VAR": VAR(p, order=model_params.get("order")),
        "Prophet": Prophet(multivariate=False),
        "LSTM": LSTM(n, f, p),
        "Bi-LSTM": LSTM(n, f, p, is_bidirectional=True),
        "Attention-LSTM": LSTM(n, f, p, has_attention=True),
        "GRU": LSTM(n, f, p, is_GRU=True),
        "CNN-LSTM": CNN_LSTM(n, f, p, has_dense=True),
        "SVD-LSTM": SVD_LSTM(n, f, p, has_dense=True),
        "GCN-LSTM": GCN_LSTM(n, f, p, adj),
    }[model_type]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(model_params["learning_rate"]),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(),
            tf.keras.metrics.RootMeanSquaredError(),
        ],
        run_eagerly=True,
    )

    print("Model built and compiled.")
    return model


def train_and_predict(
    model,  # model object
    trainX,
    trainY,
    testX,
    testY,
    test_indexes,
    experiment_params,
    model_params,
    dataset_params,
    adj,
    node=0,
):
    print(
        "Training starting with:\n\t"
        + f"Training X shape: {trainX.shape}\n\t"
        + f"Training Y shape: {trainY.shape}\n\t"
        # + f"Validation X shape: {valX.shape}\n\t"
        # + f"Validation Y shape: {valY.shape}\n\t"
        + f"Test X shape: {testX.shape}\n\t"
        + f"Test Y shape: {testY.shape}\n\t"
        + f"Batch size: {model_params['batch_size']}\n\t"
        + f"Epochs: {model_params['epochs']}"
    )

    callbacks = []

    if model_params["lr_scheduler"]:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=model_params["lrs_monitor"],
                factor=model_params["lrs_factor"],
                patience=model_params["lrs_patience"],
                mode="min",
                verbose=1,
            )
        )

    if model_params["early_stopping"]:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=model_params["es_monitor"],
                patience=model_params["es_patience"],
                restore_best_weights=model_params["es_restore"],
                mode="min",
                verbose=1,
            )
        )
    if experiment_params.get("tensorboard"):
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                experiment_params.get("summary", None), update_freq=1
            )
        )

    preds = []
    if experiment_params.get("interpretable"):
        print("Creating explainer")
        explainer = lime.lime_tabular.LimeTabularExplainer(
            np.reshape(trainX, (trainX.shape[0], 19 * 7 * 11)), mode="regression"
        )

    if experiment_params["mode"] == "default":
        model.fit(
            x=trainX,
            y=trainY,
            batch_size=model_params["batch_size"],
            epochs=model_params["epochs"],
            callbacks=callbacks,
            # validation_data=(valX, valY),
            shuffle=False,
        )
        preds = model.predict(testX)

    elif experiment_params["mode"] == "landmark":
        last_index = 0
        for i, index in enumerate(test_indexes):
            print(f"Training for test index: {i+1}/{len(test_indexes)}")

            start_index = min(
                index
                - math.ceil((index - last_index) / model_params["batch_size"])
                * model_params["batch_size"],
                index
                - math.ceil(experiment_params["replay"] / model_params["batch_size"])
                * model_params["batch_size"],
            )
            if start_index < 0:
                start_index += model_params[
                    "batch_size"
                ]  # Fix per un problema per cui può andare in negativo per la prima data

            print(f"Training on sequences [{start_index}, {index}]")
            if model_params.get("dynamic_adj"):
                model.set_adj(
                    create_adj(
                        sequence=trainX[start_index:index],
                        replicate_nodes=experiment_params.get("replicate_nodes"),
                    )
                )

            model.fit(
                x=trainX[start_index:index],
                y=trainY[start_index:index],
                batch_size=model_params["batch_size"],
                epochs=model_params["epochs"],
                callbacks=callbacks,
                validation_data=None,
                shuffle=False,
                verbose=2,
            )

            pred = model.predict(np.expand_dims(testX[i], 0), batch_size=1)
            print(model.summary())
            preds.append(pred)

            last_index = index

            # Save model
            saved_models_path = os.path.join(
                experiment_params["path"],
                "saved_models",
            )
            if not os.path.exists(saved_models_path):
                os.makedirs(saved_models_path)
            model.save_weights(
                os.path.join(
                    saved_models_path,
                    f"{model_params['name']}-{dataset_params['name']}-{i}.h5",
                )
            )

        if experiment_params.get("interpretable"):
            for i in range(testX.shape[0]):
                print(f"Trying to obtain an explanation on {testX[i:i+1].shape}")
                # shap_values = explainer.shap_values(np.reshape(testX[i:i+1], (testX[i:i+1].shape[0], 19 * 7 * 11)))
                lime_values = explainer.explain_instance(
                    testX[i], lambda x: model.predict(x, batch_size=1), labels=range(19)
                )
                """
                with open(f'shap_values-n{node}-d{i}.pkl', 'wb') as f:  # A ogni iterazione salvo più roba, così se crasha prima almeno ho qualcosa
                    pickle.dump(shap_values, f, pickle.HIGHEST_PROTOCOL)
                    # np.save('shap_values.npy', np.stack(shap_values))
                shap.summary_plot(shap_values, testX[i:i+1], show=False)
                import matplotlib.pyplot as plt
                plt.savefig("shap_summary.svg", dpi=700)
                """
                with open(
                    f"lime_values-n{node}-d{i}.pkl", "wb"
                ) as f:  # A ogni iterazione salvo più roba, così se crasha prima almeno ho qualcosa
                    pickle.dump(lime_values, f, pickle.HIGHEST_PROTOCOL)
            """
            shap_values_all = explainer.shap_values(np.reshape(testX, (testX.shape[0], 19 * 7 * 11)))
            with open(f'shap_values-n{node}.pkl', 'wb') as f: # Salvo anche quelli calcolati su tutto (non so se ha senso)
                pickle.dump(shap_values_all, f, pickle.HIGHEST_PROTOCOL)
                # np.save('shap_values.npy', np.stack(shap_values))
            """
    elif experiment_params["mode"] == "sliding":
        for i, index in enumerate(test_indexes):
            print(f"Training for test index: {i + 1}/{len(test_indexes)}")
            model = build_model(
                model_params["name"],
                model_params.get("learning_rate"),
                model_params["loss"],
                dataset_params["n"],
                dataset_params["f"],
                dataset_params["h"],
                dataset_params["p"],
                adj,
                order=model_params.get("order"),
                summary=experiment_params.get("summary", None),
                gnn_type=model_params.get("gnn_type"),
            )
            if model_params.get("single_node"):
                model.set_node(node)

            start_index = (
                index
                - math.ceil(experiment_params["interval"] / model_params["batch_size"])
                * model_params["batch_size"]
            )  # arrotonda in eccesso per riempire il batch. potrebbe dare errore se interval + batch_size > test_indexes[0]
            if start_index < 0:
                start_index += model_params[
                    "batch_size"
                ]  # aggiunto per non farlo andare in negativo

            print(f"Training on sequences [{start_index}, {index}]")
            if model_params.get("dynamic_adj"):
                model.set_adj(
                    create_adj(
                        sequence=trainX[start_index:index],
                        replicate_nodes=experiment_params.get("replicate_nodes"),
                    )
                )
            model.fit(
                x=trainX[start_index:index],
                y=trainY[start_index:index],
                batch_size=model_params["batch_size"],
                epochs=model_params["epochs"],
                callbacks=callbacks,
                validation_data=None,
                shuffle=False,
            )
            pred = model.predict(
                np.expand_dims(testX[i], 0), batch_size=1
            )  # controllare se l'indice è giusto
            preds.append(pred)

    else:
        print("MODE is not valid.")

    preds = np.squeeze(preds)
    return preds


# def evaluate(preds, labels, node=None):
#     """
#     Compute the MAE, RMSE and SMAPE for each test date (sequence-wise), and then average across test dates.
#     """
#     print("preds: " + str(np.shape(preds)))
#     print("labels: " + str(np.shape(labels)))

#     assert (
#         preds.shape == labels.shape
#     )  # se modello multi-nodo [S,T,N], se singolo nodo [S,T]
#     axes = tuple(
#         range(1, preds.ndim)
#     )  # Assi su cui calcolare la media (tutti tranne l'asse delle sequenze)

#     # Con GCLSTM devo testare per un singolo nodo alla volta
#     if node:
#         labels = labels[..., node]

#     maes = np.mean(np.abs(np.subtract(preds, labels)), axis=axes)  # MAE
#     rmses = np.sqrt(np.mean(np.square(np.subtract(preds, labels)), axis=axes))  # RMSE
#     smapes = np.mean(
#         np.divide(np.abs(np.subtract(preds, labels)), np.add(preds, labels) / 2),
#         axis=axes,
#     )

#     print(f"maes: {maes.shape}, rmses: {rmses.shape}, smapes: {smapes.shape}")
#     print(f"Sequence-wise MAE: \n{maes}")
#     print(f"Sequence-wise RMSE: \n{rmses}")
#     print(f"Sequence-wise SMAPE: \n{smapes}")

#     avg_mae, avg_rmse, avg_smape = np.mean(maes), np.mean(rmses), np.mean(smapes)
#     print(f"Sequence-wise AVG MAE: {avg_mae:.4f}")
#     print(f"Sequence-wise AVG RMSE: {avg_rmse:.4f}")
#     print(f"Sequence-wise AVG SMAPE: {avg_smape:.4f}")

#     std_mae, std_rmse, std_smape = np.std(maes), np.std(rmses), np.std(smapes)
#     print(f"Sequence-wise STD MAE: {std_mae:.4f}")
#     print(f"Sequence-wise STD RMSE: {std_rmse:.4f}")
#     print(f"Sequence-wise STD SMAPE: {std_smape:.4f}")

#     return avg_mae, avg_rmse, avg_smape, std_mae, std_rmse, std_smape


# def plot_preds(
#     preds,
#     testY,
#     steps_per_plot=75,
#     nodes=5,
#     save_figures=False,
#     show_figs=False,
#     save_path=None,
# ):
#     """
#     steps_per_plot: amount of timesteps to show predictions and ground truth for each plot.
#     nodes: amount of nodes to show.
#     save_figures: whether to write plots on disk.
#     save_path: specify path where to write plots to.
#     """
#     S, P, _ = testY.shape
#     testY_plot = np.reshape(testY, (S * P, -1))  # Converte da [S,P,N] a [S*P,N]
#     preds_plot = np.reshape(preds, (S * P, -1))  # Converte da [S,P,N] a [S*P,N]

#     for i in range((S * P) // steps_per_plot):
#         plt.xlabel("Test steps")
#         plt.ylabel("Production")
#         plt.plot(
#             testY_plot[i * steps_per_plot : (i + 1) * steps_per_plot, :nodes],
#             "r-",
#             label="testY",
#         )
#         plt.plot(
#             preds_plot[i * steps_per_plot : (i + 1) * steps_per_plot, :nodes],
#             "b-",
#             label="preds",
#         )
#         plt.legend()
#         if save_figures:
#             plt.savefig(save_path + "-" + str(i) + ".png")
#         if show_figs:
#             plt.show()


# def write_results(
#     avg_mae,
#     avg_rmse,
#     avg_smape,
#     std_mae,
#     std_rmse,
#     std_smape,
#     training_time,
#     experiment_params,
#     model_params,
#     dataset_params,
#     results_path,
#     meta_path,
# ):
#     print("Saving evaluation results...")
#     with open(
#         os.sep.join(results_path.split(os.sep)[:-1])
#         + "/recap-"
#         + results_path.split(os.sep)[-1],
#         "w",
#     ) as recap_results_file:
#         recap_results_file.write(
#             "avg_mae,avg_rmse,avg_smape,std_mae,std_rmse,std_smape\n"
#         )
#         recap_results_file.write(
#             f"{avg_mae},{avg_rmse},{avg_smape},{std_mae},{std_rmse},{std_smape}\n"
#         )
#         recap_results_file.flush()
#     print("Evaluation results saved.")

#     # Questo pezzo dei metadati è da rivedere, molto obsoleto
#     print("Saving metadata...")
#     with open(meta_path, "a") as meta_file:
#         meta_file.write(f"Time: {(int)(time.time())}\n")
#         meta_file.write(f"Model name: {model_params['name']}\n")
#         meta_file.write(f"Dataset: {dataset_params['name']}\n")
#         meta_file.write(f"Mini-dataset mode: {experiment_params['mini_dataset']}\n")
#         meta_file.write(
#             f"Allow partial batches: {experiment_params['allow_partial_batches']}\n"
#         )
#         meta_file.write(f"Batch size: {model_params['batch_size']}\n")
#         meta_file.write(f"Epochs: {model_params['epochs']}\n")
#         meta_file.write(f"Learning rate: {model_params['learning_rate']}\n")
#         meta_file.write("Learning rate scheduler: ")
#         if model_params["lr_scheduler"]:
#             meta_file.write("reduceLROnPlateau; Factor=0.1, Patience=5, cooldown=0\n")
#         else:
#             meta_file.write("off\n")
#         meta_file.write(f"Evaluation mode: " + experiment_params["mode"] + "\n")
#         meta_file.write("Training time (seconds): " + str((int)(training_time)) + "\n")
#     print("Metadata saved.")


# def plot_results(maes, rmses, smapes, p):
#     plt.xlabel("Timestep")
#     plt.ylabel("MAE")
#     plt.xticks(ticks=range(p))
#     plt.grid(True)
#     plt.plot(maes[:-1], "r-", label="MAE for each step")
#     plt.plot(rmses[:-1], "b-", label="RMSE for each step")
#     plt.plot(
#         [maes[-1] for i in range(p)],
#         "r-",
#         linewidth=1,
#         label="Global avg MAE",
#     )
#     plt.plot(
#         [rmses[-1] for i in range(p)],
#         "b-",
#         linewidth=1,
#         label="Global avg RMSE",
#     )
#     plt.plot(
#         [np.mean(rmses[:-1]) for i in range(p)],
#         "b--",
#         linewidth=1,
#         label="Avg RMSE across steps",
#     )
#     plt.legend()
#     plt.show()

#     plt.xlabel("Timestep")
#     plt.ylabel("SMAPE")
#     plt.xticks(ticks=range(p))
#     plt.grid(True)
#     plt.plot(smapes[:-1], "g-", label="SMAPE for each step")
#     plt.plot(
#         [smapes[-1] for i in range(p)],
#         "g-",
#         linewidth=1,
#         label="Global avg SMAPE",
#     )
#     plt.legend()
#     plt.show()


class ExperimentRunner:
    """Experiment runner class."""

    def __init__(
        self,
        models,
        datasets,
        experiment_params,
        model_params,
        dataset_params,
        cl_args,
        name=None,
        prediction_seqs_file=None,
        **kwargs,
    ):
        """Initialize a new experiment runner.

        Positional arguments:
        models -- list of model names to run experiments for.
        datasets -- list of dataset names to run experiments on.
        results_path -- path where to save results to.

        Keyword arguments:
        model_params -- [Currently not supported]. JSON containing hyperparameter values for each model. If not specified, the current costants will be used.
        dataset_params -- [Currently not supported]. JSON containing configurations and setting for each dataset. If not specified, the current constants will be used.
        It contains information about the data, like number of nodes, features, sequences, the appropriate amount of history and prediction steps, the appropriate stride, and so on.
        """
        super(ExperimentRunner, self).__init__()

        self.models = models
        self.datasets = datasets

        print(vars(cl_args))
        print(model_params)
        self.experiment_params = dict_union(experiment_params, vars(cl_args))
        self.model_params = {
            model: dict_union(
                model_params["_default"], model_params[model], vars(cl_args)
            )
            for model in model_params
        }
        self.dataset_params = {
            dataset: dict_union(
                dataset_params["_default"], dataset_params[dataset], vars(cl_args)
            )
            for dataset in dataset_params
        }

        print(self.experiment_params)
        for k in self.model_params:
            print(self.model_params[k])
        for k in self.dataset_params:
            print(self.dataset_params[k])

        self.path = self.experiment_params["path"]
        self.time = "{:.0f}".format(time.time())
        self.name = name
        if not self.name:
            self.name = "exp-" + self.time
        self.run_already = False
        self.prediction_seqs_file = prediction_seqs_file
        self.tensorboard = True if kwargs.get("summaries_path") else False
        self.summaries_path = kwargs.get("summaries_path", None)

        self.run_path = os.path.join(self.path, "experiments", self.name)
        if not os.path.exists(self.run_path):
            os.mkdir(self.run_path)

    def info(self):
        return {
            "models": self.models,
            "datasets": self.datasets,
            "experiment_params": self.experiment_params,
            "model_params": self.model_params,
            "dataset_params": self.dataset_params,
            "time": self.time,
            "name": self.name,
            "run_already": self.run_already,
        }

    def set_starting_node(self, node):
        self.starting_node = node

    def set_prediction_seqs(self, file_type):
        self.prediction_seqs_file = file_type

    def enable_tensorboard(self, path):
        """Enable saving logs for TensorBoard visualization.
        Params
        ----------
        path: path where to save logs.
        """
        self.tensorboard = True
        self.summaries_path = path

    def disable_tensorboard(self):
        self.tensorboard = False
        self.summaries_path = None

    def is_finished(self, folder, run_name):
        """Returns true if the specified run is already complete and its results written to file, and false otherwise.
        Params
        ----------
        folder: folder name in the experiments folder (e.g. opt-SVD-LSTM-lightsource-landmark)
        run_name: includes both model, conf and dataset (e.g. SVD-LSTM-BS2LR1e-3LRSES-lightsource)
        """
        return os.path.exists(
            os.path.join(
                self.path,
                "experiments",
                folder,
                "recap-{}.csv".format(run_name),
            )
        )

    def run_all(self):
        if self.run_already:
            print(
                "This experiment ran already. Results are in the {} folder. Create a new runner for a new experiment.".format(
                    self.name
                )
            )
            return
        for dataset in self.datasets:
            for model in self.models:
                self._run(model, dataset, self.name)
        self.run_already = True

    def get_all_runs(self):
        return [
            f"{model}-{dataset}" for model in self.models for dataset in self.datasets
        ]

    # def collect_preds(self):
    #     """
    #     Join all predictions from each dataset in this experiment into a single csv file, containing a column for each model with its
    #     respective predictions, and a column (the first one) for the ground truth.
    #     """
    #     for dataset in self.datasets:
    #         df_path = os.path.join(
    #             self.path,
    #             "experiments",
    #             self.name,
    #             "{}.csv".format(dataset),
    #         )
    #         preds_df = pd.DataFrame()
    #         added_truth = False
    #         for model in self.models:
    #             df = pd.read_csv(
    #                 os.path.join(
    #                     self.path,
    #                     "experiments",
    #                     self.name,
    #                     "preds-{}-{}.csv".format(model, dataset),
    #                 )
    #             )
    #             if not added_truth:
    #                 truth = df.iloc[:, 1]
    #                 preds_df[df.columns[-2]] = truth
    #                 added_truth = True
    #             col = df.iloc[:, -1]
    #             preds_df[df.columns[-1]] = col
    #         preds_df.to_csv(df_path)

    #     # Join and summarize metrics of all models
    #     for dataset in self.datasets:
    #         df_path = os.path.join(
    #             self.path,
    #             "experiments",
    #             self.name,
    #             "recap-{}.csv".format(dataset),
    #         )
    #         recap_df = pd.DataFrame()
    #         recap_df["metrics"] = [
    #             "avg_mae",
    #             "avg_rmse",
    #             "avg_smape",
    #             "std_mae",
    #             "std_rmse",
    #             "std_smape",
    #         ]
    #         for model in self.models:
    #             df = pd.read_csv(
    #                 os.path.join(
    #                     self.path,
    #                     "experiments",
    #                     self.name,
    #                     "recap-{}-{}.csv".format(model, dataset),
    #                 )
    #             )
    #             recap_df[model] = df.iloc[0].to_numpy()
    #         recap_df.to_csv(df_path)

    def _run(self, model_name, dataset_name, run_name):
        """Run experiments and save them with the given name."""

        print(f"Run started for model {model_name} and dataset {dataset_name}...")

        model_params = self.model_params[model_name]
        dataset_params = self.dataset_params[dataset_name]

        # Defining paths
        # run_path = os.path.join(results_path, run_name + ".csv")
        # meta_path = os.path.join(results_path, run_name + ".meta")
        preds_path = os.path.join(
            self.run_path, f"preds-{model_name}-{dataset_name}.csv"
        )
        attw_path = os.path.join(self.run_path, f"attw-{model_name}-{dataset_name}.npy")
        int_path = os.path.join(self.run_path, f"int-{model_name}-{dataset_name}.npz")
        timeseries_file_path = os.path.join(
            self.path, "data", dataset_name, f"{dataset_name}.npz"
        )

        # Load test seqs
        self.experiment_params["pred_seqs_file_suffix"] = (
            self.experiment_params["test_file_suffix"]
            if self.experiment_params["prediction_seqs"] == "test"
            else self.experiment_params["val_file_suffix"]
        )
        test_file_path = os.path.join(
            self.path,
            "data",
            dataset_name,
            dataset_name
            + "_"
            + self.experiment_params["pred_seqs_file_suffix"]
            + ".npy",
        )

        # Prepare adj
        adj_file_path = os.path.join(
            self.path,
            "data",
            dataset_name,
            self.model_params[model_name]["adj_type"] + "-" + dataset_name + ".npy",
        )
        print(self.experiment_params.keys())
        adj = create_adj(
            adj_file_path, replicate_nodes=self.experiment_params["replicate_nodes"]
        )

        print(
            f"{Style.BRIGHT}Experiment params: \n{Style.RESET_ALL}"
            + "\n".join(
                [f"{k}: {self.experiment_params[k]}" for k in self.experiment_params]
            )
            + "\n"
        )
        print(
            f"{Style.BRIGHT}Model params: \n{Style.RESET_ALL}"
            + "\n".join([f"{k}: {model_params[k]}" for k in model_params])
            + "\n"
        )
        print(
            f"{Style.BRIGHT}Dataset params: \n{Style.RESET_ALL}"
            + "\n".join([f"{k}: {dataset_params[k]}" for k in dataset_params])
            + "\n"
        )

        # Data Loading and preprocessing
        trainX, trainY, testX, testY, test_indexes = create_timeseries(
            timeseries_file_path,
            test_file_path,
            self.experiment_params,
            model_params,
            dataset_params,
        )

        startTime = time.time()

        # Training and predicting
        if model_params["single_node"]:  # i single node model forse li devo togliere
            try:  # Ricarica le predizioni già fatte
                saved_preds = [
                    np.load(f"{preds_path}-{n}.npy")
                    for n in range(self.experiment_params["starting_node"])
                ]
                print(f"Loaded {len(saved_preds)} saved preds.")
            except:
                print(
                    "Could not load saved preds. Probably the starting node is beyond was is currently saved."
                )
                sys.exit(1)
            assert len(saved_preds) == self.experiment_params["starting_node"]

            multipreds = (
                saved_preds.copy()
            )  # Inizia con quelle già salvate, poi aggiungo le nuove
            for n in range(
                self.experiment_params["starting_node"], dataset_params["n"]
            ):
                print("Loop iteration started")

                # Se --distributed, devo runnare solo sui nodi di competenza di questo worker
                if not self.conf.distributed or n % hvd.size() == hvd.rank():
                    print("Starting training for node " + str(n))
                    model = build_model(
                        self.experiment_params, model_params, dataset_params
                    )
                    print("Model built")

                    model.set_node(n)
                    print(f"Node {n} set")

                    preds = train_and_predict(
                        model,
                        trainX,
                        trainY,
                        testX,
                        testY,
                        test_indexes,
                        self.experiment_params,
                        model_params,
                        dataset_params,
                        adj,
                        node=n,
                    )
                    print("Fit ended")
                    multipreds.append(preds)
                    np.save(f"{preds_path}-{n}.npy", preds)
                    if (
                        self.conf.distributed and hvd.rank() != 0
                    ):  # Se è un worker, deve mandare il file al master
                        file = f"{preds_path}-{n}.npy"  # @TODO è brutto hardcoded
                        print(
                            f"Attempting to scp {file} from worker {hvd.rank()} to master"
                        )
                        ssh = SSHClient()
                        ssh.load_system_host_keys()
                        ssh.connect(
                            "147.9.188.154", username="massimiliano"
                        )  # @TODO l'IP del master deve essere configurabile, ora è hardcoded
                        with SCPClient(ssh.get_transport()) as scp:
                            scp.put(
                                file,
                                os.path.join(
                                    self.path,
                                    "experiments",
                                    f"{preds_path}-{n}.npy",
                                ),
                            )
                print("Loop iteration ended")
        else:
            model = build_model(
                self.experiment_params,
                model_params,
                dataset_params,
                adj=adj,
            )
            preds = train_and_predict(
                model,
                trainX,
                trainY,
                testX,
                testY,
                test_indexes,
                self.experiment_params,
                model_params,
                dataset_params,
                adj,
            )

        # Il master aspetta che tutti i worker abbiano finito
        if (
            self.experiment_params.get("distributed") and hvd.rank() == 0
        ):  # Solo il master deve aspettare
            check_interval = 10
            while True:
                ls = [
                    file
                    for file in os.listdir(os.path.join(self.results_path, self.name))
                    if file.startswith(f"preds-{run_name}.csv-")
                ]  # Prendi i file preds (run che hanno finito)
                print(ls)
                if (
                    len(ls)
                    == dataset_params["n"] - self.experiment_params["starting_node"]
                ):  # Se ci sono tutti (ovvero, se hanno finito tutti i worker)
                    break
                print(
                    f"{dataset_params['nodes'] - len(ls)} run predictions were not found by master in the output folder. Checking again in {check_interval} seconds."
                )
                time.sleep(check_interval)  # Altrimenti ricontrolla ogni 10 secondi

        if (
            self.experiment_params.get("distributed") and hvd.rank() != 0
        ):  # I worker possono anche terminare qui
            print(
                f"Worker {hvd.rank()} has finished his work, pushed to master, and is now terminating."
            )
            sys.exit(0)

        # Log execution time
        endTime = time.time()
        trainingTime = endTime - startTime
        print("Training complete in {:d} seconds.".format(int(trainingTime)))

        # Save attention weights
        if self.experiment_params["save_attention_weights"]:
            attw = model.layers[2].decoder.attw
            np.save(attw_path, attw.numpy())
            if self.experiment_params["show_figures"]:
                loaded_attw = np.load(attw_path)
                heatmap = np.mean(loaded_attw, axis=(0, 2))
                ax = sns.heatmap(heatmap, linewidth=0.5)
                plt.show()  # Asse x: h, Asse y: p

            # Prendo in prestito quest'if momentaneamente, in realtà bisogna aggiungere un altro parametro nel json save_hidden_states
            enc_hidden_states_path = os.path.join(
                self.results_path, self.name, "enc-" + run_name + ".npy"
            )
            dec_hidden_states_path = os.path.join(
                self.results_path, self.name, "dec-" + run_name + ".npy"
            )
            enc_hidden_states = model.layers[2].log_encoder_states
            dec_hidden_states = model.layers[2].log_decoder_states
            np.save(enc_hidden_states_path, enc_hidden_states.numpy())
            np.save(dec_hidden_states_path, dec_hidden_states.numpy())

        if self.experiment_params.get("interpretable"):
            (
                gcnmaps,
                convmaps,
                summaries,
            ) = (
                model.get_interpretation()
            )  # per ora l'interfaccia get_interpretation() può restituire qualsiasi cosa.
            np.savez(int_path, gcnmaps=gcnmaps, convmaps=convmaps, summaries=summaries)

        # Saving predictions. The CSV file rows are ALWAYS nodes first, timestep second, test sequence last.
        # Meaning that grouping N rows at once you abstract nodes, grouping T*N rows at once you abstract timesteps and nodes.
        # If you want to select predictions for node k you select rows k, k+N, k+2N, ..., k+STN.
        preds_df = pd.DataFrame()
        preds_df["truth"] = testY.flatten(order="C")
        if model_params["single_node"]:
            preds_merged = np.empty(
                (dataset_params["n"] * multipreds[0].size,),
                dtype=multipreds[0].dtype,
            )
            for i, preds in enumerate(multipreds):
                flattened_preds = preds.flatten()
                preds_merged[i :: dataset_params["n"]] = flattened_preds
            preds_df[model_name] = preds_merged.flatten(order="C")
        else:
            preds_df[model_name] = preds.flatten(order="C")
        preds_df.to_csv(preds_path)

        # # Plotting predictions
        # plot_preds(preds, testY, show_figs=self.experiment_params["show_figures"])

        # # Model Evaluation
        # print("Evaluating model...")
        # if model_params["single_node"]:
        #     preds = np.stack(multipreds, axis=2)
        #     print(
        #         f"stack(multipreds, axis=2): {np.shape(multipreds)} -> {np.shape(np.stack(multipreds, axis=2))}"
        #     )

        # avg_mae, avg_rmse, avg_smape, std_mae, std_rmse, std_smape = evaluate(
        #     preds, testY
        # )
        # print(
        #     f"Writing results:\n\tavg_mae: {avg_mae:.4f}\n\tavg_rmse: {avg_rmse:.4f}\n\tavg_smape: {avg_smape:.4f}\n\tstd_mae: {std_mae:.4f}\n\tstd_rmse: {std_rmse:.4f}\n\tstd_smape: {std_smape:.4f}"
        # )
        # write_results(
        #     avg_mae,
        #     avg_rmse,
        #     avg_smape,
        #     std_mae,
        #     std_rmse,
        #     std_smape,
        #     trainingTime,
        #     self.experiment_params,
        #     model_params,
        #     dataset_params,
        #     run_path,
        #     meta_path,
        # )

        # if self.experiment_params["show_figures"]:
        #     plot_results(maes, rmses, smapes, dataset_params["p"])

        # print("Evaluating complete.")

        return model, preds  # For debugging purposes


def load_params(experiment_params_path, model_params_path, dataset_params_path):
    # Extract default params from files
    with open(experiment_params_path, "r") as f:
        experiment_params = json.load(f)
    with open(model_params_path, "r") as f:
        model_params = json.load(f)
    with open(dataset_params_path, "r") as f:
        dataset_params = json.load(f)

    return experiment_params, model_params, dataset_params


def main():
    args = parse_args()

    args.path = "~/research/" if args.distributed else ".."

    gpu_config(args.distributed)

    # Load config files
    experiment_params_path = os.path.join(
        args.path, "experiments", ".config", "experiment_params.json"
    )
    model_params_path = os.path.join(
        args.path, "experiments", ".config", "model_params_prototypes.json"
    )
    dataset_params_path = os.path.join(
        args.path, "experiments", ".config", "dataset_params.json"
    )
    experiment_params, model_params, dataset_params = load_params(
        experiment_params_path, model_params_path, dataset_params_path
    )

    runner = ExperimentRunner(
        [args.model],
        [args.dataset],
        experiment_params,
        model_params,
        dataset_params,
        args,
        name=args.run_name,
    )

    # <----- All of these must be done internally in ExperimentRunner, according to self.cl_args
    # if args.mode:
    #     print("Mode (Overridden): " + args.mode)
    #     runner.override_mode = args.mode

    # if args.wsize:
    #     print("Window size (Overridden): " + str(args.wsize))
    #     runner.override_window = args.wsize

    # runner.set_starting_node(args.starting_node)

    # runner.set_prediction_seqs(args.prediction_seqs)

    # if args.tensorboard:
    #     runner.enable_tensorboard(
    #         os.path.join(
    #             args.path,
    #             "log",
    #             f"{args.model}-{args.dataset}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-summaries",
    #         )
    #     )

    # ----->

    if not args.collect_only:
        runner.run_all()

    # runner.collect_preds()


if __name__ == "__main__":
    main()
