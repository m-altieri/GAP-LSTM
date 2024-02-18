# Arg Parsing
import argparse

# Math
import math
import numpy as np
import pandas as pd
import scipy

# Utility
import time
import datetime
from datetime import date, timedelta
import os
import sys
import random
import csv
import json
import utils.math, utils.logging, utils.sequence
import pickle

# PyTorch
import torch
import torch.nn as nn
import gwnet_util

# Models
sys.path.append('/lustrehome/altieri/research/src/models')
sys.path.append('./models')
from models.GraphWaveNet import gwnet

argparser = argparse.ArgumentParser(description='Run the experiments.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument('model', action='store', help='select the model to run')
argparser.add_argument('dataset', action='store', help='select the dataset to run the model on')
argparser.add_argument('-c', '--conf', action='store', help='select the model configuration to run; not setting it has the same effect of setting -a')
argparser.add_argument('-a', '--all', action='store_const', const=True, help='run all configurations; if this flag is set, -c is ignored')
argparser.add_argument('-o', '--output', action='store', help='set the name of the output folder')
argparser.add_argument('-m', '--mode', action='store', choices=['landmark', 'sliding'], help='select the evaluation mode; default is landmark')
argparser.add_argument('-w', '--wsize', action='store', type=int, choices=[30, 60, 90], help='select the window size; if -m is landmark, this is ignored')
argparser.add_argument('-n', '--starting-node', action='store', default=0, type=int, help='choose the node you want to start training from, and skip the previous ones. This only affects single-node models.')
argparser.add_argument('-d', '--prediction-seqs', action='store', required=True, choices=['test', 'val'], help='choose the file containing prediction sequence indexes that you want to use')
argparser.add_argument('-e', '--epochs', action='store', type=int, help='override the number of epochs in the config file')
argparser.add_argument('-b', '--batch-size', action='store', type=int, help='override the batch size in the config file')
argparser.add_argument('-l', '--learning-rate', action='store', type=float, help='override the learning rate in the config file')
argparser.add_argument('--recas', action='store_const', const=True, help='set the correct path for execution on recas')
argparser.add_argument('--collect-only', action='store_const', const=True, help='skip to results collection')
argparser.add_argument('--interpretable', action='store_const', const=True, help='attempt to retrieve interpretable information from the model')
argparser.add_argument('--skip-finished', action='store_const', const=True, help='skip configurations whose results have already been written. Only has effect if -a is set')
argparser.add_argument('--tensorboard', action='store_const', const=True, help='log information to tensorboard')  # @TODO molto rudimentale; per ora semi-funziona solo per GCLSTM
argparser.add_argument('--distributed', action='store_const', const=True, help='run the experiment in distributed mode. use only with horovodrun.')
argparser.add_argument('--adj', action='store', choices=['ones', 'closeness', 'corr'], help='run the experiment in distributed mode. use only with horovodrun.')
args = argparser.parse_args()

PATH = '/lustrehome/altieri/research' if args.recas else '..'

# Create logger
log_file_name = '{}-{}-{}.log'.format(args.model, args.dataset, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
logger = utils.logging.create_logger(os.path.join(PATH, 'log', log_file_name))
logger.info('Args: {}'.format(args))
logger.info(os.listdir(PATH))

def create_timeseries(timeseries_path, test_file_path, h, p, stride, mini_dataset=False, test_source='file', train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, buffer=30, allow_partial_batches=True, batch_size=2, mode='landmark'):
    logger.warning('Loading data...')

    data = np.load(timeseries_path, allow_pickle=True)['data']
    if mini_dataset:
        data = data[:len(data)/h//10*h, :, :]

    logger.warning('Data loaded with shape ' + str(data.shape) + '.')

    # Splitta le timeseries in tanti blocchi di lunghezza HISTORY_STEPS + PREDICTION_STEPS
    logger.warning('Converting timeseries into windows of history steps and prediction steps...')
    X, Y = utils.sequence.obs2seqs(data, h, p, stride)

    sequences, steps, nodes, features = X.shape
    logger.warning('Conversion complete: ' + str(sequences) + ' sequences produced from ' + str(data.shape[0]) + ' observations.')

    # Calcolo dimensione di train, val e test set
    logger.warning('Calculating training, validation and test splits (' + str(train_ratio) + ', ' + str(val_ratio) + ', ' + str(test_ratio) + ')...')
    training_seqs, val_seqs, test_seqs = utils.math.divide_into_3_integers(sequences, train_ratio, val_ratio, test_ratio)

    if test_source == 'random':
        # Split casuale delle sequenze in train, val e test. Qui vengono solo selezionate le sequenze perchè mi devo salvare gli indici, lo split vero viene dopo.
        logger.warning('Shuffling data with seed ' + str(seed) + '.')
        indexes = np.arange(sequences)
        rnd_above_buffer_perm = shuffle(indexes[buffer:], random_state=seed)
        val_indexes = rnd_above_buffer_perm[: val_seqs]
        test_indexes = rnd_above_buffer_perm[val_seqs: val_seqs + test_seqs]
        perm = shuffle(indexes, random_state=seed)
        train_indexes = np.setdiff1d(perm, np.concatenate((val_indexes, test_indexes)), assume_unique=True)

    elif test_source == 'file':
        test_indexes = np.load(test_file_path) - h // stride  # l'indice è da quando parte
        train_indexes = np.setdiff1d(np.arange(sequences), test_indexes, assume_unique=True)
        val_indexes = []

    # Se ALLOW_PARTIAL_BATCHES == False, tronca gli ultimi indici per far sì che il totale sia un multiplo di BATCH_SIZE
    if not allow_partial_batches:
        logger.warning('Making number of samples a multiple of batch size (' + str(batch_size) + ')...')
        train_indexes = train_indexes[: len(train_indexes) // batch_size * batch_size]
        val_indexes = val_indexes[: len(val_indexes) // batch_size * batch_size]
        test_indexes = test_indexes[: len(test_indexes) // batch_size * batch_size]
        logger.warning('Number of samples is a multiple of batch size')
    else:
        logger.warning('Skipped truncation of partial batches.')

    # Riordina gli indici
    train_indexes.sort()
    val_indexes.sort()
    test_indexes.sort()

    # Uso tutte le date per il training, anche quelle di test (dopo che le ho usate per predire)
    trainX = X
    trainY = Y
    valX = X[val_indexes if mode == 'default' else []]
    valY = Y[val_indexes if mode == 'default' else []]
    testX = X[test_indexes]
    testY = Y[test_indexes]

    # Dall'Y prendo solo la label
    trainY = trainY[..., 0]
    valY = valY[..., 0]
    testY = testY[..., 0]

    logger.warning('Loading and preprocessing complete with shapes:' +
                   '\n\tTrainX shape: ' + str(trainX.shape) +
                   '\n\tTrainY shape: ' + str(trainY.shape) +
                   '\n\tValX shape: ' + str(valX.shape) +
                   '\n\tValY shape: ' + str(valY.shape) +
                   '\n\tTestX shape: ' + str(testX.shape) +
                   '\n\tTestY shape: ' + str(testY.shape))

    return trainX, trainY, valX, valY, testX, testY, train_indexes, val_indexes, test_indexes


def create_adj(adj_path=None, sequence=None, show=False, save=False):
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
    assert (adj_path is None) + (sequence is None) == 1  # One must be specified, the other must be None

    if adj_path:
        logger.warning('Loading adjacency matrix from ' + adj_path + '...')
        adj = np.load(adj_path).astype(np.float32)
    elif sequence is not None:
        logger.warning('Computing adjacency matrix from sequence of shape {}...'.format(np.shape(sequence)))
        B, T, N, F = np.shape(sequence)
        sequence = sequence[..., 0]  # [B,T,N]
        adj = np.corrcoef(np.reshape(sequence, (B * T, N)), rowvar=False)  # Variables are nodes, which are on the columns
    else:
        raise ValueError('Exactly one between adj_path and sequence must be specified.')

    D = np.zeros_like(adj)
    for row in range(len(D)):
        D[row, row] = np.sum(adj[row])  # Degree matrix (D_ii = sum(adj[i,:]))
    sqinv_D = np.sqrt(np.linalg.inv(D))  # Calcola l'inversa e la splitta in due radici
    adj = np.matmul(sqinv_D, np.matmul(adj, sqinv_D))

    if show:
        plt.figure(figsize=(12, 12))
        sns.heatmap(adj)
    if save:
        plt.savefig(os.path.join(PATH, 'heatmaps', 'corr-' + datetime.datetime.now().strftime("%H%M%S") + '.png'))
    logger.warning('Adjacency matrix loaded with shape ' + str(adj.shape) + '.')

    # Checking for NaN or infinite values
    if np.isnan(adj).any():
        adj = np.nan_to_num(adj, nan=0)
        logger.critical('Adjacency matrix contains NaN values. They have been replaced with 0.')
    if np.isnan(adj).any() or np.isinf(adj).any():
        logger.critical('Adjacency matrix is nan or infinite:\n' + str(adj) + '\nTerminating due to critical error.')
        sys.exit(1)

    return adj


def build_model(model_type, loss, nodes, features=None, history_steps=None, prediction_steps=None, adj=None, order=4, **kwargs):
    logger.warning('Starting model construction for model type {}....'.format(model_type))

    model = gwnet(torch.device('cuda'), nodes)
    model.to(torch.device('cuda'))

    return model


def train_and_predict(model, trainX, trainY, valX, valY, testX, testY, test_indexes, mode, optimizer, loss, nodes, features, history_steps, prediction_steps, adj,
                      batch_size, epochs, replay, interval, model_name, checkpoint=False, checkpoint_path=None, lr_scheduler=False, lrs_monitor=None,
                      lrs_factor=None, lrs_patience=None, early_stopping=False, es_monitor=None, es_patience=None, es_restore=None, single_node_model=False, node=0, dynamic_adj=False, order=4, **kwargs):
    logger.info('Training starting with:' +
                f'Training X shape: {trainX.shape}\n\t' +
                f'Training Y shape: {trainY.shape}\n\t' +
                f'Validation X shape: {valX.shape}\n\t' +
                f'Validation Y shape: {valY.shape}\n\t' +
                f'Test X shape: {testX.shape}\n\t' +
                f'Test Y shape: {testY.shape}\n\t' +
                f'Batch size: {batch_size}\n\t' +
                f'Epochs: {epochs}')

    preds = []
    
    last_index = 0
    for i, index in enumerate(test_indexes):
        logger.warning('Training for test index: %d/%d' % (i + 1, len(test_indexes)))

        start_index = min(index - math.ceil((index - last_index) / batch_size) * batch_size, index - math.ceil(replay / batch_size) * batch_size)
        if start_index < 0: start_index += batch_size  # Fix per un problema per cui può andare in negativo per la prima data

        logger.warning('Training on [%d, %d]' % (start_index, index))
        if dynamic_adj: model.set_adj(create_adj(sequence=trainX[start_index:index]))
        
        x = trainX[start_index: index]
        y = trainY[start_index: index]

        input = torch.Tensor(x)
        real_val = torch.Tensor(y)

        ###  TRAIN  ###
        model.train()
        optimizer.zero_grad()
        input = nn.functional.pad(input, (1,0,0,0))
        output = model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = output  # no scaler

        loss = gwnet_util.masked_mae(predict, real, 0.0)  # è quella che usa
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)  # clip = 5
        optimizer.step()

        
        preds.append(predict)
            
        #model.fit(, batch_size=batch_size, epochs=epochs, callbacks=callbacks, validation_data=None, shuffle=False)
        #pred = model.predict(np.expand_dims(testX[i], 0), batch_size=1)
        #preds.append(pred)

        last_index = index

    preds = np.squeeze(preds)
    return preds


def evaluate(preds, labels, node=None):
    """
  Compute the MAE, RMSE and SMAPE for each test date (sequence-wise), and then average across test dates.
  """
    logger.info('preds: ' + str(np.shape(preds)))
    logger.info('labels: ' + str(np.shape(labels)))

    assert preds.shape == labels.shape  # se modello multi-nodo [S,T,N], se singolo nodo [S,T]
    axes = tuple(range(1, preds.ndim))  # Assi su cui calcolare la media (tutti tranne l'asse delle sequenze)

    # Con GCLSTM devo testare per un singolo nodo alla volta
    if node:
        labels = labels[..., node]

    maes = np.mean(np.abs(np.subtract(preds, labels)), axis=axes)  # MAE
    rmses = np.sqrt(
        np.mean(
            np.square(
                np.subtract(preds, labels)
            ), axis=axes
        )
    )  # RMSE
    smapes = np.mean(
        np.divide(
            np.abs(
                np.subtract(preds, labels)
            ),
            np.add(preds, labels) / 2
        ), axis=axes
    )  # SMAPE

    # maes, rmses e smapes dovrebbero essere tutti shape (S,)
    logger.info(f'maes: {maes.shape}, rmses: {rmses.shape}, smapes: {smapes.shape}')
    logger.info(f'Sequence-wise MAE: \n{maes}')
    logger.info(f'Sequence-wise RMSE: \n{rmses}')
    logger.info(f'Sequence-wise SMAPE: \n{smapes}')

    avg_mae, avg_rmse, avg_smape = np.mean(maes), np.mean(rmses), np.mean(smapes)
    logger.info(f'Sequence-wise AVG MAE: {avg_mae:.4f}')
    logger.info(f'Sequence-wise AVG RMSE: {avg_rmse:.4f}')
    logger.info(f'Sequence-wise AVG SMAPE: {avg_smape:.4f}')

    std_mae, std_rmse, std_smape = np.std(maes), np.std(rmses), np.std(smapes)
    logger.info(f'Sequence-wise STD MAE: {std_mae:.4f}')
    logger.info(f'Sequence-wise STD RMSE: {std_rmse:.4f}')
    logger.info(f'Sequence-wise STD SMAPE: {std_smape:.4f}')

    return avg_mae, avg_rmse, avg_smape, std_mae, std_rmse, std_smape


def write_results(avg_mae, avg_rmse, avg_smape, std_mae, std_rmse, std_smape, results_path, meta_path, dataset_name, model_name, mini_dataset, features, allow_partial_batches,
                  history_steps, prediction_steps, stride, batch_size, epochs, learning_rate, lr_scheduler, mode, training_time):
    logger.warning('Saving evaluation results...')
    with open(os.sep.join(results_path.split(os.sep)[:-1]) + '/recap-' + results_path.split(os.sep)[-1], 'w') as recap_results_file:
        recap_results_file.write('avg_mae,avg_rmse,avg_smape,std_mae,std_rmse,std_smape\n')
        recap_results_file.write(f'{avg_mae},{avg_rmse},{avg_smape},{std_mae},{std_rmse},{std_smape}\n')
        recap_results_file.flush()
    logger.warning("Evaluation results saved.")

    # Questo pezzo dei metadati è da rivedere, molto obsoleto
    logger.warning("Saving metadata...")
    with open(meta_path, 'a') as meta_file:
        meta_file.write('Time: ' + str((int)(time.time())) + '\n')
        meta_file.write('Model name: ' + model_name + '\n')
        meta_file.write('Dataset: ' + dataset_name + '\n')
        meta_file.write('Mini-dataset mode: ' + str(mini_dataset) + '\n')
        #meta_file.write('Num of features: ' + str(features) + '\n')
        meta_file.write('Allow partial batches: ' + str(allow_partial_batches) + '\n')
        meta_file.write('History steps: ' + str(history_steps) + '\n')
        meta_file.write('Prediction steps: ' + str(prediction_steps) + '\n')
        meta_file.write('Stride: ' + str(stride) + '\n')
        meta_file.write('Batch size: ' + str(batch_size) + '\n')
        meta_file.write('Epochs: ' + str(epochs) + '\n')
        meta_file.write('Learning rate: ' + str(learning_rate) + '\n')
        meta_file.write('Learning rate scheduler: ')
        if lr_scheduler:
            meta_file.write('reduceLROnPlateau; Factor=0.1, Patience=5, cooldown=0\n')
        else:
            meta_file.write('off\n')
        meta_file.write('Evaluation mode: ' + mode + '\n')
        meta_file.write('Training time (seconds): ' + str((int)(training_time)) + '\n')
    logger.warning("Metadata saved.")

class ExperimentRunner():
    """Experiment runner class."""

    def __init__(self, models, datasets, results_path, experiment_params, model_params, dataset_params, name=None, prediction_seqs_file=None, **kwargs):
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
        self.results_path = results_path
        self.experiment_params = experiment_params
        self.model_params = model_params
        self.dataset_params = dataset_params
        self.override_mode = None
        self.override_window = None
        self.starting_node = 0
        self.time = '{:.0f}'.format(time.time())
        self.name = name
        if not self.name: self.name = 'exp-' + self.time
        self.run_already = False
        self.prediction_seqs_file = prediction_seqs_file
        self.summaries_path = kwargs.get('summaries_path', None)

    def info(self):
        return {'models': self.models,
                'datasets': self.datasets,
                'prediction_dates': self.prediction_seqs_file,
                'override_mode': self.override_mode,
                'override_window': self.override_window,
                'starting_node': self.starting_node,
                'time': self.time,
                'name': self.name,
                'run_already': self.run_already}

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

    def is_finished(folder, run_name):
        """Returns true if the specified run is already complete and its results written to file, and false otherwise.
    Params
    ----------
    folder: folder name in the experiments folder (e.g. opt-SVD-LSTM-lightsource-landmark)
    run_name: includes both model, conf and dataset (e.g. SVD-LSTM-BS2LR1e-3LRSES-lightsource)
    """
        return os.path.exists(os.path.join(PATH, 'experiments', folder, 'recap-{}.csv'.format(run_name)))

    def run_all(self):
        if self.run_already:
            logger.critical('This experiment ran already. Results are in the {} folder. Create a new runner for a new experiment.'.format(self.name))
            return
        for dataset in self.datasets:
            for model in self.models:
                self._run(model, dataset, '{}-{}'.format(model, dataset))
        self.run_already = True

    def get_all_runs(self):
        return [f'{model}-{dataset}' for model in self.models for dataset in self.datasets]

    def collect_preds(self):
        """
    Join all predictions from each dataset in this experiment into a single csv file, containing a column for each model with its
    respective predictions, and a column (the first one) for the ground truth.
    """
        for dataset in self.datasets:
            df_path = os.path.join(PATH, 'experiments', self.name, '{}.csv'.format(dataset))
            preds_df = pd.DataFrame()
            added_truth = False
            for model in self.models:
                df = pd.read_csv(os.path.join(PATH, 'experiments', self.name, 'preds-{}-{}.csv'.format(model, dataset)))
                if not added_truth:
                    truth = df.iloc[:, 1]
                    preds_df[df.columns[-2]] = truth
                    added_truth = True
                col = df.iloc[:, -1]
                preds_df[df.columns[-1]] = col
            preds_df.to_csv(df_path)

        # Join and summarize metrics of all models
        for dataset in self.datasets:
            df_path = os.path.join(PATH, 'experiments', self.name, 'recap-{}.csv'.format(dataset))
            recap_df = pd.DataFrame()
            recap_df['metrics'] = ['avg_mae', 'avg_rmse', 'avg_smape', 'std_mae', 'std_rmse', 'std_smape']
            for model in self.models:
                df = pd.read_csv(os.path.join(PATH, 'experiments', self.name, 'recap-{}-{}.csv'.format(model, dataset)))
                recap_df[model] = df.iloc[0].to_numpy()
            recap_df.to_csv(df_path)

    def _run(self, model_name, dataset_name, run_name):
        """Run experiments and save them with the given name."""

        if not os.path.exists(os.path.join(self.results_path, self.name)): os.mkdir(os.path.join(self.results_path, self.name))
        run_path = os.path.join(self.results_path, self.name, run_name + '.csv')
        meta_path = os.path.join(self.results_path, self.name, run_name + '.meta')
        preds_path = os.path.join(self.results_path, self.name, 'preds-' + run_name + '.csv')
        attw_path = os.path.join(self.results_path, self.name, 'attw-' + run_name + '.npy')
        int_path = os.path.join(self.results_path, self.name, 'int-' + run_name + '.npz')

        logger.warning('Training starting for model {} and dataset {}....'.format(model_name, dataset_name))

        # Find params of given model and dataset
        model_p = self.model_params['default'].copy()  # Initialize with default params
        for model in self.model_params['models']:
            if model['name'] == model_name:
                # Update with model-specific params
                for param in model.keys():
                    model_p[param] = model[param]
        dataset_p = self.dataset_params['default'].copy()  # Initialize with default params
        for dataset in self.dataset_params['datasets']:
            if dataset['name'] == dataset_name:
                # Update with dataset-specific params
                for param in dataset.keys():
                    dataset_p[param] = dataset[param]

        timeseries_file_path = os.path.join(PATH, 'data', dataset_name, dataset_name + '.npz')

        pred_seqs_file_suffix = None
        if self.prediction_seqs_file == 'test':
            pred_seqs_file_suffix = experiment_params['test_file_suffix']
        if self.prediction_seqs_file == 'val':
            pred_seqs_file_suffix = experiment_params['val_file_suffix']
        if not pred_seqs_file_suffix: raise AttributeError('Can\'t find a value for prediction_seqs_file. If the program was run from command line, probably no value was passed for flag -d.')
        experiment_params['pred_seqs_file_suffix'] = pred_seqs_file_suffix
        test_file_path = os.path.join(PATH, 'data', dataset_name, dataset_name + '_' + experiment_params['pred_seqs_file_suffix'] + '.npy')
        history_steps = dataset_p['history_steps']
        prediction_steps = dataset_p['prediction_steps']
        stride = dataset_p['stride']
        if args.adj: 
            model_p['adj_type'] = args.adj
        adj_type = model_p['adj_type']
        nodes = dataset_p['nodes']
        features = dataset_p['features'] if 'features' in dataset_p else None

        single_node_model = model_p['single_node']
        # if args.distributed: single_node_model = False # @TODO *** escamotage, funziona solo temporaneamente perchè adesso
        # l'unico modello supportato da --distributed è un single node model, ma normalmente il meccanismo deve essere automatizzato:
        # se il modello è single node, diventa multi node; se il modello è multi node, la distribuzione avviene con gradient sharing.
        # ancora meglio, implementare un meccanismo dove si seleziona la distribution strategy da utilizzare
        batch_size = model_p['batch_size']
        if args.batch_size:
            batch_size = args.batch_size
            logger.info(f'Using {batch_size} batch size.')
        save_attention_weights = model_p['save_attention_weights']
        learning_rate = model_p['learning_rate']
        if args.learning_rate:
            learning_rate = args.learning_rate
            logger.info(f'Using {learning_rate} learning rate.')

        loss = gwnet_util.masked_mae
        epochs = model_p['epochs']
        if args.epochs:
            epochs = args.epochs
            logger.info(f'Using {epochs} epochs.')

        checkpoint = model_p['checkpoint']
        checkpoint_path = None
        if checkpoint: checkpoint_path = os.path.join(PATH, 'checkpoints', model_name)
        lr_scheduler = model_p['lr_scheduler']
        lrs_monitor, lrs_factor, lrs_patience = None, None, None
        if lr_scheduler:
            lrs_monitor = model_p['lrs_monitor']
            lrs_factor = model_p['lrs_factor']
            lrs_patience = model_p['lrs_patience']
        early_stopping = model_p['early_stopping']
        es_monitor, es_patience, es_restore = None, None, None
        if early_stopping:
            es_monitor = model_p['es_monitor']
            es_patience = model_p['es_patience']
            es_restore = model_p['es_restore']
        order = None
        if 'order' in model_p:
            order = model_p['order']

        show_figures = experiment_params['show_figures']
        mini_dataset = experiment_params['mini_dataset']
        test_seqs_source = experiment_params['test_seqs_source']
        buffer = experiment_params['buffer']
        replay = experiment_params['replay']
        allow_partial_batches = experiment_params['allow_partial_batches']

        if self.override_mode:
            experiment_params['mode'] = self.override_mode
        mode = experiment_params['mode']

        if self.override_window:
            experiment_params['interval'] = self.override_window
        interval = experiment_params['interval']

        save_figures = experiment_params['save_figures']
        dynamic_adj = model_p['dynamic_adj'] if not args.adj else False

        adj_file_path = os.path.join(PATH, 'data', dataset_name, adj_type + '-' + dataset_name + '.npy')
        adj = create_adj(adj_file_path)

        logger.info('Info: {}'.format(self.info()))
        logger.warning('Model params: {}\n'.format(model_p))
        logger.warning('Dataset params: {}\n'.format(dataset_p))
        logger.warning('Experiment params: {}\n'.format(experiment_params))

        # Data Loading and preprocessing
        trainX, trainY, valX, valY, testX, testY, train_indexes, val_indexes, test_indexes = create_timeseries(timeseries_file_path, test_file_path, history_steps, prediction_steps, stride, mini_dataset=mini_dataset,
                                                                                                               test_source=test_seqs_source, buffer=buffer, allow_partial_batches=allow_partial_batches,
                                                                                                               batch_size=batch_size, mode=mode)

        logger.info(f'Single-node model: {single_node_model}')

        startTime = time.time()

        # Training and predicting
        if single_node_model:  # i single node model forse li devo togliere

            try:  # Ricarica le predizioni già fatte
                saved_preds = [np.load(f'{preds_path}({n}).npy') for n in range(self.starting_node)]
                logger.info(f'Loaded {len(saved_preds)} saved preds.')
            except:
                logger.error('Could not load saved preds. Probably the starting node is beyond was is currently saved.')
                sys.exit(1)
            assert len(saved_preds) == self.starting_node

            multipreds = saved_preds.copy()  # Inizia con quelle già salvate, poi aggiungo le nuove
            for n in range(self.starting_node, nodes):
                logger.warning('Starting training for node ' + str(n))
                model = build_model(model_p['alias'], loss, nodes, features, history_steps, prediction_steps, adj=adj, order=order)
                optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters(), weight_decay=0.0001)
                model.set_node(n)
                preds = train_and_predict(model, trainX, trainY, valX, valY, testX, testY, test_indexes, mode, optimizer, loss, nodes, features, history_steps, prediction_steps, adj, batch_size, epochs, replay, interval, model_p['alias'],
                                          checkpoint, checkpoint_path, lr_scheduler, lrs_monitor, lrs_factor, lrs_patience, early_stopping, es_monitor, es_patience, es_restore, single_node_model, node=n, dynamic_adj=dynamic_adj, summary=self.summaries_path)
                multipreds.append(preds)
                np.save(f'{preds_path}({n}).npy', preds)
        else:
            model = build_model(model_p['alias'], loss, nodes, features, history_steps, prediction_steps, adj=adj, order=order)
            optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters(), weight_decay=0.0001)
            preds = train_and_predict(model, trainX, trainY, valX, valY, testX, testY, test_indexes, mode, optimizer, loss, nodes, features, history_steps, prediction_steps, adj, batch_size, epochs, replay, interval, model_p['alias'],
                                      checkpoint, checkpoint_path, lr_scheduler, lrs_monitor, lrs_factor, lrs_patience, early_stopping, es_monitor, es_patience, es_restore, single_node_model, dynamic_adj=dynamic_adj, summary=self.summaries_path)
        # Log execution time
        endTime = time.time()
        trainingTime = endTime - startTime
        logger.warning('Training complete in {:d} seconds.'.format(int(trainingTime)))

        # Saving predictions. The CSV file rows are ALWAYS nodes first, timestep second, test sequence last.
        # Meaning that grouping N rows at once you abstract nodes, grouping T*N rows at once you abstract timesteps and nodes.
        # If you want to select predictions for node k you select rows k, k+N, k+2N, ..., k+STN.
        preds_df = pd.DataFrame()
        preds_df['truth'] = testY.flatten(order='C')
        if single_node_model:
            preds_merged = np.empty((nodes * multipreds[0].size,), dtype=multipreds[0].dtype)
            for i, preds in enumerate(multipreds):
                flattened_preds = preds.flatten()
                preds_merged[i::nodes] = flattened_preds
            preds_df[model_name] = preds_merged.flatten(order='C')
        else:
            preds_df[model_name] = preds.flatten(order='C')
        preds_df.to_csv(os.path.join(PATH, 'experiments', preds_path))

        # Model Evaluation
        logger.warning('Evaluating model...')
        if single_node_model:
            preds = np.stack(multipreds, axis=2)
            logger.info(f'stack(multipreds, axis=2): {np.shape(multipreds)} -> {np.shape(np.stack(multipreds, axis=2))}')

        avg_mae, avg_rmse, avg_smape, std_mae, std_rmse, std_smape = evaluate(preds, testY)
        logger.info(f'Writing results:\n\tavg_mae: {avg_mae:.4f}\n\tavg_rmse: {avg_rmse:.4f}\n\tavg_smape: {avg_smape:.4f}\n\tstd_mae: {std_mae:.4f}\n\tstd_rmse: {std_rmse:.4f}\n\tstd_smape: {std_smape:.4f}')
        write_results(avg_mae, avg_rmse, avg_smape, std_mae, std_rmse, std_smape, run_path, meta_path, dataset_name, model_name, mini_dataset, features, allow_partial_batches, history_steps, prediction_steps, stride, batch_size, epochs, learning_rate, lr_scheduler, mode, trainingTime)

        logger.warning('Evaluating complete.')

        return model, preds  # For debugging purposes


# Load config files
experiment_params_path = os.path.join(PATH, 'experiments', 'config', 'experiment_params.json')
model_params_path = os.path.join(PATH, 'experiments', 'config', 'model_params.json')
dataset_params_path = os.path.join(PATH, 'experiments', 'config', 'dataset_params.json')
experiment_params_file = open(experiment_params_path, 'r')
model_params_file = open(model_params_path, 'r', encoding='utf-8-sig')
dataset_params_file = open(dataset_params_path, 'r')
experiment_params = json.load(experiment_params_file)
model_params = json.load(model_params_file)
dataset_params = json.load(dataset_params_file)

defaultResultsFolder = f'undefined-{args.model}-{args.dataset}-{datetime.datetime.now().strftime("%H%M%S")}'
if not args.output:
    args.output = defaultResultsFolder

if args.conf and not args.all:
    runner = ExperimentRunner([args.model + '-' + args.conf], [args.dataset], os.path.join(PATH, 'experiments'), experiment_params, model_params, dataset_params, args.output)
else:  # Prendo le conf a runtime da model_params.json
    confs = []
    for model in model_params['models']:
        if model['alias'] == args.model:
            finished = ExperimentRunner.is_finished(args.output, '{}-{}'.format(model['name'], args.dataset))
            if args.skip_finished and finished:
                logger.critical('Run {}-{} will be skipped, as it is already finished and flag --skip-finished is set.'.format(model['name'], args.dataset))
                continue
            confs.append(model['name'])

    runner = ExperimentRunner(confs,
                              [args.dataset],
                              os.path.join(PATH, 'experiments'),
                              experiment_params, model_params, dataset_params,
                              name=args.output)

if args.mode:
    logger.critical('Mode (Overridden): ' + args.mode)
    runner.override_mode = args.mode

if args.wsize:
    logger.critical('Window size (Overridden): ' + str(args.wsize))
    runner.override_window = args.wsize

runner.set_starting_node(args.starting_node)

runner.set_prediction_seqs(args.prediction_seqs)

if args.tensorboard:
    runner.enable_tensorboard(os.path.join(PATH, 'log', '{}-{}-{}-summaries'.format(args.model, args.dataset, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))))

if not args.collect_only:
    runner.run_all()
runner.collect_preds()
