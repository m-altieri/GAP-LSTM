import numpy as np
import os
import random

DATASET = 'pems-sf-weather'
T = 6
PERCENTAGE_DATES = 0.1
MIN_IDX = 95
SUFFIX = '-val'

data_path = '..\..\data\{}'.format(DATASET)
data = np.load(os.path.join(data_path, '{}.npz'.format(DATASET)))['data']

O, N, F = data.shape
seqs = O // T
dates = int(np.round(seqs * PERCENTAGE_DATES))
idx = random.sample(range(MIN_IDX, seqs-1), dates)
idx = np.sort(idx)

np.save(os.path.join(data_path, '{}_{}{}.npy'.format(DATASET, PERCENTAGE_DATES, SUFFIX)), idx)