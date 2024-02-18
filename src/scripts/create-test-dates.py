# TF
import tensorflow as tf

# Math
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from sklearn.utils import shuffle
import seaborn as sns

# Utility
import time
import datetime
from datetime import date, timedelta
import os
import sys
import random
from tqdm import tqdm
import csv
import argparse
import logging
import json

dataset = 'pems-sf-weather'


import random
idx = random.sample(range(90,440), 44)
idx = np.sort(idx)
print(idx)
np.save('../data/{}/{}_0.1-val.npy'.format(dataset, dataset), idx)

test_indexes = np.load('../data/{}/{}_0.1.npy'.format(dataset, dataset))
print(test_indexes)
print(test_indexes.shape)