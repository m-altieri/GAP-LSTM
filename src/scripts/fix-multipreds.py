import numpy as np
import os
import pandas as pd

model_name = 'ARIMA'
dataset = 'wind-nrel'
path = '../../experiments//test-ARIMA-{}-AUTO-landmark'.format(dataset)
n0 = np.load(os.path.join(path, 'preds-ARIMA-AUTO-{}.csv(0).npz'.format(dataset)))['preds']
n1 = np.load(os.path.join(path, 'preds-ARIMA-AUTO-{}.csv(1).npz'.format(dataset)))['preds']
n2 = np.load(os.path.join(path, 'preds-ARIMA-AUTO-{}.csv(2).npz'.format(dataset)))['preds']
n3 = np.load(os.path.join(path, 'preds-ARIMA-AUTO-{}.csv(3).npz'.format(dataset)))['preds']
n4 = np.load(os.path.join(path, 'preds-ARIMA-AUTO-{}.csv(4).npz'.format(dataset)))['preds']
#n5 = np.load(os.path.join(path, 'preds-ARIMA-AUTO-{}.csv(5).npz'.format(dataset)))['preds']
#n6 = np.load(os.path.join(path, 'preds-ARIMA-AUTO-{}.csv(6).npz'.format(dataset)))['preds']
#n7 = np.load(os.path.join(path, 'preds-ARIMA-AUTO-{}.csv(7).npz'.format(dataset)))['preds']
#n8 = np.load(os.path.join(path, 'preds-ARIMA-AUTO-{}.csv(8).npz'.format(dataset)))['preds']
#n9 = np.load(os.path.join(path, 'preds-ARIMA-AUTO-{}.csv(9).npz'.format(dataset)))['preds']
#n10 = np.load(os.path.join(path, 'preds-ARIMA-AUTO-{}.csv(10).npz'.format(dataset)))['preds']
#n11 = np.load(os.path.join(path, 'preds-ARIMA-AUTO-{}.csv(11).npz'.format(dataset)))['preds']
#n12 = np.load(os.path.join(path, 'preds-ARIMA-AUTO-{}.csv(12).npz'.format(dataset)))['preds']
#n13 = np.load(os.path.join(path, 'preds-ARIMA-AUTO-{}.csv(13).npz'.format(dataset)))['preds']
#n14 = np.load(os.path.join(path, 'preds-ARIMA-AUTO-{}.csv(14).npz'.format(dataset)))['preds']
#n15 = np.load(os.path.join(path, 'preds-ARIMA-AUTO-{}.csv(15).npz'.format(dataset)))['preds']
#n16 = np.load(os.path.join(path, 'preds-ARIMA-AUTO-{}.csv(16).npz'.format(dataset)))['preds']

multipreds = [
    n0, n1, n2, n3, n4#, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16
              ]
nodes = len(multipreds)

preds_df = pd.DataFrame()
preds_merged = np.empty((nodes * multipreds[0].size,), dtype=multipreds[0].dtype)
for i, preds in enumerate(multipreds):
    flattened_preds = preds.flatten()
    preds_merged[i::nodes] = flattened_preds
preds_df[model_name] = preds_merged.flatten(order='C')
preds_df.to_csv(os.path.join(path, 'fix-{}.csv'.format(model_name)))