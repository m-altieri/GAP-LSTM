import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import math

DEFAULT_INTFILE = '../../experiments/attw-S2S-GCLSTM-AT-spektral-wind-nrel-BS2LR1e-2LRSES-landmark/attw-S2S-GCLSTM-AT-BS2LR1e-2LRSES-wind-nrel.npy'
#DEFAULT_TESTDATES_PATH = '../../data/wind-nrel/wind-nrel_0.1.npy'
DEFAULT_OUTPUT_NAME = 'undefined'

argparser = argparse.ArgumentParser(description='Show interpretable results.')
argparser.add_argument('-i', '--intfile', action='store', default=DEFAULT_INTFILE, dest='intfile_path', help='path of the .npz int file to use.')
#argparser.add_argument('--datesfile', action='store', default=DEFAULT_TESTDATES_PATH, dest='datesfile_path', help='select the path of the test dates file to use.')
argparser.add_argument('-o', '--output-name', action='store', default=DEFAULT_OUTPUT_NAME, dest='output_name', help='path and name of the output folder.')
argparser.add_argument('--normalize', action='store_const', const=True, default=False, help='whether to normalize heatmaps or show them raw.')

args = vars(argparser.parse_args())

intfile_path = args['intfile_path']
#test_dates_path = args['datesfile_path']
output_name = args['output_name']
#data_path = '../../data/wind-nrel/wind-nrel.npz'

intfile = np.load(intfile_path)
#test_dates = np.load(test_dates_path)
#data = np.load(data_path)['data']

def closestDivisors(n):
    a = round(math.sqrt(n))
    while n%a > 0: a -= 1
    return a,n//a

if not os.path.exists(output_name): os.makedirs(output_name)

for f in intfile.files:
    # Visualize each file
    for seq in range(len(intfile[f])):
        if len(intfile[f].shape) == 3:
            heatmap = intfile[f][seq]
            if args['normalize']:
                for col in range(heatmap.shape[-1]):
                    heatmap[:, col] -= heatmap[:, col].mean()
                # <---
                for row in range(heatmap.shape[0]):
                    heatmap[row, :] -= heatmap[row, :].mean()
                # --->
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # Minmax
            sns.heatmap(heatmap, cmap='YlGnBu', square=True, cbar_kws={'format': '%.2f'})
            plt.savefig(os.path.join(output_name, '{}-{}.png'.format(f, seq)))
            plt.clf()
        if len(intfile[f].shape) == 4:
            rows, cols = closestDivisors(intfile[f].shape[1])
            fig, ax = plt.subplots(rows, cols, figsize=(50, 50), num=1, clear=True)
            nmaps = len(intfile[f][seq])
            for n in range(nmaps):
                heatmap = intfile[f][seq][n]
                if args['normalize']:
                    for col in range(heatmap.shape[-1]):
                        heatmap[:, col] -= heatmap[:-1, col].mean()
                        heatmap[-1, col] = heatmap[-2, col]
                    # <---
                    for row in range(heatmap.shape[0]):
                        heatmap[row, :] -= heatmap[row, :].mean() / 3.0
                    # --->
                    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # Minmax
                sns.heatmap(heatmap, ax=ax[n//rows,n%cols], cmap='YlGnBu', square=True, cbar_kws={'format': '%.2f'})
            fig.savefig(os.path.join(output_name, '{}-{}.png'.format(f, seq)))
            plt.clf()

'''Codice vecchio per attw di GAP-LSTM.

for i in range(len(test_dates)):
    print(i)

    P = 12
    fig, ax = plt.subplots(P, 3, figsize=(50, 50), num=1, clear=True)

    avgSameSeq = np.mean(attw[i, :], axis=0)
    avgAllSeqs = np.mean(attw, axis=(0,1))

    for p in range(P):
        sns.heatmap(attw[i, p*2] - avgSameSeq, ax=ax[p,0], cbar=False, square=True, xticklabels=False, yticklabels=False)  # center=1 / attw.shape[2],
        sns.heatmap(attw[i, p*2] - avgAllSeqs, ax=ax[p,1], cbar=False, center=1 / attw.shape[2], square=True, xticklabels=False, yticklabels=False)  # center=1 / attw.shape[2],

        sns.heatmap(attw[i, p*2], ax=ax[p,2], cbar=False, center=1 / attw.shape[2], square=True, xticklabels=False, yticklabels=False)
        #sns.heatmap(attw[i+2, 0], ax=ax[2], cbar=False, center=1 / attw.shape[2], square=True, xticklabels=False, yticklabels=False)
        #sns.heatmap(attw[i+3, 0], ax=ax[3], cbar=False, center=1 / attw.shape[2], square=True, xticklabels=False, yticklabels=False)

        fig.savefig('attw_wind/{}-delta.png'.format(i))

    #plt.clf()
    #plt.plot(data[test_dates[i]*24:(test_dates[i+1])*24,:,0])
    #plt.savefig('attw_beijing/plot-{}.png'.format(i))
    #plt.clf()
'''