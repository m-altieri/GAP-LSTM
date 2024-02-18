import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

attw_path = '../../experiments/attw-S2S-GCLSTM-AT-spektral-wind-nrel-BS2LR1e-2LRSES-landmark/attw-S2S-GCLSTM-AT-BS2LR1e-2LRSES-wind-nrel.npy'
data_path = '../../data/wind-nrel/wind-nrel.npz'
test_dates_path = '../../data/wind-nrel/wind-nrel_0.1.npy'
attw = np.load(attw_path)
data = np.load(data_path)['data']
test_dates = np.load(test_dates_path)

print(attw.shape)
print(data.shape)
print(test_dates)

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