import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

enc_states_path = '../../experiments/tsne-S2S-GCLSTM-AT-spektral-lightsource-BS16LR1e-3LRSES/enc-S2S-GCLSTM-AT-BS16LR1e-3LRSES-lightsource.npy'
dec_states_path = '../../experiments/tsne-S2S-GCLSTM-AT-spektral-lightsource-BS16LR1e-3LRSES/dec-S2S-GCLSTM-AT-BS16LR1e-3LRSES-lightsource.npy'
enc_states = np.load(enc_states_path)
dec_states = np.load(dec_states_path)

test_dates_path = '../../data/lightsource/lightsource_0.1.npy'
test_dates = np.load(test_dates_path)

print(enc_states.shape)
print(dec_states.shape)
print(test_dates.shape)

S, H, N, F = enc_states.shape
S, P, N, F = dec_states.shape

enc_states = np.reshape(enc_states, (S*H*N, F))
dec_states = np.reshape(dec_states, (S*P*N, F))

for perplexity in [5, 30, 50, 100]:
    plt.clf()

    enc_embeds = TSNE(n_components=2, perplexity=perplexity, learning_rate='auto', init='pca').fit_transform(enc_states)
    dec_embeds = TSNE(n_components=2, perplexity=perplexity, learning_rate='auto', init='pca').fit_transform(dec_states)

    enc_embeds = np.reshape(enc_embeds, (S, H, N, 2))
    dec_embeds = np.reshape(dec_embeds, (S, P, N, 2))

    sequence = 31
    if not os.path.exists('tsne-lightsource/sequence-{}'.format(sequence)): os.mkdir('tsne-lightsource/sequence-{}'.format(sequence))
    enc_embeds = enc_embeds[sequence]
    dec_embeds = dec_embeds[sequence]

    fig, ax = plt.subplots(4, 5, num=1, clear=True)
    for h in range(H):
        ax[h // 5][h % 5].set_xlim([enc_embeds[..., 0].min(), enc_embeds[..., 0].max()])
        ax[h // 5][h % 5].set_ylim([enc_embeds[..., 1].min(), enc_embeds[..., 1].max()])
        for n in range(enc_embeds.shape[1]):
            ax[h // 5][h % 5].scatter(enc_embeds[h, n, 0], enc_embeds[h, n, 1], label=n)
            plt.gcf().set_size_inches((20, 20))
    ax[-1][-2].legend()

    fig.savefig('tsne-lightsource/sequence-{}/enc-perplexity{}.png'.format(sequence, perplexity), dpi=80)

    fig, ax = plt.subplots(4, 5, num=1, clear=True)
    for p in range(P):
        ax[p // 5][p % 5].set_xlim([dec_embeds[..., 0].min(), dec_embeds[..., 0].max()])
        ax[p // 5][p % 5].set_ylim([dec_embeds[..., 1].min(), dec_embeds[..., 1].max()])
        for n in range(dec_embeds.shape[1]):
            ax[p // 5][p % 5].scatter(dec_embeds[p, n, 0], dec_embeds[p, n, 1], label=n)
            plt.gcf().set_size_inches((20, 20))
    ax[-1][-2].legend()

    fig.savefig('tsne-lightsource/sequence-{}/dec-perplexity{}.png'.format(sequence, perplexity), dpi=80)
