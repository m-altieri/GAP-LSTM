import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

closeness_path = '../../data/lightsource/closeness-lightsource.npy'

closeness = np.load(closeness_path)
mask = np.full(closeness.shape, False)
for i in range(len(mask)):
    for j in range(len(mask)):
        if j > i:
            mask[i,j] = True
sns.heatmap(closeness, mask=mask)

print(mask)
plt.savefig('tsne-lightsource/closeness-heatmap.png')