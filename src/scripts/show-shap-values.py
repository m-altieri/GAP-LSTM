import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

shap_values_path = '../shap_values-n0-d0.pkl'
output_path = '../plot.svg'

with open(shap_values_path, 'rb') as f:
	shap_values = pickle.load(f)

tmp = [np.reshape(shap_values[i][0], (19, 7, 11)) for i in range(len(shap_values))]
shap_values = np.stack(tmp) # (19, 19, 7, 11)
shap_values = np.sum(shap_values, axis=(1,2)) # (19, 11)

# shap_values = np.transpose(shap_values, (2, 0, 1))
# shap_values = np.squeeze(shap_values)


# shap_values = np.transpose(shap_values) # (19, 1463)
# shap_values = np.reshape(shap_values, (19, 19, 7, 11))
# shap_values = np.sum(shap_values, axis=(1,2)) # (19, 11)

# shap_values = np.reshape(shap_values, (19, 7, 11, 19))
# shap_values = np.mean(shap_values, axis=(0,1))
# shap_values = np.transpose(shap_values)


shap_values = np.transpose(shap_values)
expl = shap.Explanation(values=shap_values[:-1], base_values=shap_values[-1])
shap.plots.bar(expl, show=False, max_display=20)
plt.savefig(output_path)
