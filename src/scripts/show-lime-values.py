import numpy as np
import pickle
import lime
#import matplotlib.pyplot as plt

lime_values_path = '../lime/lime_values-n0-d14.pkl'
output_path = '../lime/lime-expl-n0-d14.html'

T = 19
N = 7
F = 11

with open(lime_values_path, 'rb') as f:
	lime_values = pickle.load(f)

aggregated_values = [0 for i in range(F)]

# Estrazione values e sommo quelli della stessa feature
lime_values_list = lime_values.as_list()
for i in range(len(lime_values_list)):
	# Non so com'è strutturata la condizione
	try:
		feature_id = int(lime_values_list[i][0].split(' ')[0]) # Caso 55 <= 0.23
	except:
		feature_id = int(lime_values_list[i][0].split(' ')[2]) # Caso 0.23 <= 55 <= 0.45

	value = lime_values_list[i][1]
	aggregated_values[feature_id % F] += value # @TODO forse devo tenere 2 valori separati per ogni feature, uno per i contributi positivi e uno negativi

print(aggregated_values)


# Ricreo i lime values usando le feature aggregate
lime_values_new = [(f'{i} => 0.00', aggregated_values[i]) for i in range(len(aggregated_values))]


# Visualizzo i valori
with open(output_path, 'w') as w:
	w.write(lime_values_new.as_html())

