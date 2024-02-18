import numpy as np
import pandas as pd
import json
import os
from io import StringIO

results_path = 'C:\\Users\\maltieri\\Downloads\\Statistical Tests\\Statistical Tests\\mydata-unomeno.csv'
json_path = 'collected-results.json'

df = pd.read_csv(results_path)
print(df)
print(len(df))

j = {}
for i in range(len(df)):
    row = df.iloc[i]
    row_dict = {}
    print(row)
    for k in range(1,len(df.columns)):
        row_dict[df.columns[k]] = row[k]
    print(row_dict)
    j['{}-{}'.format(df.iloc[i,0], i)] = row_dict

print(j)
with open(json_path, 'w') as f:
    json.dump(j, f, indent=2)
"""
{
  "ngids_0": {
    "IF": 0.6474757569084856,
    "LOF": 0.6466076405190228,
    "OC-SVM": 0.5996495382211526,
    "SUOD": 0.6127828215363834,
    "COPOD": 0.6868362823096399,
    "VAE": 0.581976790236424,
    "VLAD": 0.7851425781219726
  },
  "ngids_1": {
    "IF": 0.5833044067352162,
    "LOF": 0.6134857279899084,
    "OC-SVM": 0.600803818750252,
    "SUOD": 0.5676684064259178,
    "COPOD": 0.6783013518859832,
    "VAE": 0.5788493354247722,
    "VLAD": 0.7541150454283791
  },
  ...
  "ngids_43": {
    "IF": 0.5833044067352162,
    "LOF": 0.6134857279899084,
    "OC-SVM": 0.600803818750252,
    "SUOD": 0.5676684064259178,
    "COPOD": 0.6783013518859832,
    "VAE": 0.5788493354247722,
    "VLAD": 0.7541150454283791
  }
}
"""