import numpy as np
import json
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
npz_path = os.path.join(current_dir,'..','model_weights.npz')
json_path = os.path.join(current_dir,'weights.json')

print(f'Loading weights from: {npz_path}')
data = np.load(npz_path)
weight_dict = {}

for key in data.files:
    weight_dict[key] = data[key].tolist()

with open(json_path,'w') as f:
    json.dump(weight_dict,f,indent=4)

print(f'Weight successfully exported')