import numpy as np
import pandas as pd
from utils import name_to_numbers

data = pd.read_csv('data/dataset.csv')

names = data['name'].values
genders = pd.get_dummies(data['gender']).values

del data

dataset = []
for i in range(len(names)):
    name = name_to_numbers(names[i])
    gender = genders[i]
    row = np.hstack((name, gender.reshape(1,-1)))
    dataset.append(row)
    
np.save('data/dataset.npy', dataset)