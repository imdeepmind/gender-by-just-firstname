import pandas as pd

data = pd.read_csv('name_gender.csv')

data.dropna(inplace=True)

data = data[data['probability'] == 1.0]

data = data.drop(['probability'], axis=1)

data = data.sample(frac=1)

data.to_csv('dataset.csv', index=False)