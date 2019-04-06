import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

dataset = np.load('data/dataset.npy')
dataset = dataset.reshape(len(dataset), dataset.shape[2])

X = dataset[:, 0:25]
y = dataset[:, 25:27]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=1969)

model = Sequential()
model.add(Dense(28, activation='relu', input_dim=25))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2)