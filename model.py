import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split

dataset = np.load('data/dataset.npy')
dataset = dataset.reshape(len(dataset), dataset.shape[2])

X = dataset[:, 0:25]
y = dataset[:, 25:27]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=1969)

model = Sequential()
model.add(Embedding(500, 32, input_length=25))
model.add(Dropout(0.33))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

model.save('data/model.h5')