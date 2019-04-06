import keras
from utils import name_to_numbers

model = keras.models.load_model('data/model.h5')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

while True:
    n = input('Please enter yout name please: ')
    
    if n == 'quit':
        break
    
    name = name_to_numbers(n)
    
    pred = model.predict_classes(name)
    pred_prob = model.predict(name)

    pred = pred[0]
    
    if pred == 0:
        print('Female')
    elif pred == 1:
        print('Male')
    
    print('')
    print('The model is predicted that the {}\'s gender is {}% female and {}% male'.format(n,pred_prob[0][0]*100, pred_prob[0][1]*100))
    
