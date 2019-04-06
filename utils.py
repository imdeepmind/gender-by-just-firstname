import re
from keras.preprocessing.sequence import pad_sequences

def clean_name(name):
    name = name.strip()
    name = re.sub(r'[\W]', '', name)
    name = re.sub(r'[0-9]', '', name)
    return name

def letter_to_number(letter):
    letter = letter.lower()
    return ord(letter) - 97

def name_to_numbers(name, max_length=25):
    ascii = []
    for letter in clean_name(name):
        ascii.append(letter_to_number(letter))
    
    return pad_sequences([ascii], max_length, padding='post')