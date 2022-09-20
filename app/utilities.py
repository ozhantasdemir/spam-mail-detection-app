import re
import string
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


def remove_hyperlink(word):
    return  re.sub(r"http\S+", "", word)

def to_lower(word):
    result = word.lower()
    return result

def remove_number(word):
    result = re.sub(r'\d+', '', word)
    return result

def remove_punctuation(word):
    result = word.translate(str.maketrans(dict.fromkeys(string.punctuation)))
    return result

def remove_whitespace(word):
    result = word.strip()
    return result

def replace_newline(word):
    return word.replace('\n','')

def clean_up_pipeline(sentence):
    cleaning_utils = [remove_hyperlink,
                      replace_newline,
                      to_lower,
                      remove_number,
                      remove_punctuation,remove_whitespace]
    for o in cleaning_utils:
        sentence = o(sentence)
    return sentence


max_len = 2000 # max number of words in a question to use

with open('models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


model_final = load_model('models/model.h5')


def predict_mail(model, sample_mail):

    pred_to_label = {0: 'Ham', 1: 'Spam'}

    text = np.array(tokenizer.texts_to_sequences(sample_mail))
    test_features = pad_sequences(text,maxlen=max_len)

    y_predict  = [1 if o>0.5 else 0 for o in model.predict(test_features)]


    data = []
    for mail, pred in zip(sample_mail, y_predict):
        #data.append((mail, pred, pred_to_label[pred]))
        data.append(pred)

    return data
