import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from joblib import load

tokenizer = load('tokenizer.joblib')

def predict_polarity(tweet):
    loaded_model = load('trained_model.joblib')
    token_list = tokenizer.texts_to_sequences([tweet])[0]
    # Pad the sequences
    token_list = pad_sequences([token_list], maxlen=16, padding='post')
    # Get the probabilities of predicting a word
    predicted = loaded_model.predict(token_list, verbose=0)
    predicted=predicted[0][0]
    if predicted > 0.5:
        return 1
    else:
        return 0

    


print(predict_polarity('love you'))