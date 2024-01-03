import tensorflow as tf
from model import tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


def predict_polarity(tweet):
    loaded_model = tf.keras.models.load_model('my_model.h5')
    token_list = tokenizer.texts_to_sequences([tweet])[0]
    # Pad the sequences
    token_list = pad_sequences([token_list], maxlen=16, padding='post')
    # Get the probabilities of predicting a word
    predicted = loaded_model.predict(token_list, verbose=0)

    return predicted


print(predict_polarity('love you'))