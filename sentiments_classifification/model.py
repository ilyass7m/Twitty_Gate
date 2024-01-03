import pandas as pd 

from joblib import dump







#Loading the dataset

df = pd.read_csv( r'C:\Users\HOME\twitty_gate\sentiments_classifification\sentiment_analysis.csv', delimiter=',')


#print(df.head())

sentences = df['tweet'].to_list()
labels=df['label'].to_list()



#print(len(sentences))

# Hyperparameters

# Number of examples to use for training
training_size = 6000

# Vocabulary size of the tokenizer
vocab_size = 4000

# Maximum length of the padded sequences
max_length = 16

# Output dimensions of the Embedding layer
embedding_dim = 8




# Split the sentences
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]

# Split the labels
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]



import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Parameters for padding and OOV tokens
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"

# Initialize the Tokenizer class
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

# Generate the word index dictionary
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

# Generate and pad the training sequences
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Generate and pad the testing sequences
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Convert the labels lists into numpy arrays
training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels)


#print(training_padded[:3])


#Build and Compile the model

# ----- we will use a Globalaveradepooling1D layer after the Embedding one instead of a Flatten
# this will reduce the dimensionality and thus the number of training parameters




import tensorflow as tf
import numpy as np

# Initialize a GlobalAveragePooling1D (GAP1D) layer
#gap1d_layer = tf.keras.layers.GlobalAveragePooling1D()

# Define sample array
#sample_array = np.array([[[10,2],[1,3],[1,1]]])

# Print shape and contents of sample array
#print(f'shape of sample_array = {sample_array.shape}')
#print(f'sample array: {sample_array}')

# Pass the sample array to the GAP1D layer
#output = gap1d_layer(sample_array)

# Print shape and contents of the GAP1D output array
#print(f'output shape of gap1d_layer: {output.shape}')
#print(f'output array of gap1d_layer: {output.numpy()}')


# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Print the model summary
model.summary()


# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Print the model summary
model.summary()



# Compile the model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


num_epochs = 30

# Train the model
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)

dump(model, 'trained_model.joblib')
dump(tokenizer, 'tokenizer.joblib')





import matplotlib.pyplot as plt

# Plot utility
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
# Plot the accuracy and loss
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")




