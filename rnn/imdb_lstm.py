'''Trains a LSTM on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
Notes:

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''

# Layer (type)                 Output Shape              Param #   
# =================================================================
# embedding_1 (Embedding)      (None, None, 128)         2560000   
# _________________________________________________________________
# lstm_1 (LSTM)                (None, 128)               131584    
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 129       
# =================================================================
# Total params: 2,691,713
# Trainable params: 2,691,713
# Non-trainable params: 0
# _________________________________________________________________


from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

# Reference:
# - https://keras.io/layers/recurrent/
# - https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(x_train.shape, y_train.shape, len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print(x_train[:10], x_test[:10], y_train[:10], y_test[:10])

print('Pad sequences (samples x time)')
# https://keras.io/preprocessing/sequence/
# Extracts the last maxlength data from the input list of lists
# Input: List of lists (or sequence)
# Padding done in pre by default since extraction happens from the last
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape, x_train[:4,:10])
print('x_test shape:', x_test.shape, x_test[:4,:10])

print('Build model...')
model = Sequential()
# https://keras.io/layers/embeddings/
# Input is a index for a one hot encoding i.e. if value is 10
# then the actual input is a 20000x1 zero vector with 1 value 
# in the 10th index
# Input is 20000x1
# Total number of parameters 2560000 = 128*20000
# Output is 128x1
model.add(Embedding(max_features, 128))
# Input is 128x1
# Total Parameters = 128*128 + 4*128 (input, output, forget and )
# Output is 128x1
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
# Weights is 128+1 vector (weight + bias)
# Input a 128x1 vector output a 1x1 vector
model.add(Dense(1, activation='sigmoid'))

# The model dumps a prediction every time step
# The output error is averaged over all time steps 
# Also across all the batch sizes too
# Then it is back propagated to all the weights
# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=1,
          # epochs=15,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)

print('Test score:', score)
print('Test accuracy:', acc)

y_pred = model.predict(x_test, batch_size=batch_size)

print('Test Pred ', y_pred.shape, y_pred[:10])

