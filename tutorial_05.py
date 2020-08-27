import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM #or CuDNNLSTM for gpu

#get mnist from keras and save to variables
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#normalize data
x_train = x_train/255.0
x_test = x_test/255.0

#start sequential model
model = Sequential()

#input layer 
model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))
model.add(Dropout(0.2))
#hidden layer 1
model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.2))
#hidden layer 2
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
#output layer
model.add(Dense(10, activation='softmax'))

#specified Adam optimizer
opt = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-3)  #dimish learning rate over time

#train model
model.compile(loss='sparse_categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
