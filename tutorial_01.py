import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist #28x28 images of hand-written 0-9

#unpack data to variables
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#show the number of train[0]
#plt.imshow(x_train[0], cmap=plt.cm.binary)
#plt.show()

#scale data to normalize(0~1, easier for network to learn)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#build model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())    #first layer, flattens the 2d array
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))    #hidden layer1, arg1: units/neurons in layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))    #hidden layer2, arg2: activation func(result)
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  #output layer, softmax >> probability distrubtion

#training the model
model.compile(optimizer='adam', #optimizer: allows neural network to learn faster
            loss='sparse_categorical_crossentropy', #loss: degree of error(should be minimized)
            metrics=['accuracy'])   #metrics: what to track
model.fit(x_train, y_train, epochs=3)   #train model with data

#calculate validation loss and accuracy
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

#saving/loading the model
model.save('first_model.model')
new_model = tf.keras.models.load_model('first_model.model')

#making prediciton
predictions = new_model.predict([x_test])
plt.imshow(x_test[0])               #show test[0] image
plt.show()
import numpy as np
print(np.argmax(predictions[0]))    #print model's prediction