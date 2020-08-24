import os
import time
import cv2  #'pip install opencv-python' required

import random
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

NAME = "CatvDog_Cnn_64x2{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

DATADIR = "C:/Datasets/PetImages"
CATEGORIES = ["DOG", "CAT"]
IMG_SIZE = 50

training_data = []
X, y = [], []

#func for creating training data
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) #path to cats or dogs directory
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                #convert images in to an array while grayscaling the image
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                #resize the image
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                #add the image to traing dataset
                training_data.append([new_array, class_num])
            except Exception as _:
                pass


#create training data
create_training_data()

#shuffle traing dataset for efficient learning
random.shuffle(training_data)

#split features and label into X and y
for features, label in training_data:
    X.append(features)
    y.append(label)
    
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #1 because its grayscale(if rgb: 3)
X = X/255.0 #scale the data to 0~1(max pixel data: 255)
y = np.array(y)


#helpful for running multiple models on one gpu
#gpu_options = tf.GPUOptions(per_process_gpu_memorty_fraction=0.33)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#create sequential model
model = Sequential()

#input layer: Convolution layer(64 neurons, 3x3 window, ignore -1, ReLU)
model.add(Conv2D(64, (3,3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
#could be regarded as separate layer
model.add(MaxPooling2D(pool_size=(2,2)))#picks most significant data in pool from conv layer output

#hidden layer 1: Convolution layer(64 neurons, 3x3 window, ReLU)
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

#hidden layer 2: Flatten-Dense layer(64 neurons)
model.add(Flatten())    #convert data(input for Dense == 3D feature map) to 1D array
model.add(Dense(64))    #
model.add(Activation("relu"))

#output layer: Dense layer(1 neurons, Sigmoid)
model.add(Dense(1)) 
model.add(Activation("sigmoid"))

#training the model
model.compile(loss="binary_crossentropy", #because its either cats or dogs
            optimizer="adam",
            metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1, callbacks=[tensorboard])