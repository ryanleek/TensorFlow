#Using trained model to predict a new image
from cv2 import cv2
import tensorflow as tf

CATEGORIES = ["Dog", "Cat"]

#resize and grayscale the image in filepath, then return in 4D array 
def prepare_img(filepath):
    IMG_SIZE = 50   #size of resized image

    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  #grayscale the image
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) #resize the image to 50x50

    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1) #return in array for the model

#load previously created model
model = tf.keras.models.load_model("64x3-CNN.model")
#predict the new image with the model
prediction = model.predict([prepare_img('cat.jpg')])
#print the prediction result
print(CATEGORIES[int(prediction[0][0])])