#!/usr/bin/env python
# coding: utf-8

# # Welcome to Jupyter!

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2  #'pip install opencv-python' require

DATADIR = "C:/Datasets/PetImages"
CATEGORIES = ["DOG", "CAT"]
IMG_SIZE = 50

for category in CATEGORIES:
    path = os.path.join(DATADIR, category) #path to cats or dogs directory
    for img in os.listdir(path):
        #convert images in to an array while grayscaling the image
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        
        plt.imshow(img_array, cmap="gray")
        plt.show()
        #break
    #break > to test grayscale image


# In[23]:


print(img_array.shape)


# In[17]:


training_data = []

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
            
create_training_data()


# In[58]:


print(len(training_data))


# In[64]:


import random
#shuffle traing dataset for efficient learning
random.shuffle(training_data)


# In[72]:


X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)
    
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #1 because its grayscale


# In[85]:


import pickle

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()


# In[99]:


pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)


# This repo contains an introduction to [Jupyter](https://jupyter.org) and [IPython](https://ipython.org).
# 
# Outline of some basics:
# 
# * [Notebook Basics](../examples/Notebook/Notebook%20Basics.ipynb)
# * [IPython - beyond plain python](../examples/IPython%20Kernel/Beyond%20Plain%20Python.ipynb)
# * [Markdown Cells](../examples/Notebook/Working%20With%20Markdown%20Cells.ipynb)
# * [Rich Display System](../examples/IPython%20Kernel/Rich%20Output.ipynb)
# * [Custom Display logic](../examples/IPython%20Kernel/Custom%20Display%20Logic.ipynb)
# * [Running a Secure Public Notebook Server](../examples/Notebook/Running%20the%20Notebook%20Server.ipynb#Securing-the-notebook-server)
# * [How Jupyter works](../examples/Notebook/Multiple%20Languages%2C%20Frontends.ipynb) to run code in different languages.

# You can also get this tutorial and run it on your laptop:
# 
#     git clone https://github.com/ipython/ipython-in-depth
# 
# Install IPython and Jupyter:
# 
# with [conda](https://www.anaconda.com/download):
# 
#     conda install ipython jupyter
# 
# with pip:
# 
#     # first, always upgrade pip!
#     pip install --upgrade pip
#     pip install --upgrade ipython jupyter
# 
# Start the notebook in the tutorial directory:
# 
#     cd ipython-in-depth
#     jupyter notebook
