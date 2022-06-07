import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from skimage.transform import resize
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

#normalize the data
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images = train_images/255.0
test_images = test_images/255.0

#reshape train & test data so they can be used in KNeighborsClassifier
train_images = train_images.reshape(train_images.shape[0], train_images.shape[1]*train_images.shape[2])
test_images = test_images.reshape(test_images.shape[0], test_images.shape[1]*test_images.shape[2])

neigh = KNeighborsClassifier(n_neighbors=17, metric='cosine')
neigh.fit(train_images, train_labels)

pred = neigh.predict(test_images)

print('accuracy = ', accuracy_score(test_labels, pred))

print('recall = ', recall_score(test_labels, pred, average=None))
print('precision = ', precision_score(test_labels, pred, average=None))
print('f1_score = ', f1_score(test_labels, pred, average=None))







