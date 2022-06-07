import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from skimage.transform import resize

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

train_images_resized = []
test_images_resized = []

#normalize the data
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images = train_images/255.0
test_images = test_images/255.0

#resize to 14x14
for i in range(0, len(train_images)):
	resized = resize(train_images[i], (14,14))

	train_images_resized.append(resized)

train_images = np.array(train_images_resized)

for i in range(0, len(test_images)):
	resized = resize(test_images[i], (14,14))

	test_images_resized.append(resized)

test_images = np.array(test_images_resized)

#reshape train & test data so they can be used in KNeighborsClassifier
train_images = train_images.reshape(train_images.shape[0], train_images.shape[1]*train_images.shape[2])
test_images = test_images.reshape(test_images.shape[0], test_images.shape[1]*test_images.shape[2])

print(train_images.shape)
print(test_images.shape)

clf = OneVsRestClassifier(SVC(kernel=cosine_similarity)).fit(train_images[0:40000], train_labels[0:40000])

pred = clf.predict(test_images)

print('accuracy = ', accuracy_score(test_labels, pred))

print('recall = ', recall_score(test_labels, pred, average=None))
print('precision = ', precision_score(test_labels, pred, average=None))
print('f1_score = ', f1_score(test_labels, pred, average=None))