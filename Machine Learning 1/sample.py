import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from skimage.transform import resize
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

for i in range(0, len(train_images)):
	resized = resize(train_images[i], (14,14))

	train_images_resized.append(resized)

train = np.array(train_images_resized)

plt.figure()
plt.imshow(train[1000])
plt.colorbar()
plt.grid(False)
plt.show()

for i in range(0, len(test_images)):
	resized = resize(test_images[i], (14,14))

	test_images_resized.append(resized)

test = np.array(test_images_resized)

#reshape train & test data so they can be used in KNeighborsClassifier
train_images = train.reshape(train.shape[0], train.shape[1]*train.shape[2])
test_images = test.reshape(test.shape[0], test.shape[1]*test.shape[2])

print(train_images.shape)
print(test_images.shape)

