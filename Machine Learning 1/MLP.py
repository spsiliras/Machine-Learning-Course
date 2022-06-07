import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation

from sklearn import preprocessing
from skimage.transform import resize
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0

model = tf.keras.Sequential([
	tf.keras.layers.Flatten(input_shape=(28,28)),
	tf.keras.layers.Dense(100, activation='sigmoid'),
	tf.keras.layers.Dense(50, activation='sigmoid'),
	tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=40)

results = model.evaluate(test_images, test_labels, verbose=0)

propability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = propability_model.predict(test_images)

pred = []

for i in predictions:
	pred.append(np.argmax(i))

print('accuracy = ', accuracy_score(test_labels, pred))
print('recall = ', recall_score(test_labels, pred, average=None))
print('precision = ', precision_score(test_labels, pred, average=None))
print('f1 score = ', f1_score(test_labels, pred, average=None))

