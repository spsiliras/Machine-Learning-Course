import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt

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

dimensions = 7*7

#normalize the data
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images = train_images/255.0
test_images = test_images/255.0

for i in range(0, len(train_images[0:60000])):
	resized = resize(train_images[i], (7,7))

	train_images_resized.append(resized)

train_images = np.array(train_images_resized)

for i in range(0, len(test_images)):
	resized = resize(test_images[i], (7,7))

	test_images_resized.append(resized)

test_images = np.array(test_images_resized)

#reshape train & test data so they can be used in KNeighborsClassifier
train_images = train_images.reshape(train_images.shape[0], train_images.shape[1]*train_images.shape[2])
test_images = test_images.reshape(test_images.shape[0], test_images.shape[1]*test_images.shape[2])

mean = []
variance = []

def compute_mean_variance():
	#countClasses = [0,0,0,0,0,0,0,0,0,0]
	values = []

	for i in range (0, 10):
		l = []
		for j in range (0, dimensions):
			l.append(0)

		values.append(l)

	for i in range(0, len(train_images)):
		for j in range(0, dimensions):
			values[train_labels[i]][j] += train_images[i][j]

		#countClasses[train_labels[i]] += 1

	for i in range(0, 10):
		m = []
		v = []
		for j in range(0, dimensions):
			m.append(values[i][j] / 6000)

		mean.append(m)

		for j in range(0, dimensions):
			v.append(((values[i][j] - mean[i][j]) ** 2) / 6000)

		variance.append(v)

compute_mean_variance()

print(test_images.shape)

pred = []

def fit():
	
	for i in range(0, len(test_images)):
		print(i)
		g_value = []
		for j in range(0, 10):
			sum1 = 0
			sum2 = 0
			for k in range(0, dimensions):
				if(variance[j][k] == 0):
					sum1 += 0
				else:
					sum1 += math.log(math.sqrt(variance[j][k]))
				sum2 += (((test_images[i][k] - mean[j][k]) ** 2) / 2*variance[j][k])

			g_value.append(-sum1 - sum2)

		max_value = max(g_value)

		pred.append(g_value.index(max_value))


fit()

predictions = np.array(pred)

print('accuracy = ', accuracy_score(test_labels, predictions))

print('recall = ', recall_score(test_labels, predictions, average=None))
print('precision = ', precision_score(test_labels, predictions, average=None))
print('f1_score = ', f1_score(test_labels, predictions, average=None))