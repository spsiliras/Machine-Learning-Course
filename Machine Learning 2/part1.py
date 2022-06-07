import tensorflow as tf
import math
import random
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

dimensions = 28*28

#method (=0 for R1 form of train images, =1 for histogram)
method = 1

#which distance type to use, 0=euclidean, 1=manhattan and 2=cosine
distance = 2

#number of clusters
k = 10

#the centers of the clusters
centroids = []

#keeps the id of train images of each one cluster
clusters = {}

#finishing error threshold
errorThreshold = 100

#the number of samples to use
train_number = 10000

#number of bins if we use R2 format (it can be 16, 32,64 or 128)
numberOfBins = 16

#normalize the data
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images = train_images/255.0
test_images = test_images/255.0

train_images = train_images.reshape(train_images.shape[0], train_images.shape[1]*train_images.shape[2])
test_images = test_images.reshape(test_images.shape[0], test_images.shape[1]*test_images.shape[2])

train_histogram = []

if method == 1:

	dimensions = numberOfBins

	bins = []
	bins.append(0)

	for i in range(1, numberOfBins+1):
		bins.append(i/numberOfBins)

	for i in range(0, len(train_images[0:train_number])):
		hist = np.histogram(train_images[i], bins)
		train_histogram.append(hist[0])

def euclidean_distance(s1, s2):
	total = 0

	for i in range(0, dimensions):
		total += (s1[i] - s2[i])**2

	return math.sqrt(total)

def manhattan_distance(s1, s2):
	total = 0

	for i in range(0, dimensions):
		total += abs(s1[i]-s2[i])

	return total

def cosine_distance(s1, s2):
	sum1 = 0
	sum2 = 0
	sum3 = 0

	for i in range(0, dimensions):
		sum1 += s1[i]*s2[i]
		sum2 += s1[i]**2
		sum3 += s2[i]**2

	total = sum1 / (math.sqrt(sum2)*math.sqrt(sum3))

	return total

#initialize centroids and clusters
def init_centers():
	for i in range(0, k):
		center = []

		#initialize with choosing a random sample from all train samples
		ind = random.randint(0, train_number)
		
		if method == 0:
			for j in range (0, dimensions):
				#a random float between 0 and 1
				center.append(train_images[ind][j])

		elif method == 1:
			for j in range(0,dimensions):
				center.append(train_histogram[ind][j])

		#store the centers to global variable centroids
		centroids.append(center)

def update_centers():

	for i in range(0, k):
		for j in range(0, dimensions):
			val = 0
			for z in clusters[i]:
				if method == 0:
					val += train_images[z][j]
				elif method == 1:
					val += train_histogram[z][j]

			if len(clusters[i]) == 0:
				centroids[i][j] = 1
			else:
				centroids[i][j] = val / len(clusters[i])

def k_means():

	error = 0

	for i in range(0, k):
		clusters[i] = []


	#compute error and update clusters
	if method == 0:
		for i in range(0, train_number):
			#print(i)
			minimum = float('inf')
			minIndex = -1

			for j in range(0, k):
				if distance == 0:
					d = euclidean_distance(train_images[i], centroids[j])
					if d < minimum:
						minimum = d
						minIndex = j

				elif distance == 1:
					d = manhattan_distance(train_images[i], centroids[j])
					if d < minimum:
						minimum = d
						minIndex = j

				elif distance == 2:
					d = cosine_distance(train_images[i], centroids[j])
					if d < minimum:
						minimum = d
						minIndex = j

			error += minimum
			clusters[minIndex].append(i)

		update_centers()

	if method == 1:
		for i in range(0, train_number):
			
			minimum = float('inf')
			minIndex = -1

			for j in range(0, k):
				if distance == 0:
					d = euclidean_distance(train_histogram[i], centroids[j])
					if d < minimum:
						minimum = d
						minIndex = j

				elif distance == 1:
					d = manhattan_distance(train_histogram[i], centroids[j])
					if d < minimum:
						minimum = d
						minIndex = j

				elif distance == 2:
					d = cosine_distance(train_histogram[i], centroids[j])
					if d < minimum:
						minimum = d
						minIndex = j

			error += minimum
			clusters[minIndex].append(i)

		update_centers()

	return error

def main():
	init_centers()
	#first call of k-means and keep the error value
	err1 = k_means()
	print('err1=',err1)

	#second call of k-means
	err2 = k_means()
	print('err2=',err2)

	while(err1-err2 > errorThreshold):

		err1 = err2

		err2 = k_means()

		print('err2=',err2)

main()

TP_total = 0
F_measure = 0

for i in range(0, k):

	find_cluster = {}

	for j in clusters[i]:
		label = train_labels[j]
		if label not in find_cluster:
			find_cluster[label] = 1
		else:
			find_cluster[label] += 1

	count = -1
	cluster_index = -1

	for j in find_cluster:
		if find_cluster[j] > count:
			count = find_cluster[j]
			cluster_index = j

	TP_total += find_cluster[cluster_index]

	#TP is eqwual to the number of the category with the most labels in cluster
	true_positives = find_cluster[cluster_index]

	#FP is equal to all the labels contained in cluster minus that which predicted as TP
	false_positives = len(clusters[i]) - find_cluster[cluster_index]

	#FN is equal to all labels in train set where its category is the same as
	#the majority label in the current cluster
	#each category has the same number of items
	false_negatives = (train_number/k) - find_cluster[cluster_index]

	precision = true_positives / (true_positives + false_positives)

	recall = true_positives / (true_positives + false_negatives)

	F1_score = 2*((precision*recall) / (precision+recall))

	F_measure += F1_score

print("Purity is: " +str(TP_total/train_number))
print("F_measure is: " + str(F_measure))
