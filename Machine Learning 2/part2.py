import tensorflow as tf
import math
import random
import numpy as np
from sklearn.cluster import KMeans

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

#normalize the data
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images = train_images/255.0
test_images = test_images/255.0

train_images = train_images.reshape(train_images.shape[0], train_images.shape[1]*train_images.shape[2])
test_images = test_images.reshape(test_images.shape[0], test_images.shape[1]*test_images.shape[2])

dimensions = 28*28

#method (=0 for R1 form of train images, =1 for histogram)
method = 1

#number of bins if we use R2 format (it can be 16, 32,64 or 128)
numberOfBins = 128

#0= eucklidean distance, 1= manhattan distance and 2=cosine distance
distance = 2

#the number of train set photos used
train_number = 20000

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

#number of clusters
M = 10

tree = []

leaf = []

leaf_index = []

train_images_index = []

for i in range(0, train_number):
	leaf.append(i)

train_images_index.append(leaf)

if method == 0:
	tree.append(train_images[0:train_number])

elif method == 1:
	tree.append(train_histogram[0:train_number])

def hierarchical_clustering():

	k = 1

	while k != M:
		print(k)
		if k == 1:

			new_indexes1 = []
			new_indexes2 = []

			new_node1 = []
			new_node2 = []

			if method == 0:

				kmeans = KMeans(n_clusters=2).fit(train_images[0:train_number])

				for i in range(0, train_number):
					if distance == 0:
						if euclidean_distance(train_images[i], kmeans.cluster_centers_[0]) > euclidean_distance(train_images[i], kmeans.cluster_centers_[1]):
							new_indexes1.append(i)

							new_node1.append(train_images[i])

						else:
							new_indexes2.append(i)

							new_node2.append(train_images[i])

					elif distance == 1:
						if manhattan_distance(train_images[i], kmeans.cluster_centers_[0]) > manhattan_distance(train_images[i], kmeans.cluster_centers_[1]):
							new_indexes1.append(i)

							new_node1.append(train_images[i])

						else:
							new_indexes2.append(i)

							new_node2.append(train_images[i])

					elif distance == 2:
						if cosine_distance(train_images[i], kmeans.cluster_centers_[0]) > cosine_distance(train_images[i], kmeans.cluster_centers_[1]):
							new_indexes1.append(i)

							new_node1.append(train_images[i])

						else:
							new_indexes2.append(i)

							new_node2.append(train_images[i])

				tree.append(new_node1)
				tree.append(new_node2)

				leaf_index.append(len(tree) - 2)
				leaf_index.append(len(tree) - 1)

				train_images_index.append(new_indexes1)
				train_images_index.append(new_indexes2)

			elif method == 1:

				kmeans = KMeans(n_clusters=2).fit(train_histogram[0:train_number])

				for i in range(0, train_number):
					if distance == 0:
						if euclidean_distance(train_histogram[i], kmeans.cluster_centers_[0]) > euclidean_distance(train_histogram[i], kmeans.cluster_centers_[1]):
							new_indexes1.append(i)

							new_node1.append(train_histogram[i])

						else:
							new_indexes2.append(i)

							new_node2.append(train_histogram[i])

					elif distance == 1:
						if manhattan_distance(train_histogram[i], kmeans.cluster_centers_[0]) > manhattan_distance(train_histogram[i], kmeans.cluster_centers_[1]):
							new_indexes1.append(i)

							new_node1.append(train_histogram[i])

						else:
							new_indexes2.append(i)

							new_node2.append(train_histogram[i])

					elif distance == 2:
						if cosine_distance(train_histogram[i], kmeans.cluster_centers_[0]) > cosine_distance(train_histogram[i], kmeans.cluster_centers_[1]):
							new_indexes1.append(i)

							new_node1.append(train_histogram[i])

						else:
							new_indexes2.append(i)

							new_node2.append(train_histogram[i])

				tree.append(new_node1)
				tree.append(new_node2)

				leaf_index.append(len(tree) - 2)
				leaf_index.append(len(tree) - 1)

				train_images_index.append(new_indexes1)
				train_images_index.append(new_indexes2)


		else:
			max_var_index = -1
			maxim = float("-inf")

			for i in leaf_index:
				var = np.var(tree[i])

				if var > maxim:
					maxim = var
					max_var_index = i

			leaf_index.remove(max_var_index)

			print(len(tree[max_var_index]))

			kmeans = KMeans(n_clusters=2).fit(tree[max_var_index])

			new_indexes1 = []
			new_indexes2 = []

			new_node1 = []
			new_node2 = []

			for i in range(0, len(tree[max_var_index])):

				if distance == 0:
					if euclidean_distance(tree[max_var_index][i], kmeans.cluster_centers_[0]) > euclidean_distance(tree[max_var_index][i], kmeans.cluster_centers_[1]):
						new_indexes1.append(train_images_index[max_var_index][i])

						new_node1.append(tree[max_var_index][i])

					else:
						new_indexes2.append(train_images_index[max_var_index][i])

						new_node2.append(tree[max_var_index][i])

				elif distance == 1:
					if manhattan_distance(tree[max_var_index][i], kmeans.cluster_centers_[0]) > manhattan_distance(tree[max_var_index][i], kmeans.cluster_centers_[1]):
						new_indexes1.append(train_images_index[max_var_index][i])

						new_node1.append(tree[max_var_index][i])

					else:
						new_indexes2.append(train_images_index[max_var_index][i])

						new_node2.append(tree[max_var_index][i])

				elif distance == 2:
					if cosine_distance(tree[max_var_index][i], kmeans.cluster_centers_[0]) > cosine_distance(tree[max_var_index][i], kmeans.cluster_centers_[1]):
						new_indexes1.append(train_images_index[max_var_index][i])

						new_node1.append(tree[max_var_index][i])

					else:
						new_indexes2.append(train_images_index[max_var_index][i])

						new_node2.append(tree[max_var_index][i])

			tree.append(new_node1)
			tree.append(new_node2)

			leaf_index.append(len(tree) - 2)
			leaf_index.append(len(tree) - 1)

			train_images_index.append(new_indexes1)
			train_images_index.append(new_indexes2)

		k += 1

hierarchical_clustering()

TP_total = 0
F_measure = 0

for i in leaf_index:

	find_cluster = {}

	for j in train_images_index[i]:
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
	false_positives = len(train_images_index[i]) - find_cluster[cluster_index]

	#FN is equal to all labels in train set where its category is the same as
	#the majority label in the current cluster
	#each category has the same number of items
	false_negatives = (train_number/M) - find_cluster[cluster_index]

	precision = true_positives / (true_positives + false_positives)

	recall = true_positives / (true_positives + false_negatives)

	F1_score = 2*((precision*recall) / (precision+recall))

	F_measure += F1_score

print("Purity is: " +str(TP_total/train_number))
print("F_measure is: " + str(F_measure))