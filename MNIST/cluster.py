import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq, kmeans, whiten, kmeans2
import time

from load_data import seperate_and_split_data
from plot_MNIST import display_image
import matplotlib.pyplot as plt

def sort_data_to_classes(train_labels,train_vectors):



    make_list = []
    for i in train_labels:
        make_list.append(i[0])


    sorted_lables = np.sort(make_list)
    numbers = [0,0,0,0,0,0,0,0,0,0]

    for count,i in enumerate(sorted_lables):
        numbers[i] += 1

    sorted_lables_arg =np.argsort(make_list)
    sorted_vector = np.empty_like(train_vectors)


    for count, item in enumerate(sorted_lables_arg):
        #print(count,": ",train_labels[item])
        sorted_vector[count] = train_vectors[item]

    return sorted_vector, numbers








def cluster(train_labels,train_vectors):
    sorted_vector,numbers = sort_data_to_classes(train_labels,train_vectors)
    store_clusters = np.empty((10,64,784))
    n_first=0
    n_last=0



    for i in range(len(numbers)):
        n_last += numbers[i]

        sliced = sorted_vector[n_first:n_last]


        n_first = n_last


        print("progress:", i)

        codebook = KMeans(n_clusters=64, random_state=0).fit(sliced).cluster_centers_

        store_clusters[i] = codebook



    to_store = np.empty((64*10,784))
    count = 0
    for i in store_clusters:
        for j in i:

            to_store[count] = j
            count +=1


    return to_store


