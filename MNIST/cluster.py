import numpy as np
from sklearn.cluster import KMeans


def sort_data_to_classes(train_labels, train_vectors):
    """
    Sorts the data in order by lable 0,1,2,3,4,5,6,7,8,9
    :param train_labels: Lables of training data
    :param train_vectors: training vectors
    :return: sorted vector list
    """

    make_list = []
    for i in train_labels:
        # hte labels are on the wrong format so we format it
        make_list.append(i[0])

    sorted_lables = np.sort(make_list)  # sort the lable "list"
    numbers = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # stores the number of 0-9 labels

    for count, i in enumerate(sorted_lables):
        # counts how many of eache lable in the data set.
        numbers[i] += 1

    sorted_lables_arg = np.argsort(make_list)  # sorts the training labels by index
    sorted_vector = np.empty_like(train_vectors)

    for count, item in enumerate(sorted_lables_arg):
        # print(count,": ",train_labels[item])
        sorted_vector[count] = train_vectors[item]  # sorts the training vector arcording to the sorted lables

    return sorted_vector, numbers


def cluster(train_labels, train_vectors):
    """
    Clusters the rainingdata aording to the KmMeans algorithem
    :param train_labels: Lables of data to cluster
    :param train_vectors: Data to cluster
    :return: Clusterd data
    """
    sorted_vector, numbers = sort_data_to_classes(train_labels, train_vectors)  # Sort the list by class
    store_clusters = np.empty((10, 64, 784))
    n_first = 0
    n_last = 0

    for i in range(len(numbers)):
        # Calculates the cluster for class in the sorted data.
        n_last += numbers[i]

        sliced = sorted_vector[n_first:n_last]

        n_first = n_last

        print("progress:", i)

        # calculate the Kmeans with the sklearn library.
        # N_clusters is the numver of clusters to create.
        # Random_state = 0 sets the random varibaol in the calculation for reproductivity of the resukt
        codebook = KMeans(n_clusters=64, random_state=0).fit(sliced).cluster_centers_

        store_clusters[i] = codebook

    to_store = np.empty((64 * 10, 784))
    count = 0
    for i in store_clusters:
        # Re formats to the desierd format.
        for j in i:
            to_store[count] = j
            count += 1

    return to_store
