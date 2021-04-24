import numpy as np
import scipy.io
from scipy.spatial import distance
from load_data import seperate_and_split_data
import time
import datetime

data = scipy.io.loadmat('MNist_ttt4275/data_all.mat')


def KNN(train_data,train_lable,ref_vector,k_neighbour):


    dist_to_ref = {}
    dist_list = []

    for i, vector in enumerate(train_data):

        dist = distance.sqeuclidean(vector,ref_vector)


        dist_list.append(dist)
        dist_to_ref[dist] = train_lable[i]


    print(np.argmin(dist_list))
    sorted_dist_to_ref = sorted(dist_to_ref)

    k_nearest_neighbour = []
    for count, dist in enumerate(sorted_dist_to_ref):

        k_nearest_neighbour.append(dist_to_ref[dist][0])
        if(count == k_neighbour-1):

            return k_nearest_neighbour






def NN_classifier(number_of_samples_train,number_of_samples_test):
    start_time_tot = time.time()
    train_vectors, train_labels, test_vectors, test_labels = seperate_and_split_data(number_of_samples_train, number_of_samples_test)
    error = 0
    for count, vector in enumerate(test_vectors):
        start_time = time.time()

        NN = KNN(train_vectors, train_labels, vector, 1)
        if NN != test_labels[count]:
            error +=1
        if(number_of_samples_test%100 == 0):
            percentage = count/number_of_samples_test*100
            print(round(percentage,2),round(time.time()-start_time_tot,2))
    error_rate = error/number_of_samples_test*100

    print("The error rate is:",round(error_rate,2))
    total_time = time.time()-start_time_tot
    print("Total time:", str(datetime.timedelta(seconds=total_time)))
    return 0

NN_classifier(1000,600)




