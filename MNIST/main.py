##Extarnal libraries
import numpy as np
import scipy.io
from scipy.spatial import distance
import time
import datetime
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#### Local imports
from load_data import seperate_and_split_data
from plot_MNIST import display_image
from cluster import cluster

data = scipy.io.loadmat('MNist_ttt4275/data_all.mat')


def KNN(train_data,train_lable,ref_vector,k_neighbour):


    dist_to_ref = {}
    dist_list = []

    for i, vector in enumerate(train_data):

        dist = distance.sqeuclidean(ref_vector,vector)


        dist_list.append(dist)
        dist_to_ref[dist] = train_lable[i]


    print(dist_to_ref.keys())
    sorted_dist_to_ref = sorted(dist_to_ref)
    #print(sorted_dist_to_ref)

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

#NN_classifier(600,100)




def NN(number_of_samples_train,number_of_samples_test):
    start_time_tot = time.time()
    train_vectors, train_labels, test_vectors, test_labels = seperate_and_split_data(number_of_samples_train, number_of_samples_test)


    dist = distance.cdist(test_vectors,train_vectors,"sqeuclidean")

    label_store = []
    vector_store =[]

    for count, item in enumerate(dist):
        item_argmin = np.argmin(item)

        label_store.append(train_labels[item_argmin])
        vector_store.append(item_argmin)


    error = 0 #stores the number of false predictions to calculate error rate
    false_predictions_cord = [] #Stores info about the false predictions to plot
    correct_predictions_cord = [] # store info about the correct predictions to plot.

    for i in range(len(test_labels)):
        if test_labels[i] != label_store[i]:
            error += 1
            false_predictions_cord.append([test_labels[i],label_store[i],test_vectors[i],train_vectors[vector_store[i]]])
        else:
            correct_predictions_cord.append([test_labels[i],label_store[i],test_vectors[i],train_vectors[vector_store[i]]])

    display_image(false_predictions_cord,1)
    display_image(correct_predictions_cord, 1)

    #Utelise the sklearn library to find and plot the confusion matrix
    cm = confusion_matrix(label_store, test_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    error_rate = error / number_of_samples_test * 100

    print("The error rate is:", round(error_rate, 2))








    total_time = time.time() - start_time_tot
    print("Total time:", str(datetime.timedelta(seconds=total_time)))




#NN(60000,10000)

def NN_clustering(number_of_samples_train,number_of_samples_test):
    start_time_tot = time.time()
    train_vectors, train_labels, test_vectors, test_labels = seperate_and_split_data(number_of_samples_train, number_of_samples_test)
    start_time_cluster = time.time()
    store_cluster = cluster(train_labels,train_vectors)


    total_time = time.time() - start_time_cluster
    print("Total time:", str(datetime.timedelta(seconds=total_time)))




    dist = distance.cdist(test_vectors,store_cluster,"sqeuclidean")





    label = []
    count = 0
    for i in range(10):
        for j in range(64):
            label.append(i)





    label_store =[]

    for count, item in enumerate(dist):


        item_argmin = np.argmin(item)
        label_store.append(label[item_argmin])





    error = 0 #stores the number of false predictions to calculate error rate
    false_predictions_cord = [] #Stores info about the false predictions to plot
    correct_predictions_cord = [] # store info about the correct predictions to plot.

    for i in range(len(test_labels)):

        if test_labels[i][0] != label_store[i]:
            error += 1

    #display_image(false_predictions_cord,1)
    #display_image(correct_predictions_cord, 1)

    #Utelise the sklearn library to find and plot the confusion matrix
    cm = confusion_matrix(label_store, test_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    error_rate = error / number_of_samples_test * 100

    print("The error rate is:", round(error_rate, 2))





    total_time = time.time() - start_time_tot
    print("Total time:", str(datetime.timedelta(seconds=total_time)))

#NN_clustering(6000,1000)



def K_NN(number_of_samples_train,number_of_samples_test,K):
    start_time_tot = time.time()
    train_vectors, train_labels, test_vectors, test_labels = seperate_and_split_data(number_of_samples_train, number_of_samples_test)
    store_cluster = cluster(train_labels, train_vectors)

    dist = distance.cdist(test_vectors,store_cluster,"sqeuclidean")
    print("finished eq")

    label_store = []
    dist_store =[]

    label = []

    for i in range(10):
        for j in range(64):
            label.append(i)



    for count, item in enumerate(dist):
        item_argmin = np.argsort(item)
        item_argmin = item_argmin[:K]

        K_label_store = []
        dist_store = []
        for i in item_argmin:

            K_label_store.append(label[i])
            dist_store.append(item[i])

        #numbers = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0] ]
        numbers = [0,0,0,0,0,0,0,0,0,0]
        avg_dist = [0,0,0,0,0,0,0,0,0,0]

        label_NN = 0
        for c,j in enumerate(K_label_store):

            numbers[j] += 1
            avg_dist[j] = sum([dist_store[c],avg_dist[j]])/2

        arg_max_1 = np.argmax(numbers)
        number_1_store = numbers[arg_max_1]
        dist_1_store = avg_dist[arg_max_1]

        numbers.pop(arg_max_1)
        arg_max_2 = np.argmax(numbers)
        number_2_store = numbers[arg_max_2]
        dist_2_store = avg_dist[arg_max_2]
        numbers.pop(arg_max_2)

        if number_1_store == number_2_store:
            while_bool = True

            while (while_bool):
                if dist_1_store < dist_2_store:
                    label_NN = arg_max_1
                else:
                    label_NN = arg_max_2
                    arg_max_1 = arg_max_2
                    number_1_store = number_2_store
                    dist_1_store = dist_2_store
                arg_max_2 = np.argmax(numbers)
                number_2_store = numbers[arg_max_2]
                numbers.pop(arg_max_2)

                if number_1_store != number_2_store:
                    while_bool = False
        else:
            label_NN = arg_max_1





        label_store.append(label_NN)











    error = 0 #stores the number of false predictions to calculate error rate


    for i in range(len(test_labels)):

        if test_labels[i][0] != label_store[i]:
            #print("real:", test_labels[i][0], " Pred:", label_store[i])
            error += 1

    #Utelise the sklearn library to find and plot the confusion matrix
    cm = confusion_matrix(label_store, test_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    error_rate = error / number_of_samples_test * 100

    print("The error rate is:", round(error_rate, 2))








    total_time = time.time() - start_time_tot
    print("Total time:", str(datetime.timedelta(seconds=total_time)))

K_NN(6000,1000,7)

plt.show()