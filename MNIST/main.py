##Extarnal libraries
import numpy as np
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




def K_NN_calcultaion(dist,label,K):
    """
    Finds the the K snamlest values in each row of the matrix dist, and returns the lable that is
    most prevelant within this K NN. If to lables is equally prevelent it return the lable whos on average closest
    the referance vector.
    :param dist: A matix
    :param label: leabels relevant to the dist matrix
    :param K: numbers of neighbor to calculate
    :return: list of the predicted labels.
    """
    label_store = [] # stores eache label predicted by the alorither

    for count, item in enumerate(dist):
        item_argmin = np.argsort(item) # sorts the list by index

        item_argmin = item_argmin[:K]


        K_label_store = []
        dist_store = []
        for i in item_argmin:
            K_label_store.append(label[i])
            dist_store.append(item[i])


        numbers_lable = [0,0,0,0,0,0,0,0,0,0]
        avg_dist = [0,0,0,0,0,0,0,0,0,0]

        label_NN = 0
        for c,j in enumerate(K_label_store):

            numbers_lable[j] += 1
            if avg_dist[j] == 0:
                avg_dist[j] = dist_store[c]
            else:
                avg_dist[j] = np.average([dist_store[c],avg_dist[j]]) #calculates the average dist


        arg_max_1 = np.argmax(numbers_lable)

        number_1_store = numbers_lable[arg_max_1]
        dist_1_store = avg_dist[arg_max_1]

        numbers_lable[arg_max_1] = 0
        arg_max_2 = np.argmax(numbers_lable)

        number_2_store = numbers_lable[arg_max_2]
        dist_2_store = avg_dist[arg_max_2]
        numbers_lable[arg_max_2] = 0

        if number_1_store == number_2_store: # if more then 1 label is most prevelant it checks which label is closest
            while_bool = True

            while (while_bool):
                if dist_1_store < dist_2_store:
                    label_NN = arg_max_1
                else:
                    label_NN = arg_max_2
                    arg_max_1 = arg_max_2
                    number_1_store = number_2_store
                    dist_1_store = dist_2_store
                arg_max_2 = np.argmax(numbers_lable)
                number_2_store = numbers_lable[arg_max_2]
                numbers_lable[arg_max_2] = 0

                if number_1_store != number_2_store: # if ter is a third most prevalent label
                    while_bool = False

        else:
            label_NN = arg_max_1





        label_store.append(label_NN)
    return label_store





def NN(number_of_samples_train,number_of_samples_test):
    """
    Computes the NN using the squared euclidean distance.

    :param number_of_samples_train: Number of samples from the dataset to use for training
    :param number_of_samples_test: Number of samples from the dataset to use for testing
    :return: none
    """
    start_time_tot = time.time()
    #fetching the data
    train_vectors, train_labels, test_vectors, test_labels = \
        seperate_and_split_data(number_of_samples_train, number_of_samples_test)


    dist = distance.cdist(test_vectors,train_vectors,"sqeuclidean")  #uses the scipy library to calculet the pairwaise
    #distance between the test vectors and train vectors. returns a (number_of_samples_test x number_of_samples_train)

    label_store = []
    vector_store =[]

    for count, item in enumerate(dist):
        # finds the index of the lowest distance in eache row in the dist matrix.
        #stors the lable coresponding to the min value
        item_argmin = np.argmin(item)

        label_store.append(train_labels[item_argmin])
        vector_store.append(item_argmin)


    error = 0 #stores the number of false predictions to calculate error rate
    false_predictions_cord = [] #Stores info about the false predictions to plot
    correct_predictions_cord = [] # store info about the correct predictions to plot.

    for i in range(len(test_labels)):
        # calculates the error by coparing the predicted lables and the real label
        # and stores the correct and false prediction vectors
        if test_labels[i] != label_store[i]:
            error += 1
            false_predictions_cord.append([test_labels[i],label_store[i],test_vectors[i],train_vectors[vector_store[i]]])
        else:
            correct_predictions_cord.append([test_labels[i],label_store[i],test_vectors[i],train_vectors[vector_store[i]]])

    display_image(false_predictions_cord,1) #plots the false prediction vector ralativ to the correct
    display_image(correct_predictions_cord, 1)#plots the correct prediction vector ralativ to the correct

    #Utelise the sklearn library to find and plot the confusion matrix
    cm = confusion_matrix(label_store, test_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    error_rate = error / number_of_samples_test * 100

    print("The error rate is:", round(error_rate, 2))

    total_time = time.time() - start_time_tot
    print("Total time:", str(datetime.timedelta(seconds=total_time)))






def NN_clustering(number_of_samples_train,number_of_samples_test):
    """

    Computes the NN using the squared euclidean distance and clustering of the data.

    :param number_of_samples_train: Number of samples from the dataset to use for training
    :param number_of_samples_test: Number of samples from the dataset to use for testing
    :return: none
    """
    start_time_tot = time.time()
    train_vectors, train_labels, test_vectors, test_labels = seperate_and_split_data(number_of_samples_train, number_of_samples_test)
    start_time_cluster = time.time()
    store_cluster = cluster(train_labels,train_vectors) #Clusters the training data in to 64 clusters. returns a 640x758 matrix


    total_time = time.time() - start_time_cluster
    print("Total time clusteing:", str(datetime.timedelta(seconds=total_time)))
    start_time_NN = time.time()

    dist = distance.cdist(test_vectors,store_cluster,"sqeuclidean") #uses the scipy library to calculet the pairwaise
    #distance between the test vectors and clusterd vector. returns a (number_of_samples_test x 640) matrix


    label = []
    for i in range(10):
        for j in range(64):
            label.append(i)


    label_store =[]

    for count, item in enumerate(dist):
        # finds the index of the lowest distance in eache row in the dist matrix.
        # stores the label coresponding to the min value


        item_argmin = np.argmin(item)
        label_store.append(label[item_argmin])

    error = 0 #stores the number of false predictions to calculate error rate

    for i in range(len(test_labels)):

        if test_labels[i][0] != label_store[i]:
            error += 1

    #Utelise the sklearn library to find and plot the confusion matrix
    cm = confusion_matrix(label_store, test_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    error_rate = error / number_of_samples_test * 100

    print("The error rate is:", round(error_rate, 2))

    total_time = time.time() - start_time_NN
    print("Total time on NN calculation:", str(datetime.timedelta(seconds=total_time)))


    total_time = time.time() - start_time_tot
    print("Total time:", str(datetime.timedelta(seconds=total_time)))





def K_NN(number_of_samples_train,number_of_samples_test,K):
    """

        Computes the K-NN using the squared euclidean distance and clustering of the data.

        :param number_of_samples_train: Number of samples from the dataset to use for training
        :param number_of_samples_test: Number of samples from the dataset to use for testing
        :param K: number of NN to use.
        :return: none
        """

    start_time_tot = time.time()
    train_vectors, train_labels, test_vectors, test_labels = seperate_and_split_data(number_of_samples_train, number_of_samples_test)
    store_cluster = cluster(train_labels, train_vectors)

    dist = distance.cdist(test_vectors,store_cluster,"sqeuclidean")
    label = []
    for i in range(10):
        for j in range(64):
            label.append(i)


    label_store = K_NN_calcultaion(dist=dist,label=label,K=K) # finds the predictions lables


    error = 0 #stores the number of false predictions to calculate error rate


    for i in range(len(test_labels)):

        if test_labels[i][0] != label_store[i]:

            error += 1

    #Utelise the sklearn library to find and plot the confusion matrix
    cm = confusion_matrix(label_store, test_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()


    error_rate = error / number_of_samples_test * 100

    print("The error rate is:", round(error_rate, 2))


    total_time = time.time() - start_time_tot
    print("Total time:", str(datetime.timedelta(seconds=total_time)))

#Run the NN classification
NN(60000,10000) #Obs this fuction wil take about 10 minutes to compute

#Run the NN calcuation with clustering
NN_clustering(60000,10000)

#run the KNN classification with clustering
K_NN(60000,10000,3)

plt.show()#plots all the plots.