### External libraries ###
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt

### Internal files ###
from Split import split_training_and_test
from math_world import get_MSE_gradient
from plot import plot_MSE,plot_histogram


#### Contants #####
C=3 # the tree difrent flowers wee have
#features = 4

###################






def training_linear_classifiers(training_set,alpha,training_const,features):
    W = np.zeros((C, features + 1))
    MSE_store =[]
    W_store =[]

    print("training started")
    count = 0
    MSE_delta = 1
    while(MSE_delta >0.00001):

        grad_MSE,MSE = get_MSE_gradient(training_set, W, training_const, features)
        if(count > 0):
            MSE_delta =np.abs( (MSE_store[-1] - MSE[0][0])/2)
        MSE_store.append(MSE[0][0])
        W -= alpha * grad_MSE
        W_store.append(W)
        if(count > 100000):
            print("Error: iterations exceeded 100 000")
            print("MSE delta was ",MSE_delta)
            print("last computed MSE was",MSE[0][0])
            return

        count += 1


    print("Traning finished. iterations:",count)



    return W,W_store,MSE_store





def verification_linear_classifiers(W,test_set,testing_const,features):
    """

    :param W: The weithed matrix trained in training_linear_training_linear_classifiers
    :param test_set: The verification set used to control the predictions produced
    :param testing_const: length of the rows in the test set
    :param features:
    :return:
    """

    x_add_ones = np.ones((5, int(len(test_set[0]))))
    x_k = np.ones((features +1,1))
    predictions_list =[]
    real_list = [0]*testing_const +[1]*testing_const +[2]*testing_const


    for i in range(features):
        x_add_ones[i] = test_set[i]


    for j in range(testing_const*3):
        for i in range((features+1)):
            x_k[i] = x_add_ones[i, j]

        predictions = np.matmul(W,x_k)
        predicted_flower = np.argmax(predictions)
        predictions_list.append(predicted_flower)


    error_rate = 0
    for i in range(testing_const*3):
        if real_list[i] != predictions_list[i]:

            error_rate +=1


    print("\nError rate:",round(error_rate/(testing_const*3)*100,1),"%")
    cm = confusion_matrix(real_list, predictions_list)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["setosa","versicolor","virginica"])

    disp.plot()

    return error_rate, predictions_list,real_list



############## Excecution of tasks ##################3


"""
To execute the tasks belowe change the desierd tasks to 1.
"""

task_1A = 0
task_1B = 0
task_1C = 0
task_1D = 0

task_2A = 0
task_2B_2_features =0
task_2B_1_feature =0



####### TASK 1A #######
if(task_1A or task_1B):
    training_set, test_set = split_training_and_test("Iris_TTT4275/class_1","Iris_TTT4275/class_2", "Iris_TTT4275/class_3", 0.6)

####### TASK 1B #######

### Tuning alpha
alpha = 0.008
MSE_dictionary = {}

if(task_1B):
    while(alpha>= 0.002):
        print("\nTraining with alpha:",alpha)
        W_last, W_list, MSE_list = training_linear_classifiers(training_set, 10000, alpha, training_const=30, features=4)
        MSE_dictionary[str(alpha)] = MSE_list
        alpha -= 0.002
    plot_MSE(MSE_dictionary)




####### TASK 1C######
if(task_1C):
    print("TASK 1C")
    W_last, W_list, MSE_list = training_linear_classifiers(training_set, 0.004, training_const=30, features=4)
    ### TEsting on testset##
    verification_linear_classifiers(W_last,test_set,testing_const=20,features=4)
    ##Testing on training set##
    verification_linear_classifiers(W_last,training_set,testing_const=30,features=4)


###### TASK 1D ######
if(task_1D):
    training_set, test_set = split_training_and_test("Iris_TTT4275/class_1","Iris_TTT4275/class_2", "Iris_TTT4275/class_3", 0.4)


    print("TASK 1D")


    W_last, W_list, MSE_list = training_linear_classifiers(test_set, 0.004, training_const=20, features=4)

    ### Testing on testset##
    print("testing on: Test set")
    verification_linear_classifiers(W_last,test_set,testing_const=20,features=4)
    ##Testing on training set##
    print("testing on: Training set")
    verification_linear_classifiers(W_last,training_set,testing_const=30,features=4)

#####################################################################3

#### TASK 2A#####
if(task_2A):
    training_set, test_set = split_training_and_test("Iris_TTT4275/class_1","Iris_TTT4275/class_2", "Iris_TTT4275/class_3", 0.6)
    plot_histogram(training_set,test_set)



    training_set = np.delete(training_set, 1, 0)#deleting the sepal length from
    test_set = np.delete(test_set, 1, 0)


    W_last, W_list, MSE_list = training_linear_classifiers(training_set, 0.006, training_const=30, features=3)
    verification_linear_classifiers(W_last,test_set,testing_const=20,features=3)

##### TASK 2B ####
if(task_2B_2_features):
    training_set, test_set = split_training_and_test("Iris_TTT4275/class_1","Iris_TTT4275/class_2", "Iris_TTT4275/class_3", 0.6)
    training_set = np.delete(training_set, 0, 0)#deleting the sepal length from
    test_set = np.delete(test_set, 0, 0)

    training_set = np.delete(training_set, 0, 0)#deleting the sepal length from
    test_set = np.delete(test_set, 0, 0)

    W_last, W_list, MSE_list = training_linear_classifiers(training_set, 0.006, training_const=30, features=2)
    verification_linear_classifiers(W_last, test_set, testing_const=20, features=2)

if(task_2B_1_feature):
    training_set, test_set = split_training_and_test("Iris_TTT4275/class_1","Iris_TTT4275/class_2", "Iris_TTT4275/class_3", 0.6)
    training_set = np.delete(training_set, 0, 0)#deleting the sepal length from
    test_set = np.delete(test_set, 0, 0)

    training_set = np.delete(training_set, 0, 0)#deleting the sepal length from
    test_set = np.delete(test_set, 0, 0)

    training_set = np.delete(training_set, 0, 0)  # deleting the sepal length from
    test_set = np.delete(test_set, 0, 0)

    W_last, W_list, MSE_list = training_linear_classifiers(training_set, 0.006, training_const=30, features=1)
    verification_linear_classifiers(W_last, test_set, testing_const=20, features=1)


plt.show()