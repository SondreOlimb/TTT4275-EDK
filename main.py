from Split import split_training_and_test
from math_world import get_MSE_gradient
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt


#### Contants #####
C=3 # the tree difrent flowers wee have
#features = 4

###################






def training_linear_classifiers(training_set,iterations,alpha,training_const,features):
    W = np.zeros((C, features + 1))
    MSE_store =[]
    W_store =[]

    print("training started")
    count = 0
    MSE_delta = 1
    while(MSE_delta >0.001):
        count +=1
        grad_MSE,MSE = get_MSE_gradient(training_set, W, training_const, features)

        MSE_delta =np.abs( (MSE_store[-1] - MSE[0][0])/2)
        MSE_store.append(MSE[0][0])
        W -= alpha * grad_MSE
        W_store.append(W)
        if(count > 100000):
            print("Error: iterations exceeded 100 000")
            print("MSE delta was ",MSE_delta)
            print("last computed MSE was",MSE[0][0])
            return




    print("Traning finished. iterations:",count)

    print(MSE_store)
    plt.plot(MSE_store)

    return W,W_store,MSE_store





def verification_linear_classifiers(W,test_set,testing_const,features):


    x_add_ones = np.ones((5, int(len(test_set[0]))))
    x_k = np.ones((features +1,1))
    predictions_list =[]
    real_list = [0]*testing_const +[1]*testing_const +[2]*testing_const


    for i in range(features):
        x_add_ones[i] = test_set[i]


    for j in range(60):
        for i in range((features+1)):
            x_k[i] = x_add_ones[i, j]

        predictions = np.matmul(W,x_k)
        predicted_flower = np.argmax(predictions)
        predictions_list.append(predicted_flower)


    for clas in range(C):
        flower =[0,1,2]
        for item in range(20):
            p_f = predictions_list[int(clas*20+item)]
            dif = p_f-flower[clas]
            print("real:", flower[clas] ,"predicted:", p_f, "diff:",dif  )
    cm = confusion_matrix(real_list, predictions_list)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


####### TASK 1B #######
training_set, test_set = split_training_and_test("Iris_TTT4275/class_1","Iris_TTT4275/class_2", "Iris_TTT4275/class_3", 0.6)

W_traind, W_list, MSE_list =training_linear_classifiers(training_set,10000,0.01,training_const=30,features=4)


####### TASK 1C######
#verification_linear_classifiers(W_traind,test_set,testing_const=20,features=4)

