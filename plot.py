"""All the plots utilised in main.py is implemented int his file"""
import matplotlib.pyplot as plt
import numpy as np



def plot_MSE(MSE_dictionary):
    """

    :param MSE_dictionary: dictionary of MSE pr iteration for diferent alphas
    :return: none
    """
    plt.figure()
    for alpha, MSE_list in MSE_dictionary.items():

        plt.plot(MSE_list,label="\u03B1="+alpha)

    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("MSE")

    return 0

def plot_histogram(training_set,test_set):

    """

    :param training_set: The training set return from the function defined in Split.py
    :param test_set: The test set return from the function defined in Split.py
    :return: 0
    """

    size_info =["sepal length", "sepal width", "petal length", "petal width"]

    total_set = [[],[],[],[]]
    for i in range(4):

        training=training_set[i].tolist()

        test = test_set[i].tolist()
        tot = training[0:30] +test[0:20] +training[30:60] +test[20:40] + training[60:90] +test[40:60]

        total_set[i] = tot


    fig, axs = plt.subplots(nrows=2, ncols=2)

    axs[0, 0].hist(total_set[0][0:50], 20, density=True, histtype='stepfilled', facecolor='r',alpha=0.75,label ="setosa")
    axs[0, 0].hist(total_set[0][50:100], 20, density=True, histtype='stepfilled', facecolor='g', alpha=0.75,
                   label="versicolor")
    axs[0, 0].hist(total_set[0][100: 150], 20, density=True, histtype='stepfilled', facecolor='b', alpha=0.75,
                   label="virginica")

    axs[0, 0].set_title('Sepal length')
    axs[0, 0].set_ylabel("Number")
    axs[0, 0].set_xlabel("Length [cm]")

    axs[0, 0].legend()

    ###Histogram:Sepal Width



    axs[0, 1].hist(total_set[1][0:50], 20, density=True, histtype='stepfilled', facecolor='r', alpha=0.75,
                   label="setosa")
    axs[0, 1].hist(total_set[1][50:100], 20, density=True, histtype='stepfilled', facecolor='g', alpha=0.75,
                   label="versicolor")
    axs[0, 1].hist(total_set[1][100:150], 20, density=True, histtype='stepfilled', facecolor='b', alpha=0.75,
                   label="virginica")
    axs[0, 1].set_title('sepal width')
    axs[0, 1].set_ylabel("Number")
    axs[0, 1].set_xlabel("Width [cm]")
    axs[0,1].legend()
    ###Histogram:Petal length

    axs[1, 0].hist(total_set[2][0:50], 20, density=True, histtype='stepfilled', facecolor='r', alpha=0.75,
                   label="setosa")
    axs[1, 0].hist(total_set[2][50:100], 20, density=True, histtype='stepfilled', facecolor='g', alpha=0.75,
                   label="versicolor")
    axs[1, 0].hist(total_set[2][100:150], 20, density=True, histtype='stepfilled', facecolor='b', alpha=0.75,
                   label="virginica")

    axs[1, 0].set_title('petal length')
    axs[1, 0].set_ylabel("Number")
    axs[1, 0].set_xlabel("Length [cm]")
    axs[1,0].legend()

    ###Histogram:petal width
    axs[1, 1].hist(total_set[3][0:50], 20, density=True, histtype='stepfilled', facecolor='r', alpha=0.75,
                   label="setosa")
    axs[1, 1].hist(total_set[3][50:100], 20, density=True, histtype='stepfilled', facecolor='g', alpha=0.75,
                   label="versicolor")
    axs[1, 1].hist(total_set[3][100:150], 20, density=True, histtype='stepfilled', facecolor='b', alpha=0.75,
                   label="virginica")
    axs[1, 1].set_title("petal width")
    axs[1, 1].legend()
    axs[1, 1].set_ylabel("Number")
    axs[1, 1].set_xlabel("Width [cm]")

    fig.tight_layout()

    return 0