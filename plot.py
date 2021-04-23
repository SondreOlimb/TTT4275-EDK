import matplotlib.pyplot as plt
import numpy as np



def plot_MSE(MSE_dictionary):
    plt.figure()
    for alpha, MSE_list in MSE_dictionary.items():

        plt.plot(MSE_list,label=alpha)

    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("MSE")

def plot_histogram(training_set,test_set):
    size_info =["sepal length", "sepal width", "petal length", "petal width"]


