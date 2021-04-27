import numpy as np
import scipy.io


def seperate_and_split_data(number_of_samples_train,number_of_samples_test):
    """
    Fetches the data and returns it.
    :param number_of_samples_train:
    :param number_of_samples_test:
    :return: Data fetched
    """
    data = scipy.io.loadmat('MNist_ttt4275/data_all.mat')
    test_labels = data["testlab"]
    test_vectors = data["testv"]
    train_labels = data["trainlab"]
    train_vectors = data["trainv"]


    train_labels = train_labels[:number_of_samples_train]
    train_vectors = train_vectors[:number_of_samples_train]
    test_labels = test_labels[:number_of_samples_test]
    test_vectors = test_vectors[:number_of_samples_test]
    return train_vectors,train_labels,test_vectors,test_labels


