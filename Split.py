"""
This script is used to split the data sets in to a traning and a test set
"""
import numpy as np

def split_training_and_test(file_location_1,file_location_2,file_location_3,split_factor):
    """

    :param file_location_1: [type= str] Location of the first datafile
    :param file_location_2:[type= str] Location of the first datafile
    :param file_location_3: [type= str] Location of the first datafile
    :param split_factor:[type = float] what factor of the data set is for training.
    :return: training dataset and testing dataset
    """

    files = [file_location_1,file_location_2,file_location_3]


    data_set_training = [[],[],[],[]]
    data_set_testing = [[], [], [], []]

    for file in files:
        data_set = [[], [], [], []]

        temp_store = np.loadtxt(file,delimiter =",")
        for element in temp_store:

            sepal_length = element[0]
            sepal_width = element[1]
            petal_length = element[2]
            petal_width = element[3]
            data_set[0].append(sepal_length)
            data_set[1].append(sepal_width)
            data_set[2].append(petal_length)
            data_set[3].append(petal_width)
            split = int(split_factor*50)


        if split_factor >= 0.5:
            for i in range(4):

                data_set_training[i] +=data_set[i][:split]
                data_set_testing[i] +=data_set[i][split:]
        else:
            for i in range(4):
                data_set_testing[i] += data_set[i][:split]
                data_set_training[i] += data_set[i][split:]
    data_set_testing = np.array(data_set_testing)
    data_set_training = np.array(data_set_training)
    return data_set_training, data_set_testing

















