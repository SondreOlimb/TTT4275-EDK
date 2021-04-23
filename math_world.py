"""
All the math fuctions used in the main.py scriptis defined hear
"""
import numpy as np

#####Constants######
C=3 # the tree different flowers wee have
#features = 4

#testing_const = 20

####################

def sigmoid(x_k,W):
    z_i = np.matmul(W,x_k)

    g_k = 1/(1+np.exp(-z_i))

    return g_k



def get_MSE_gradient(data_set_training, W,training_const,features):
    """

    :param data_set_training: The dataset the MSE is to be calculated on
    :param W: W the matirix we will traing
    :return: The gradient og the MSE
    """
    MSE_gradient = np.zeros((C, features + 1))
    MSE = 0

    for count in range(C):

        target_vector = np.zeros((C,1))

        target_vector[count] = 1

        list_target_upper = int((count+1) * training_const)
        list_target_lower = int(count * training_const)

        x_add_ones = np.ones((features+1,training_const))
        for j in range(features):
            x_add_ones[j] = data_set_training[j, list_target_lower:list_target_upper]

        x_k = np.zeros((features+1,1))
        for i in range(training_const):
            for j in range(features+1):
                x_k[j] = x_add_ones[j,i]

            g_k = sigmoid(x_k,W)

            gradient_gk_MSE = g_k-target_vector
            gradient_z_k = g_k*(1 - g_k)
            temp = gradient_gk_MSE*gradient_z_k

            MSE_gradient += np.matmul(temp, x_k.T)

            ####MSE calculation#####

            MSE += np.matmul(gradient_z_k.T,gradient_z_k)
        MSE = MSE/2
    return MSE_gradient, MSE


