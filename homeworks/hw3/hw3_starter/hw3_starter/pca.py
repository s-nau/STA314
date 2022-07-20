"""STA314 Homework 3.

Copyright and Usage Information
===============================

This file is provided solely for the personal and private use of students
taking STA314 at the University of Toronto St. George campus. All forms of
distribution of this code, whether as given or with any changes, are
expressly prohibited.
"""


from utils import *

import matplotlib.pyplot as plt
import scipy.linalg as lin
import numpy as np


def pca(x, k):
    """ PCA algorithm. Given the data matrix x and k,
    return the eigenvectors, mean of x, and the projected data (code vectors).

    Hint: You may use NumPy or SciPy to compute the eigenvectors/eigenvalues.

    :param x: A matrix with dimension N x D, where each row corresponds to
    one data point.
    :param k: int
        Number of dimension to reduce to.
    :return: Tuple of (Numpy array, Numpy array, Numpy array)
        WHERE
        v: A matrix of dimension D x k that stores top k eigenvectors
        mean: A vector of dimension D x 1 that represents the mean of x.
        proj_x: A matrix of dimension k x N where x is projected down to k dimension.
    """
    n, d = x.shape
    
# =============================================================================
#      mean
# =============================================================================
    #xt = x.T
    
    mean = np.mean(x, axis = 0)
    centered_x = x - mean
    #####################################################################
    # TODO:                                                             #
    #####################################################################
# =============================================================================
#     v
# =============================================================================
    
   #v = None
    covmat =( (x - np.ones(d)@mean.T).T@(x-np.ones(d)@mean.T))/n
    eigvec = np.linalg.eig(covmat)[1]
    v = eigvec[:, 0:k]
   
# =============================================================================
#     projection 
# =============================================================================
   # mean = None
    proj_x = v.T@(x - np.ones(d)@mean.T).T
    
   
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return v, mean, proj_x


def show_eigenvectors(v):
    """ Display the eigenvectors as images.
    :param v: NumPy array
        The eigenvectors
    :return: None
    """
    plt.figure(1)
    plt.clf()
    for i in range(v.shape[1]):
        plt.subplot(1, v.shape[1], i + 1)
        plt.imshow(v[:, v.shape[1] - i - 1].reshape(16, 16).T, cmap=plt.cm.gray)
    plt.show()


def pca_classify():
    # Load all necessary datasets:
    x_train, y_train = load_train()
    x_valid, y_valid = load_valid()
    x_test, y_test = load_test()

    # Make sure the PCA algorithm is correctly implemented.
    v, mean, proj_x = pca(x_train, 5)
    # The below code visualize the eigenvectors.
    show_eigenvectors(v)

    #####################################################################
    # TODO:                                                             #
    #####################################################################
    k_lst = [2, 5, 10, 20, 30]
    n, d = np.shape(x_train)
    val_acc = np.zeros(len(k_lst))
    for j, k in enumerate(k_lst):
        v,mean,proj_x = pca(x_train, k)
        show_eigenvectors(v)
        x_tilde = (v @ proj_x + np.ones(d) @ mean.T)
        values = []
        options = []
        for i in range(x_valid.shape[0]):  
            x_valid_i = np.expand_dims(x_valid[i], axis = 1)
            to_min = np.linalg.norm(x_tilde.T - np.expand_dims(
                np.ones(n), axis = 1)@x_valid_i.T, axis = 1)
            opt = np.argmin(to_min)
            options.append(opt)
            y_value = y_train[opt]
            if y_value == y_valid[i]:
                values.append(1)
            else:
                values.append(0)
            #to_append = np.linalg.norm(x_tilde.T[opt] - np.ones(d)*y_valid[i])
        
            
            # For each validation sample, perform 1-NN classifier on
            # the training code vector.
        #    pass
        val_acc[j] = sum(values)/len(values)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    plt.plot(k_lst, val_acc)
    plt.title("accuracy as a function of number of eigenvectors kept")
    plt.show()
    error = []
    v,mean,proj_x = pca(x_train, 5)
    x_tilde = (v @ proj_x + np.ones(d) @ mean.T)
    for r in range(x_test.shape[0]):
         x_test_r = np.expand_dims(x_test[r], axis = 1)
         x_test_2 = np.expand_dims(np.ones(n), axis = 1)@x_test_r.T
         to_min = np.linalg.norm(x_tilde.T - x_test_2, axis = 1)          
         opt = np.argmin(to_min)
         y_value = y_train[opt]
         if y_value == y_test[r]:
             error.append(1)
         else:
             error.append(0)
         #options.append(opt)
         #to_append = np.linalg.norm(x_tilde.T[opt] - np.ones(d)*y_test[i])
         
    print(sum(error)/ len(error))


if __name__ == "__main__":
    pca_classify()
