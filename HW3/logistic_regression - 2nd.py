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
import numpy as np
# for the random starting matrix
np.random.seed(1)

def logistic_predict(weights, data):
    """ Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples
          D is the number of features per example

    :param weights: A vector of weights with dimension (D + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x D, where each row corresponds to
    one data point.
    :return: A vector of probabilities with dimension N x 1, which is the output
    to the classifier.
    """
                                                                                                                                                                                                                                                                                                               

    n,d = np.shape(data)

    ones = np.expand_dims(np.ones(n), axis = 1)
    expanded_data = np.append(data, ones, axis = 1)
    z = expanded_data @ weights
    y = 1/(1 + np.exp(z))

    return y


def evaluate(targets, y):
    """ Compute evaluation metrics.

    Note: N is the number of examples
          D is the number of features per example

    :param targets: A vector of targets with dimension N x 1.
    :param y: A vector of probabilities with dimension N x 1.
    :return: A tuple (ce, frac_correct)
        WHERE
        ce: (float) Averaged cross entropy
        frac_correct: (float) Fraction of inputs classified correctly
    """
    t = targets
    y_non_zero = np.where(y == 0, 0.0000000000001, y)
    # this helps with weird log 0 errors
    first_term = -t.T @ np.log(y_non_zero) 
    second_term = (1-t).T @ np.log(1-y_non_zero)
    ce = (first_term -second_term)/ np.size(y)
    

    pred = np.where(y > .5,  1,  0)
    correct = np.where(pred == t, 1 , 0)

    frac_correct = sum(correct)/np.size(correct)

    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """ Calculate the cost of penalized logistic regression and its derivatives
    with respect to weights. Also return the predictions.

    Note: N is the number of examples
          D is the number of features per example

    :param weights: A vector of weights with dimension (D + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x D, where each row corresponds to
    one data point.
    :param targets: A vector of targets with dimension N x 1.
    :param hyperparameters: The hyperparameter dictionary.
    :returns: A tuple (f, df, y)
        WHERE
        f: The average of the loss over all data points, plus a penalty term.
           This is the objective that we want to minimize.
        df: (D+1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of probabilities.
    """
    s = np.shape(weights)[0]
    t = targets
    y = logistic_predict(weights, data)
    w = weights.T
    x = data
    n, d = np.shape(data)
    lambd = hyperparameters["weight_regularization"]
    ce = evaluate(targets, y)[0]
    regularizer = lambd *((np.square(np.linalg.norm(w[0 ,0:s - 1]))))/2
    f = ce + regularizer   
    first_df = (y - t)/n
    ones = np.expand_dims(np.ones(n), axis = 1)
    expanded_data = np.append(data, ones, axis = 1)
    whole_first_df = expanded_data.T@first_df
    reg_df =  (np.expand_dims(lambd* w[0 , 0:s - 1], axis = 1))
    reg_df_exp = np.append(reg_df, np.array([[w[0, -1]]]), axis = 0)
    df = whole_first_df + reg_df_exp
    
    

    return f, df, y


def run_logistic_regression():
    #x_train, y_train = load_train_small()
    x_train, y_train = load_train()
    x_valid, y_valid = load_valid()
    x_test, y_test = load_test()

    n, d = x_train.shape




    
    ni = 1000

    hyperparameters = {
        "learning_rate": lr,
        "weight_regularization": 0.,
        "num_iterations": ni
        }
    x = np.arange(1, ni + 1).tolist() 
    weights = np.random.rand(d + 1, 1)
    val_test_vector = []
    train_vector = []
    for _ in range(hyperparameters["num_iterations"]):
        f,df, y = logistic(
            weights, x_train, y_train, hyperparameters)
        update = ( hyperparameters["learning_rate"]/n) * df
        weights= np.subtract(weights, update)
        train_test = evaluate(y_train, logistic_predict(weights, x_train))

        train_vector.append(np.asscalar(train_test[0]))
        val_test = evaluate(y_valid, logistic_predict(weights, x_valid))

        val_test_vector.append(np.asscalar(val_test[0]))

        
    # takes the min of the first value in each tuple
    plt.plot(x, val_test_vector)
    plt.plot(x, train_vector)
    plt.legend(["validation curve", "training curve"])
    plt.xlabel("num iterations")
    plt.ylabel("cross entropy error")
    plt.title("num iterations v CE  where learning rate is {}".format(lr))
    #model_choice = min(val_test_vector)


if __name__ == "__main__":
    lr = 1
    run_logistic_regression()
    
 