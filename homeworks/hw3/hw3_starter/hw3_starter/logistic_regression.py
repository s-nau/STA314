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


    n, d = x_train.shape

    
    weights = np.random.rand(d + 1,1)
# =============================================================================
    val_test_vector = []

    for _ in range(hyperparameters["num_iterations"]):
        f,df, y = logistic(
            weights, x_train, y_train, hyperparameters)
        update = ( hyperparameters["learning_rate"]/n) * df
        weights = np.subtract(weights, update)

    #val_test = evaluate(y_valid, logistic_predict(weights, x_valid))

    #val_test = evaluate(y_valid, logistic_predict(weights, x_valid))


    #test_value = evaluate(y_test, logistic_predict(weights, x_test))
    #train_value = evaluate(y_train, logistic_predict(weights, x_train))
    val_value = evaluate(y_valid, logistic_predict(weights, x_valid))

  
    return(val_value[0], val_value[1], weights, lr, num_iters)

if __name__ == "__main__":

    # Load all necessary datasets:
    # x_train, y_train = load_train()
    # If you would like to use digits_train_small, please uncomment this line:
    x_train, y_train = load_train_small()
    x_valid, y_valid = load_valid()
    x_test, y_test = load_test()

    validations = []
    for num_iters in [50, 100, 500,1000]:
        for lr in [.001, .01, .1, .5, 1]:
            hyperparameters = {
            "learning_rate": lr,
            "weight_regularization": 0.,
            "num_iterations": num_iters
            }  
            x =  run_logistic_regression()
            validations.append(x)
    for i in validations:
        print(i[0], i[1])
    model_choice =  min(validations, key= lambda item:item[0])   
    index = validations.index(model_choice)
    weights = validations[index][-3]
    train_value = evaluate(y_train, logistic_predict(weights, x_train))
    test_value = evaluate(y_test, logistic_predict(weights, x_test))
    val_value = [validations[index][0], validations[index][1]]
    learning_rate_choice = validations[index][-2]
    num_iters_choice = validations[index][-1]
    print(test_value, "test")
    print(train_value, "train")
    print(val_value, "validation")
    print(learning_rate_choice)
    print(num_iters_choice)
# =============================================================================
# 
# =============================================================================
