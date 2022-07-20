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
    # Load all necessary datasets:
    #x_train, y_train = load_train()
    # If you would like to use digits_train_small, please uncomment this line:
    x_train, y_train = load_train_small()
    x_valid, y_valid = load_valid()
    

    n, d = x_train.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations                                                     #
    #####################################################################
   # lr_n = [
    #    (.0001, 50, 100, 500, 10000), 
     #(.001, 50, 100, 500, 10000), 
      #  (.01, 50, 100, 500, 10000),
       # (.05, 50, 100, 500, 10000),
        #(.1, 50, 100, 500, 10000),
        #(.5, 50, 100, 500, 10000),
        #(1., 50, 100, 500, 10000)]

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    
    # Begin learning with gradient descent
    
    
    ##############################################
    #######################
    # TODO:                                                             #
    # Modify this section to perform gradient descent, create plots,    #
    # compute test error, etc ...                                       #
    #####################################################################

    #wr = 0.
    hyperparameters = {
        "learning_rate": lr,
        "weight_regularization": wr,
        "num_iterations": ni
        }
    x = np.arange(1, ni + 1).tolist()
    weights = np.zeros((d + 1, 1))
    val_test_vector = []
    train_vector = []
    # part b
    # testing the combos in learning rate and number of iterations
    #for j in range(len(lr_n)):
     #   for k in lr_n[j][1::]:
      #      for r in range(k):
    for _ in range(hyperparameters["num_iterations"]):
        update = ( hyperparameters["learning_rate"]/n) * logistic(
            weights, x_train, y_train, hyperparameters)[1]
        weights= np.subtract(weights, update)
        # train_test = evaluate(y_train, logistic_predict(weights, x_train))
            # evaluates on the validation set
    train_test = evaluate(y_train, logistic_predict(weights, x_train))
    train_vector.append(np.asscalar(train_test[0]))
    val_test = evaluate(y_valid, logistic_predict(weights, x_valid))
            # adds the evaluation to our set
        # for part b
    val_test_vector.append(np.asscalar(val_test[0]))
        # for part a
        #val_test_vector.append(val_test[0], weights))
        
    # takes the min of the first value in each tuple
# =============================================================================
#     plt.plot(x, val_test_vector)
#     plt.plot(x, train_vector)
#     plt.legend(["validation curve", "training curve"])
# =============================================================================
    #model_choice = min(val_test_vector)
    # getting all values on test, train and validate for our model
    #test_value = evaluate(y_test, logistic_predict(model_choice[1], x_test))
    #train_value = evaluate(y_train, logistic_predict(model_choice[1], x_train))
    #val_value = evaluate(y_valid, logistic_predict(model_choice[1], x_valid))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #print(hyperparameters["learning_rate"], hyperparameters["num_iterations"])
    #print("test")
    #print(test_value)
    #print("train")
    #print(train_value)
    #print("validation")
    #print(val_value)
    return (val_test_vector,train_vector, weights)
    #return(val_value[0], val_value[1], test_value, train_value, lr, num_iters)

if __name__ == "__main__":
    x_test, y_test = load_test()
    lr = 1
    ni = 1000
    
    validation_values =[]
    training_values = []
    weights = []
    x = [0., 0.001, 0.01, .1 , 1.0]
    for wr in [0., 0.001, 0.01, .1, 1.0]:   
        y = run_logistic_regression()
        validation_values.append(y[0])
        training_values.append(y[1])
        weights.append(y[-1])
    #validation_values = validation_values.reshape(-1,)
    #training_values = training_values.reshape(-1,)
    plt.plot(x, validation_values)
    x = validation_values.index(min(validation_values))
    #plt.plot(x, training_values)
    plt.title("CE =f(lambda) where ni is {} and lr is {}, smaller dataset".format(ni, lr))
    desired_weights = weights[x]
    regularizer = np.square(np.linalg.norm(weights))
    #n,d = np.shape(x_test)
    #ones = np.expand_dims(np.ones(n), axis = 1)
    #expanded_data = np.append(x_test, ones, axis = 1)
    #first_term =(expanded_data @ desired_weights) /n
    predicted_test = logistic_predict(desired_weights, x_test)
    ce, frac_correct = evaluate(y_test, predicted_test)
    print(ce)
    print(frac_correct)
    
   ############### #part a ##################
    #sets = []
    #lr = .001
    #num_iters = 1000
# =============================================================================

#     validations = []
#     for num_iters in [50, 100, 500, 1000]:
#         for lr in [.001, .01, .1, .5, 1]:
#     #for lr in [0.001, .01, 0.1, .5 , 1]:
#             hyperparameters = {
#             "learning_rate": lr,
#             "weight_regularization": 0.,
#             "num_iterations": num_iters
#             }
#            
#             x =  run_logistic_regression()
#             validations.append(x)
#     model_choice =  min(validations)   
#     index = validations.index(model_choice)
#     train_value = validations[index][3]
#     test_value = validations[index][2]
#     val_value = [validations[index][0], validations[index][1]]
#     learning_rate_choice = validations[index][-2]
#     num_iters_choice = validations[index][-1]
#     print(test_value, "test")
#     print(train_value, "train")
#     print(val_value, "validation")
#     print(learning_rate_choice)
#     print(num_iters_choice)
# =============================================================================
