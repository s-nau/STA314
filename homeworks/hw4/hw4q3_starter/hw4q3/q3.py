'''
HW4 Q3

Implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from math import pi
def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class. You may iterate over
    the possible digits (0 to 9), but otherwise make sure that your code
    is vectorized.

    Arguments
        train_data: size N x 64 numpy array with the images
        train_labels: size N numpy array with corresponding labels
    
    Returns
        means: size 10 x 64 numpy array with the ith row corresponding
               to the mean estimate for digit class i
    '''
    # Initialize array to store means
    means = np.zeros((10, 64))
    train_labels = np.expand_dims(train_labels, axis = 1)
    # joines the axis
    joined_data = np.concatenate((train_data, train_labels), axis = 1)
    for i in list(range(0,9 + 1)):
        ith_data_with_y = joined_data[joined_data[:, - 1] == i]
        ith_data = np.delete(ith_data_with_y, -1, axis = 1)
        m = np.mean(ith_data, axis = 0)
        means[i] = m 
        
    # == YOUR CODE GOES HERE ==
    # ====
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class. You may iterate over
    the possible digits (0 to 9), but otherwise make sure that your code
    is vectorized.

    Arguments
        train_data: size N x 64 numpy array with the images
        train_labels: size N numpy array with corresponding labels
    
    Returns
        covariances: size 10 x 64 x 64 numpy array with the ith row corresponding
               to the covariance matrix estimate for digit class i
    '''
    # Initialize array to store covariances
    covariances = np.zeros((10, 64, 64))
    means = compute_mean_mles(train_data, train_labels)
    train_labels = np.expand_dims(train_labels, axis = 1)
    # joines the axis
    joined_data = np.concatenate((train_data, train_labels), axis = 1)
    for i in list(range(0, 9 + 1)):
        ith_data_with_y = joined_data[joined_data[:, - 1] == i] # filter for label i
        
        ith_data = np.delete(ith_data_with_y, -1, axis = 1) # remove the label
        n,d = np.shape(ith_data)
        m = np.expand_dims(means[i], axis = 1)
        I = np.expand_dims(np.ones(n), axis = 1)
        first = (ith_data - I@m.T).T
        second = ith_data - I @ m.T
        cov = first @ second/n
        cov_I = 0.01 * np.ones(np.shape(cov))
        cov = cov + cov_I # for stability
        covariances[i] = cov
        
    
    # == YOUR CODE GOES HERE ==
    # ====
    return covariances

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood log p(x|t). You may iterate over
    the possible digits (0 to 9), but otherwise make sure that your code
    is vectorized.

    Arguments
        digits: size N x 64 numpy array with the images
        means: size 10 x 64 numpy array with the 10 class means
        covariances: size 10 x 64 x 64 numpy array with the 10 class covariances
    
    Returns
        likelihoods: size N x 10 numpy array with the ith row corresponding
               to logp(x^(i) | t) for t in {0, ..., 9}
    '''
    N = digits.shape[0]
    likelihoods = np.zeros((N, 10))
    d = digits.shape[1]
    for t in list(range(0,9 + 1)):
        m = np.expand_dims(means[t], axis = 1)
        c = covariances[t]
        det = np.linalg.det(c)
        inv = np.linalg.inv(c)
        first = -d/2*np.log(2*pi)
        second = -.5*np.log(det)
        I = np.expand_dims(np.ones(N), axis = 1)
        third = (digits -  m.T)
        fourth = (digits -  m.T).T
        # this gives us our desired matrix, all we have to do is consider how 
        #matrix multiplication owrks in order to get (x^i - \mu_k)@inv@(x^i - mu_k)T
        fifth = -.5*np.expand_dims((third@inv@fourth).diagonal(), axis = 1)
        ll = (first + second + fifth)
        likelihoods[::, t] = np.squeeze(ll) # squueze helps with formating, 
        # does nothign to the data
        
    # == YOUR CODE GOES HERE ==
    # ====
    return likelihoods


def conditional_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood log p(t|x). Make sure that your code
    is vectorized.

    Arguments
        digits: size N x 64 numpy array with the images
        means: size 10 x 64 numpy array with the 10 class means
        covariances: size 10 x 64 x 64 numpy array with the 10 class covariances
    
    Returns
        likelihoods: size N x 10 numpy array with the ith row corresponding
               to logp(t | x^(i)) for t in {0, ..., 9}
    '''

    N = digits.shape[0]
    likelihoods = np.zeros((N, 10))
    first = np.log(10)
    gll = generative_likelihood(digits, means, covariances)
    second = -scipy.special.logsumexp(gll/10, axis = 1)
    # special scipy function, helps with numerical stability
    for t in list(range(0, 9 + 1)):
        third = gll[::,t]
        ll =  -first + second + third
        likelihoods[::, t] = np.squeeze(ll)
    return likelihoods

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class. 
    Make sure that your code is vectorized.

    Arguments
        digits: size N x 64 numpy array with the images
        means: size 10 x 64 numpy array with the 10 class means
        covariances: size 10 x 64 x 64 numpy array with the 10 class covariances
    
    Returns
        pred: size N numpy array with the ith element corresponding
               to argmax_t log p(t | x^(i))
    '''
    genl = conditional_likelihood(digits, means, covariances)
    N = digits.shape[0]
    pred = np.zeros((N, 1))
    max_values = np.amax(genl, axis = 1)
    # this gives the max along each row
    for t in list(range(0, 9 +1)):
        genlt = genl[::,t]
        # diagonal ensures we only compare in the same column, otherwise it compares
        # each possible pair of elements
        pred[::,0] = np.where(genlt == max_values, t, pred).diagonal()
        
        
        
    
    return pred

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(t^(i) | x^(i)) )

    i.e. the average log likelihood that the model assigns to the correct class label.

    Arguments
        digits: size N x 64 numpy array with the images
        labels: size N x 10 numpy array with the labels
        means: size 10 x 64 numpy array with the 10 class means
        covariances: size 10 x 64 x 64 numpy array with the 10 class covariances
    
    Returns
        average conditional log-likelihood.
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Compute as described above and return
    assert len(digits) == len(labels)
    sample_size = len(digits)
    total_prob = 0
    for i in range(sample_size):
        total_prob += cond_likelihood[i][int(labels[i])]

    return total_prob/sample_size



def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data()

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    
    b = conditional_likelihood(train_data, means, covariances)

    # Evaluation
    train_log_llh = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    test_log_llh = avg_conditional_likelihood(test_data, test_labels, means, covariances)

    print('Train average conditional log-likelihood: ', train_log_llh)
    print('Test average conditional log-likelihood: ', test_log_llh)

    train_posterior_result = classify_data(train_data, means, covariances)
    test_posterior_result = classify_data(test_data, means, covariances)

    # this was changed so that we are only comparing equivalent entries
    train_accuracy = np.mean((train_labels.astype(int) == train_posterior_result).diagonal())
    test_accuracy = np.mean((test_labels.astype(int) == test_posterior_result).diagonal())

    print('Train posterior accuracy: ', train_accuracy)
    print('Test posterior accuracy: ', test_accuracy)

    for i in range(10):
        (e_val, e_vec) = np.linalg.eig(covariances[i])
        # In particular, note the axis to access the eigenvector
        curr_leading_evec = e_vec[:,np.argmax(e_val)].reshape((8,8))
        plt.subplot(3,4,i+1)
        plt.imshow(curr_leading_evec, cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()
