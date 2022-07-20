'''
Question 3 Solution

Implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
import scipy.special
import matplotlib.pyplot as plt

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

    # == your code goes here ==
    for i in range(10):
        sample = data.get_digits_by_label(train_data, train_labels, i)
        means[i] = np.mean(sample, 0)
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

    # == your code goes here ==
    for i in range(10):
        sample = data.get_digits_by_label(train_data, train_labels, i)
        m = np.mean(sample, 0)
        cov_sum = [np.outer(s-m,s-m) for s in sample]
        cov = np.mean(cov_sum, 0)
        cov_adj = cov + (0.01 * np.eye(64))
        covariances[i] = cov_adj
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
    # == your code goes here ==
    for i in range(10):
        log_2pi = 32 * np.log(2 * np.pi)
        log_det = 0.5 * np.log(np.linalg.det(covariances[i]))
        inv_array = np.linalg.inv(covariances[i])
        y = digits-means[i]
        quad = 0.5 * (np.dot(y, inv_array) * y).sum(axis=1)
        likelihoods[:,i] = -log_2pi - log_det - quad
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

    # == your code goes here ==
    gen_l = generative_likelihood(digits, means, covariances)
    denom = scipy.special.logsumexp(gen_l + np.log(0.1), 1)
    denom_expanded = np.array([denom] * 10).T
    return gen_l + np.log(0.1) - denom_expanded
    # ====
    pass

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
    # Compute and return the most likely class
    # == your code goes here ==
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    pred = np.argmax(cond_likelihood, 1)
    return pred
    # ====
    pass

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

    # Evaluation
    train_log_llh = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    test_log_llh = avg_conditional_likelihood(test_data, test_labels, means, covariances)

    print('Train average conditional log-likelihood: ', train_log_llh)
    print('Test average conditional log-likelihood: ', test_log_llh)

    train_posterior_result = classify_data(train_data, means, covariances)
    test_posterior_result = classify_data(test_data, means, covariances)

    train_accuracy = np.mean(train_labels.astype(int) == train_posterior_result)
    test_accuracy = np.mean(test_labels.astype(int) == test_posterior_result)

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
