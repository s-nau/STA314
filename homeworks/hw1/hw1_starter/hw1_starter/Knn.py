# These are imports and you do not need to modify these.

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import sklearn
import random
import math

# ===================================================
# You only need to complete or modify the code below.

def process_data(data, labels):
    """
    Preprocess a dataset of strings into vector representations.

    Parameters
    ----------
        data: numpy array
            An array of N strings.
        labels: numpy array
            An array of N integer labels.

    Returns
    -------
    train_X: numpy array
        Array with shape (N, D) of N inputs.
    train_Y:
        Array with shape (N,) of N labels.
    val_X:
        Array with shape (M, D) of M inputs.
    val_Y:
        Array with shape (M,) of M labels.
    test_X:
        Array with shape (M, D) of M inputs.
    test_Y:
        Array with shape (M,) of M labels.
    """ 
    
    
    # Split the dataset of string into train, validation, and test 
    # Use a 70/15/15 split
    # train_test_split shuffles the data before splitting it 
    # Stratify keeps the proportion of labels the same in each split
    
    # -- WRITE THE SPLITTING CODE HERE -- 
    tr_X, ts_X, tr_Y, ts_Y = train_test_split(
        data, labels, test_size = 0.3, stratify = labels)
  

    Val_X, tes_X, Val_Y, tes_Y = train_test_split(
        ts_X, ts_Y, test_size = 0.5, stratify = ts_Y)
    #print(Val_Y)

    # Preprocess each dataset of strings into a dataset of feature vectors
    # using the CountVectorizer function. 
    # Note, fit the Vectorizer using the training set only, and then
    # transform the validation and test sets.
    
    # -- WRITE THE PROCESSING CODE HERE -- 
    vectorizer = CountVectorizer()
    training_X = vectorizer.fit_transform(tr_X)
    testing_X = vectorizer.transform(tes_X)
    valing_X = vectorizer.transform(Val_X)


    # Return the training, validation, and test set inputs and labels
   
    # -- RETURN THE ARRAYS HERE -- 
    return training_X, tr_Y, valing_X, Val_Y, testing_X, tes_Y

def select_knn_model(train_X, val_X, train_Y, val_Y):
    """
    Test k in {1, ..., 20} and return the a k-NN model
    fitted to the training set with the best validation loss.

    Parameters
    ----------
        train_X: numpy array
            Array with shape (N, D) of N inputs.
        train_X: numpy array
            Array with shape (M, D) of M inputs.
        train_Y: numpy array
            Array with shape (N,) of N labels.
        val_Y: numpy array
            Array with shape (M,) of M labels.

    Returns
    -------
    best_model : KNeighborsClassifier
        The best k-NN classifier fit on the training data 
        and selected according to validation loss.
      best_k : int
        The best k value according to validation loss.
    """
    
    lst_of_models = []
    for k in range(1,21):
        # want values from 1:20
        neigh = KNeighborsClassifier(n_neighbors = k, metric = 'cosine')
        t = neigh.fit(train_X, train_Y)
        s = neigh.score(val_X, val_Y)
            

        lst_of_models.append((s, t, k)) # k- 1 because indexing starts at 0 for lists
    print(lst_of_models)    
    return max(lst_of_models,key = lambda t: t[0])[1], max(lst_of_models,key = lambda t: t[0])[2] # 
# the max function returns the tuple with the highest s value, 
#this is because of the key
    




# You DO NOT need to complete or modify the code below this line.
# ===============================================================


# Set random seed
np.random.seed(3142021)
random.seed(3142021)

def load_data():
    # Load the data
    with open('./clean_fake.txt', 'r') as f:
        fake = [l.strip() for l in f.readlines()]
    with open('./clean_real.txt', 'r') as f:
        real = [l.strip() for l in f.readlines()]

    # Each element is a string, corresponding to a headline
    data = np.array(real + fake)
    labels = np.array([0]*len(real) + [1]*len(fake))
    return data, labels


def main():
    data, labels = load_data()
    train_X, train_Y, val_X, val_Y, test_X, test_Y = process_data(data, labels)
    
    best_model, best_k = select_knn_model(train_X, val_X, train_Y, val_Y)
    test_accuracy = best_model.score(test_X, test_Y)
    print("Selected K: {}".format(best_k))
    print("Test Acc: {}".format(test_accuracy))


if __name__ == '__main__':
    main()
