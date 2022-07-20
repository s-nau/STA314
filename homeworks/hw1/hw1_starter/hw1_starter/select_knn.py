# -*- coding: utf-8 -*-

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
        neigh = KNeighborsClassifier(n_neighbors = k)
        t = neigh.fit(train_X, train_Y)
        s = neigh.score(val_X, val_Y)
            

        lst_of_models.append((s, t, k)) # k- 1 because indexing starts at 0 for lists
    print(lst_of_models)    
    return max(lst_of_models,key = lambda t: t[0])[1], max(lst_of_models,key = lambda t: t[0])[2] # 
# the max function returns the tuple with the highest s value, 
#this is because of the key
    
