# -*- coding: utf-8 -*-



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