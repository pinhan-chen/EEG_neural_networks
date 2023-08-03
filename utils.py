import numpy as np

def data_prep(X, y, sub_sample, average, noise):
    """
    from the discussion 9 coding notebook
    """
    
    total_X = None
    total_y = None
    
    # Trimming the data (sample,22,1000) -> (sample,22,500)
    X = X[:, :, 0:500]
    # print('Shape of X after trimming:',X.shape)
    
    # Maxpooling the data (sample,22,1000) -> (sample,22,500/sub_sample)
    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, sub_sample), axis=3)
    
    total_X = X_max
    total_y = y
    # print('Shape of X after maxpooling:',total_X.shape)
    
    # Averaging + noise 
    X_average = np.mean(X.reshape(X.shape[0], X.shape[1], -1, average),axis=3)
    X_average = X_average + np.random.normal(0.0, 0.5, X_average.shape)
    
    total_X = np.vstack((total_X, X_average))
    total_y = np.hstack((total_y, y))
    # print('Shape of X after averaging+noise and concatenating:',total_X.shape)
    
    # Subsampling
    
    for i in range(sub_sample):
        
        X_subsample = X[:, :, i::sub_sample] + \
                            (np.random.normal(0.0, 0.5, X[:, :,i::sub_sample].shape) if noise else 0.0)
            
        total_X = np.vstack((total_X, X_subsample))
        total_y = np.hstack((total_y, y))
        
    
    # print('Shape of X after subsampling and concatenating:',total_X.shape)
    return total_X, total_y

def load_data(data_path, subjects=[0, 1, 2, 3, 4, 5, 6, 7, 8]):
    """
    loading data from the numpy files
    """
    X_train_valid = np.load(data_path + "X_train_valid.npy")
    X_test = np.load(data_path + "X_test.npy")

    # Change the class labels from 769 - 772 into 0 - 3
    y_train_valid = np.load(data_path + "y_train_valid.npy") - 769
    y_test = np.load(data_path + "y_test.npy") - 769

    person_train_valid = np.load(data_path + "person_train_valid.npy")
    person_test = np.load(data_path + "person_test.npy")

    X_train_valid_s = np.empty(shape=[0, X_train_valid.shape[1], X_train_valid.shape[2]])
    X_test_s = np.empty(shape=[0, X_test.shape[1], X_test.shape[2]])
    
    y_train_valid_s = np.empty(shape=[0])
    y_test_s = np.empty(shape=[0])

    for s in subjects:

        X_train_valid_tmp = X_train_valid[np.where(person_train_valid == s)[0], :, :]
        X_test_tmp = X_test[np.where(person_test == s)[0], :, :]

        y_train_valid_tmp = y_train_valid[np.where(person_train_valid == s)[0]]
        y_test_tmp = y_test[np.where(person_test == s)[0]]

        X_train_valid_s = np.concatenate((X_train_valid_s, X_train_valid_tmp), axis=0)
        X_test_s = np.concatenate((X_test_s, X_test_tmp), axis=0)

        y_train_valid_s = np.concatenate((y_train_valid_s, y_train_valid_tmp))
        y_test_s = np.concatenate((y_test_s, y_test_tmp))

    return X_train_valid_s, y_train_valid_s, X_test_s, y_test_s