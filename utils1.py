import torch
def prepare_tensors_from_data_new(X_train, Y_train):
    #print('X_train.shape',X_train.shape)
    #print('type(X_train) ',type(X_train))
    #print('Y_train.shape',Y_train.shape)
    #print('type(Y_train) ',type(Y_train))
    #exit(0)
    X_train = torch.from_numpy(X_train).double()
    Y_train = torch.from_numpy(Y_train).double()
    #X_train = X_train.type('torch.LongTensor')
    #Y_train = Y_train.type('torch.FloatTensor')
    return X_train,Y_train
    '''
    X_train = X_train.type('torch.LongTensor')
    Y_train = Y_train.type('torch.FloatTensor')
    X_indices = np.array([list(range(1, X_train.shape[1]+1))]*X_train.shape[0])
    X_data_new = np.zeros(X_train.shape)
    non_zero_indexes = np.nonzero(X_train)
    X_data_new[non_zero_indexes] = 1
    X_data_new = X_data_new*X_indices
    X_TfIdftensor = torch.from_numpy(X_train[:, :, None])
    X_train = torch.from_numpy(X_data_new)
    Y_train = torch.from_numpy(Y_train)
    X_train = X_train.type('torch.LongTensor')
    Y_train = Y_train.type('torch.FloatTensor')
    X_TfIdftensor = X_TfIdftensor.type('torch.FloatTensor')
    print(X_train.shape, X_TfIdftensor.shape, Y_train.shape)
    return X_train, X_TfIdftensor, Y_train
    '''

'''
def prepare_tensors_from_data(X_train, Y_train):
    X_indices = np.array([list(range(1, X_train.shape[1]+1))]*X_train.shape[0])
    X_data_new = np.zeros(X_train.shape)
    non_zero_indexes = np.nonzero(X_train)
    X_data_new[non_zero_indexes] = 1
    X_data_new = X_data_new*X_indices
    X_TfIdftensor = torch.from_numpy(X_train[:, :, None])
    X_train = torch.from_numpy(X_data_new)
    Y_train = torch.from_numpy(Y_train)
    X_train = X_train.type('torch.LongTensor')
    Y_train = Y_train.type('torch.FloatTensor')
    X_TfIdftensor = X_TfIdftensor.type('torch.FloatTensor')
    print(X_train.shape, X_TfIdftensor.shape, Y_train.shape)
    return X_train, X_TfIdftensor, Y_train
'''