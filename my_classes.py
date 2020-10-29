
'''
import torch
class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load('data/' + ID + '.pt')
        y = self.labels[ID]

        return X, y

'''
import torch
class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  #def __init__(self, train_data ,test_data , list_IDs, labels):
  def __init__(self, dataX,dataY):
    self.dataX = torch.FloatTensor(dataX.astype('float'))
    self.dataY = torch.FloatTensor(dataY.astype('float'))

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.dataX)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.dataX[index].reshape((1,10,10))
        #X = self.dataX[index].reshape((1,8,8))
        y = self.dataY[index]
        #ID = self.list_IDs[index]

        # Load data and get label
        #X = torch.load('data/' + ID + '.pt')
        #y = self.labels[ID]

        return X, y

'''

class oversampdata(Dataset):
def __init__(self, data):
        self.data = torch.FloatTensor(data.values.astype('float'))
        
    def __len__(self):
        return len(self.data)
def __getitem__(self, index):
        target = self.data[index][-1]
        data_val = self.data[index] [:-1]
        return data_val,target
'''