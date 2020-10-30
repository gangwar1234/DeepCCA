import torch
import torch.nn as nn
import numpy as np
from linear_cca import linear_cca
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
from DeepCCAModels import DeepCCA
from utils import load_data, svm_classify
from utils1 import prepare_tensors_from_data_new
import time
import logging
try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle

import gzip
import numpy as np
import torch.nn as nn
from my_classes import Dataset
from load_test_dot import get_test_data
from load_train_dot import get_train_data
from scipy import sparse
torch.set_default_tensor_type(torch.DoubleTensor)


class Solver():
    def __init__(self, model, linear_cca, outdim_size, epoch_num, batch_size, learning_rate, reg_par,train_specs,test_specs, device=torch.device('cpu')):
        self.model = nn.DataParallel(model)
        self.model.to(device)
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.loss = model.loss
        self.train_specs = train_specs
        self.test_specs = test_specs
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(), lr=learning_rate, weight_decay=reg_par)
        self.device = device

        self.linear_cca = linear_cca

        self.outdim_size = outdim_size

        formatter = logging.Formatter(
            "[ %(levelname)s : %(asctime)s ] - %(message)s")
        logging.basicConfig(
            level=logging.DEBUG, format="[ %(levelname)s : %(asctime)s ] - %(message)s")
        self.logger = logging.getLogger("Pytorch")
        #fh = logging.FileHandler("DCCA.log")
        #fh.setFormatter(formatter)
        #self.logger.addHandler(fh)

        #self.logger.info(self.model)
        #self.logger.info(self.optimizer)

    def fit(self, x1, x2, vx1=None, vx2=None, tx1=None, tx2=None, checkpoint='checkpoint.model'):

        #training_set = Dataset(X_data_train,Y_data_train)
        #trainloader  = torch.utils.data.DataLoader(training_set, **params)

        #testing_set = Dataset(X_data_test,Y_data_test)
        #testloader  = torch.utils.data.DataLoader(testing_set, **params)
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """
        #x1.to(self.device)
        #x2.to(self.device)

        '''
        with torch.no_grad():
            x2temp = sparse.lil_matrix((self.train_specs['train_length'], self.train_specs['num_labels']))
            for i in x2:
                x2temp[i] = 1 
            x2 = x2temp
        '''

        data_size = x1.shape[0]
        
        print('type(x1) ',type(x1))
        print(x1.shape)

        print('type(x2) ',type(x2))
        print(x2.shape)

        #data_size = x1.size(0)

        if vx1 is not None and vx2 is not None:
            best_val_loss = 0
            vx1.to(self.device)
            vx2.to(self.device)
        if tx1 is not None and tx2 is not None:
            tx1.to(self.device)
            tx2.to(self.device)

        train_losses = []
        for epoch in range(self.epoch_num):
            epoch_start_time = time.time()
            self.model.train()
            batch_idxs = list(BatchSampler(RandomSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=False))
            for batch_idx in batch_idxs:
                self.optimizer.zero_grad()
                batch_x1 = x1[batch_idx, :].toarray()#.to(self.device)
                batch_x2 = x2[batch_idx, :].toarray().astype('int')#.to(self.device)
                batch_x1,batch_x2 = prepare_tensors_from_data_new(batch_x1,batch_x2)
                batch_x1 = batch_x1.to(self.device)
                batch_x2 = batch_x2.to(self.device)
                o1, o2 = self.model(batch_x1, batch_x2)
                loss = self.loss(o1, o2)
                train_losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
            train_loss = np.mean(train_losses)

            info_string = "Epoch {:d}/{:d} - time: {:.2f} - training_loss: {:.4f}"
            if vx1 is not None and vx2 is not None:
                with torch.no_grad():
                    self.model.eval()
                    val_loss = self.test(vx1, vx2)
                    info_string += " - val_loss: {:.4f}".format(val_loss)
                    if val_loss < best_val_loss:
                        self.logger.info(
                            "Epoch {:d}: val_loss improved from {:.4f} to {:.4f}, saving model to {}".format(epoch + 1, best_val_loss, val_loss, checkpoint))
                        best_val_loss = val_loss
                        torch.save(self.model.state_dict(), checkpoint)
                    else:
                        self.logger.info("Epoch {:d}: val_loss did not improve from {:.4f}".format(
                            epoch + 1, best_val_loss))
            else:
                torch.save(self.model.state_dict(), checkpoint)
            epoch_time = time.time() - epoch_start_time
            self.logger.info(info_string.format(
                epoch + 1, self.epoch_num, epoch_time, train_loss))
        # train_linear_cca
        if self.linear_cca is not None:
            _, outputs = self._get_outputs(x1, x2)
            self.train_linear_cca(outputs[0], outputs[1])

        checkpoint_ = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint_)
        if vx1 is not None and vx2 is not None:
            loss = self.test(vx1, vx2)
            self.logger.info("loss on validation data: {:.4f}".format(loss))

        if tx1 is not None and tx2 is not None:
            loss = self.test(tx1, tx2)
            self.logger.info('loss on test data: {:.4f}'.format(loss))

    def test(self, x1, x2, use_linear_cca=False):
        with torch.no_grad():
            losses, outputs = self._get_outputs(x1, x2)

            if use_linear_cca:
                return np.mean(losses), outputs
                #print("Linear CCA started!")
                #outputs = self.linear_cca.test(outputs[0], outputs[1])
                #return np.mean(losses), outputs
            else:
                return np.mean(losses)

    def train_linear_cca(self, x1, x2):
        self.linear_cca.fit(x1, x2, self.outdim_size)

    def _get_outputs(self, x1, x2):
        with torch.no_grad():
            self.model.eval()
            data_size = x1.shape[0]
            #data_size = self.train_specs['train_length']
            #data_size = x1.size(0)
            batch_idxs = list(BatchSampler(SequentialSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=False))
            losses = []
            outputs1 = []
            outputs2 = []
            for batch_idx in batch_idxs:
                #batch_x1 = x1[batch_idx, :]
                #batch_x2 = x2[batch_idx, :]
                batch_x1 = x1[batch_idx, :].toarray()#.to(self.device)
                batch_x2 = x2[batch_idx, :].toarray().astype('int')#.to(self.device)
                batch_x1,batch_x2 = prepare_tensors_from_data_new(batch_x1,batch_x2)
                batch_x1 = batch_x1.to(self.device)
                batch_x2 = batch_x2.to(self.device)
                o1, o2 = self.model(batch_x1, batch_x2,False)
                outputs1.append(o1)
                outputs2.append(o2)
                loss = self.loss(o1, o2)
                losses.append(loss.item())
        outputs = [torch.cat(outputs1, dim=0).cpu().numpy(),
                   torch.cat(outputs2, dim=0).cpu().numpy()]
        return losses, outputs


if __name__ == '__main__':
    ############
    # Parameters Section

    device = torch.device('cuda')
    print("Using", torch.cuda.device_count(), "GPUs")

    # the path to save the final learned features
    save_to = '/content/DeepCCA/new_features_colab.gz'

    # the size of the new space learned by the model (number of the new features)
    #outdim_size = 30
    outdim_size = 20

    # size of the input for view 1 and view 2

    train_specs, train1, train2 = get_train_data()
    #print('len(train2) ',len(train2))
    #print('type(train2[0]) ',type(train2[0]))
    #print('len(train2[0]) ',len(train2[0]))
    #exit(0)
    with torch.no_grad():
        train2temp = sparse.lil_matrix((train_specs['train_length'], train_specs['num_labels']))
        for i in range(len(train2)):
            for j in train2[i]:
                train2temp[int(i),int(j)] = 1
            #train2temp[i] = 1 
        train2 = train2temp
    test_specs, test1, test2 = get_test_data()
    print('len(test2) ',len(test2))

    with torch.no_grad():
        test2temp = sparse.lil_matrix((test_specs['test_length'], train_specs['num_labels']))
        for i in range(len(test2)):
            for j in test2[i]:
                test2temp[int(i),int(j)] = 1 
        test2 = test2temp

    input_shape1 = train_specs['num_features']
    input_shape2 = train_specs['num_labels']
    #input_shape2 = 784

    # number of layers with nodes in each one

    #layer_sizes1 = [1024,512,outdim_size]
    #layer_sizes2 = [1024,512,outdim_size]


    layer_sizes1 = [1024,512,outdim_size]
    layer_sizes2 = [1024,512,outdim_size]

    #layer_sizes1 = [1024, 1024, 1024, outdim_size]
    #layer_sizes2 = [1024, 1024, 1024, outdim_size]

    # the parameters for training the network
    #learning_rate = 1e-3
    learning_rate = 1e-5
    epoch_num = 20
    batch_size = 80

    # the regularization parameter of the network
    # seems necessary to avoid the gradient exploding especially when non-saturating activations are used
    reg_par = 1e-5

    # specifies if all the singular values should get used to calculate the correlation or just the top outdim_size ones
    # if one option does not work for a network or dataset, try the other one
    use_all_singular_values = False
    #use_all_singular_values = True

    # if a linear CCA should get applied on the learned features extracted from the networks
    # it does not affect the performance on noisy MNIST significantly
    #apply_linear_cca = True
    apply_linear_cca = True
    # end of parameters section
    ############

    # Each view is stored in a gzip file separately. They will get downloaded the first time the code gets executed.
    # Datasets get stored under the datasets folder of user's Keras folder
    # normally under [Home Folder]/.keras/datasets/
    #data1 = load_data('./noisymnist_view1.gz')
    #data2 = load_data('./noisymnist_view2.gz')
    # Building, training, and producing the new features by DCCA
    model = DeepCCA(layer_sizes1, layer_sizes2, input_shape1,
                    input_shape2, outdim_size, use_all_singular_values, device=device).double()
    l_cca = None
    if apply_linear_cca:
        l_cca = linear_cca()

    
    
    solver = Solver(model, l_cca, outdim_size, epoch_num, batch_size,
                    learning_rate, reg_par,train_specs,test_specs, device=device)
    #train1, train2 = data1[0][0], data2[0][0]
    #val1, val2 = data1[1][0], data2[1][0]
    #test1, test2 = data1[2][0], data2[2][0]



    solver.fit(train1, train2, None, None, None, None)
    # TODO: Save l_cca model if needed

    #set_size = [0, train1.size(0), train1.size(0) + val1.size(0), train1.size(0) + val1.size(0) + test1.size(0)]
    #loss, outputs = solver.test(torch.cat([train1, val1, test1], dim=0), torch.cat([train2, val2, test2], dim=0), apply_linear_cca)
    loss, outputs_train = solver.test(train1, train2, apply_linear_cca)
    loss, outputs_test = solver.test(test1, test2, apply_linear_cca)

    #print('type(outputs_train) ', type(outputs_train))
    #print('type(outputs_test) ',  )

    print('features before saving')
    print('train1.shape ',outputs_train[0].shape)
    print('train2.shape ',outputs_test[0].shape)

    np.savez('Eurlex_Data.npz', trX = outputs_train[0] ,testX = outputs_test[0])
    npzfile = np.load('Eurlex_Data.npz')
    train1 =  npzfile['trX']
    train2 = npzfile['testX']

    print('features after saving')
    print('train1.shape ',train1.shape)
    print('train2.shape ',train2.shape)



    #new_data = []
    # print(outputs)
    #for idx in range(3):
    #    new_data.append([outputs[0][set_size[idx]:set_size[idx + 1], :],outputs[1][set_size[idx]:set_size[idx + 1], :], data1[idx][1]])
    # Training and testing of SVM with linear kernel on the view 1 with new features
    #[test_acc, valid_acc] = svm_classify(new_data, C=0.01)
    #print("Accuracy on view 1 (validation data) is:", valid_acc * 100.0)
    #print("Accuracy on view 1 (test data) is:", test_acc*100.0)
    # Saving new features in a gzip pickled file specified by save_to
    #print('saving new features ...')
    #f1 = gzip.open(save_to, 'wb')
    #thepickle.dump(new_data, f1)
    #f1.close()
    d = torch.load('checkpoint.model')
    solver.model.load_state_dict(d)
    solver.model.parameters()
