import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # kill warning about tensorflow

import numpy as np
import sys

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 400)
        self.fc4 = nn.Linear(400, 400)
        self.fc5 = nn.Linear(400, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return F.relu(x)

    def save(self, path):
        torch.save(self.state_dict(), path)




class TrainModel:
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim, device):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self.device = device
        self._model = self._build_model(num_layers, width)


    def _build_model(self, num_layers, width):
        """
        Build and compile a fully connected deep neural network
        """
        net = Net(self._input_dim)
        net.to(self.device)
        return net.double()
        

    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state)
        """
        state =  torch.from_numpy(state).to(self.device)
        return self._model(state).detach().cpu().numpy()
       


    def predict_batch(self, states):
        """
        Predict the action values from a batch of states
        """
        states = torch.from_numpy(states).to(self.device)
        return self._model(states).detach().cpu().numpy()


    def train_batch(self, states, q_sa):
        """
        Train the nn using the updated q-values
        """
        
        #change with Adam optimizer
        optimizer = optim.SGD(self._model.parameters(), lr=self._learning_rate, momentum=0.9)
        # create a loss function
        criterion = nn.MSELoss().to(self.device)
         # run the main training loop
        for i in range(len(states)):
            data, target = (torch.from_numpy(states[i])).to(self.device), (torch.from_numpy(q_sa[i])).to(self.device)
           
            optimizer.zero_grad()
            net_out = self._model(data)
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()
            """
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.data[0]))
            """

    def save_model(self, path):
        """
        Save the current model in the folder as h5 file and a model architecture summary as png
        """
        print(self._model.state_dict())
        torch.save(self._model.state_dict(),os.path.join(path, 'trained_model.pth.tar'))
        #plot_model(self._model, to_file=os.path.join(path, 'model_structure.png'), show_shapes=True, show_layer_names=True)


    @property
    def input_dim(self):
        return self._input_dim


    @property
    def output_dim(self):
        return self._output_dim


    @property
    def batch_size(self):
        return self._batch_size


class TestNaiveModel:
    def __init__(self, input_dim, priorityType):
        self._input_dim     = input_dim
        self._priorityType  = priorityType 
        self.counter        = 0
        self.action         = 0
    
    def predict_one(self, state):
        if self._priorityType == "FR": # Modèle a rotation fixe ( Déja parametré pas besoin d'agir)
            if self.counter <= 30:
                self.counter += 1
                return self.action
            else:
                self.counter = 0
                self.action = (self.action + 1) % 4
                return self.action 
        elif self._priorityType == "LQ": # Rajout de collection de longeur de queue correspondante
            pass

    @property
    def priorityType(self):
        return self._priorityType

class TestModel:
    def __init__(self, input_dim, model_path):
        self._input_dim = input_dim
        self._model = self._load_my_model(model_path)
        self._priorityType = "NN"
        


    def _load_my_model(self, model_folder_path):
        """
        Load the model stored in the folder specified by the model number, if it exists
        """
        model_file_path = os.path.join(model_folder_path, 'trained_model.h5')
        if os.path.isfile(model_file_path):
            model = Net(self.input_dim)
            checkpoints = torch.load(model_file_path)
            model.load_state_dict(checkpoints)
            # print(loaded_model.state_dict())
            return (model.eval()).double()
        else:
            sys.exit("Model number not found")


    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state =  torch.from_numpy(state)
        return self._model(state.double()).detach().numpy()


    @property
    def input_dim(self):
        return self._input_dim

    @property
    def priorityType(self):
        return self._priorityType