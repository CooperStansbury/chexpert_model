import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import DataLoader
import torch.nn.functional as F
from easydict import EasyDict as edict
import json
import pandas as pd

import sklearn.metrics as _metrics


# local data loader 
from data.dataset import ImageDataset 


class SingleChannelResnet(nn.Module):
    """ A class to convert a resent torchvision model to a grayscale
    input, binary classification output model """
    
    def __init__(self, pretrain=True, in_channels=1):
        """
        args:
            : pretrain (bool): use pretrained weights?
            : in_channels (int): number of channels, either 1 or 3
        """
        super(SingleChannelResnet, self).__init__()
        
        if pretrain:
            self.model = models.resnet18(pretrained=True)
#             for param in self.model.parameters():
#                 param.requires_grad = False
        else:
            self.model = models.resnet18(pretrained=False)
        
        # modify output
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        
        # modify the input 
        self.model.conv1 = nn.Conv2d(in_channels, 
                                64, 
                                kernel_size=7, 
                                stride=2, 
                                padding=3, 
                                bias=False)
        
    def forward(self, x):
        """for good measure, include the necessary method"""
        return self.model(x)
    
    
    


class TransferModel():
    """A class to transfer resnet to CheXpert images """
    
    def __init__(self, cfg_path='config.json', use_cpu=True):
        """ initialize the ExtendedResNet18 class 
        
        args:
            : cfg_path (str): path to the configuration file
            : use_cpu (bool): if true, use cpu()
        """
        if use_cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        # load configuration params used throughout
        self.config = self._load_config(cfg_path)
        
        # define the condition
        self.condition = self.config.condition
        
        # load dataloaders and label maps
        self.dataloader_train = self._load_train()
        self.dataloader_dev = self._load_dev()
        self.dataloader_valid = self._load_valid()

        self.train_headers = self.dataloader_train.dataset._label_header
        self.dev_headers = self.dataloader_dev.dataset._label_header
        self.valid_headers = self.dataloader_valid.dataset._label_header
        
        self.trainning_map = self._get_label_map(self.train_headers)
        self.dev_map = self._get_label_map(self.dev_headers)
        self.valid_map = self._get_label_map(self.valid_headers)
        
        # construct resnet
        self.model = self._construct_base_model()
        
        # construct ADAM
        self.optimizer = self._construct_optimizer()
        
        # define loss    
        
        if self.config.use_class_weight:
            weights = self._get_class_weights()
            self.criterion = nn.CrossEntropyLoss(weights)
            self.criterion.to(self.device)
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.criterion.to(self.device)
        
        # set structure for best model
        self.best_model = None
        self.dev_acc_history = []
        self.dev_loss_history = []
        self.dev_metric_history = []
        
        self.train_acc_history = []
        self.train_loss_history = []
        self.train_metric_history = []
        
        self.mean_metric = 'f1_score'
        
    
    def _get_class_weights(self):
        """A function to compute class wieghts from trainning data
        
        NOTE: we wish to punish mistakes on class 1 more 
        harshly, so we use the weight of the other class to balance
        
        returns:
            : wieghts (torch.tensor): (neg, pos)
        """
        cond_idx = self.trainning_map[self.condition]
        data_size = len(self.dataloader_train.dataset)
        
        labels = np.asarray(self.dataloader_train.dataset._labels)
        pos_class_count = labels[:, cond_idx].sum()
        pos_weight = pos_class_count / data_size
        neg_weight = 1 - pos_weight
        return torch.FloatTensor([pos_weight, neg_weight])
    
    
    def _load_config(self, cfg_path):
        """A function to load configuration info. 

        args:
            : cfg_path (str): path to the configuration file

        returns: 
            : cfg (dict): useful model parameters
        """
        with open(cfg_path) as f:
            cfg = edict(json.load(f))
        return cfg
    
    
    def _load_train(self):
        """A function to define the custom dataloader based on the config params
        
        returns:
            : dataloader_train (torch.utils.data.DataLoader): dataloader for trainning
            : dataloader_dev (torch.utils.data.DataLoader): dataloader for testing
        """
        dataloader_train = DataLoader(
            ImageDataset(self.config.train_csv, self.config, mode='train'),
            batch_size=self.config.train_batch_size, 
            num_workers=self.config.num_workers,
            drop_last=True, 
            shuffle=True
        )
        return dataloader_train
    
    
    def _load_dev(self):
        """A function to define the custom dataloader based on the config params
        
        returns:
            : dataloader_dev (torch.utils.data.DataLoader): dataloader for testing
        """    
        dataloader_dev = DataLoader(
            ImageDataset(self.config.dev_csv, self.config, mode='dev'),
            batch_size=self.config.dev_batch_size, 
            num_workers=self.config.num_workers,
            drop_last=False, 
            shuffle=False
        )
        return dataloader_dev
    
    
    def _load_valid(self):
        """A function to define the custom dataloader based on the config params
        
        returns:
            : dataloader_valid (torch.utils.data.DataLoader): dataloader for testing
        """    
        dataloader_valid = DataLoader(
            ImageDataset(self.config.valid_csv, self.config, mode='test'),
            batch_size=self.config.dev_batch_size, 
            num_workers=self.config.num_workers,
            drop_last=False, 
            shuffle=False
        )
        return dataloader_valid
  

    def _construct_base_model(self):
        """construct a a model architecture. Default to
        pretrained instance of resnet18 for binary classification.
        Note that the layers are not trainable for this model.
        
        returns:
            : model (torchvision.models.resnet.ResNet)
        """
        return SingleChannelResnet(pretrain=self.config.pretrained, 
                            in_channels=self.config.n_channels)
    
    
    def _get_label_map(self, label_headers):
        """A function to return a label-to-index mapping dictionary for
        coversion of a condition to a vector in the trainning label tensor
        
        args:
            : label_headers (list of str): from the dataloader header
        
        returns:
            : label_map (dict): index (key) to string label (value)
        """
        return {label : i for i, label in enumerate(label_headers)}
       
        
    def _construct_optimizer(self):
        """A function to construct the optimizer for the trainable layers
        
        returns:
            : opt (torch.optim.Optimizer): an optimizer for the model
        """
        
#         opt = optim.Adam(self.model.parameters(), 
#                          lr=self.config.learning_rate, 
#                          betas=(self.config.beta1, self.config.beta2))
        
        opt = optim.SGD(self.model.parameters(), 
                        lr=self.config.learning_rate, 
                        momentum=self.config.momentum)
                         
        return opt
    
    
    def _get_loss(self, output, labels, label_map):
        """A function to compute loss on the class of interest 
        
        args:
            : output (torch.Tensor): model output
            : labels (torch.Tensor): matrix of labels
            : label_map (dict): the label map to use
        
        returns:
            : loss (float): the loss of the batch
        """
        cond_idx = label_map[self.condition]
        target = labels[:, cond_idx].type(torch.LongTensor)
        
        loss = self.criterion(output, target)
        _, y_pred = torch.max(output, 1)
        n_correct = torch.sum(y_pred == target)
        return loss
     
        
    def _get_batch_metrics(self, output, labels, label_map):
        """A function to compute batch metrics 
        
        args:
            : output (torch.Tensor): model output
            : labels (torch.Tensor): matrix of labels
            : label_map (dict): the label map to use
            
        returns:
            : metrics (dict): dictionary of metrics for the batch
        """
        cond_idx = label_map[self.condition]
        y_true = labels[:, cond_idx].detach().numpy().astype(int)
        _, y_pred = torch.max(output, 1)
        y_pred = y_pred.detach().numpy().astype(int)
        
        n_correct = np.sum(y_pred == y_true)
        
        metrics = {
            'n_correct' : n_correct,
            'f1_score' : _metrics.f1_score(y_true, y_pred, zero_division=0),
            'precision_score' : _metrics.precision_score(y_true, y_pred, zero_division=0),
            'recall_score' : _metrics.recall_score(y_true, y_pred),
            
        }
        
        return metrics
        
        
    
    def _train_epoch(self):
        """A function to wrap trainning procedure
        
        returns:
            : epoch_loss (float): the loss of the epoch
            : epoch_acc (float): the accuracy of the epoch
        """
        self.model.train()  
            
        running_loss = 0.0
        running_corrects = 0
        mean_metric = 0
         
        with torch.set_grad_enabled(True):
            for b_id, (inputs, labels) in enumerate(self.dataloader_train):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                output = self.model(inputs)       
                loss = self._get_loss(output, labels, self.trainning_map)
                
                batch_scores = self._get_batch_metrics(output, labels, self.trainning_map)
                batch_scores['batch'] = batch_scores
                self.train_metric_history.append(batch_scores)
                
                mean_metric += batch_scores[self.mean_metric]

                # backward + optimize only if in training phase
                loss.backward()
                self.optimizer.step()

                # batch statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += batch_scores['n_correct']
            
        # epoch statistics
        data_size = len(self.dataloader_train.dataset)
        epoch_metric = mean_metric / data_size
        epoch_loss = running_loss / data_size
        epoch_acc = running_corrects / data_size
            
        return epoch_loss, epoch_acc, epoch_metric
    
    
    def _eval_epoch(self):
        """A function to wrap trainning procedure
        
        returns:
            : epoch_loss (float): the loss of the epoch
            : epoch_acc (float): the accuracy of the epoch
        """
        self.model.eval()   
            
        running_loss = 0.0
        running_corrects = 0
        mean_metric = 0
        
        with torch.no_grad():
            for b_id, (inputs, labels) in enumerate(self.dataloader_dev):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                output = self.model(inputs)       
                loss = self._get_loss(output, labels, self.dev_map)
                
                batch_scores = self._get_batch_metrics(output, labels, self.dev_map)
                batch_scores['batch'] = batch_scores
                
                mean_metric += batch_scores[self.mean_metric]
                
                self.dev_metric_history.append(batch_scores)

                # batch statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += batch_scores['n_correct']
            
        # epoch statistics
        data_size = len(self.dataloader_dev.dataset)
        epoch_loss = running_loss / data_size
        epoch_acc = running_corrects / data_size
        epoch_metric = mean_metric / data_size
            
        return epoch_loss, epoch_acc, epoch_metric
        
        
    def train(self):
        """Function to train a single epoch on a binary classification 
        task as defined by the 'condition' in the config file.
        
        args:
            : condition (str): one of the conditions below
                 0: 'Cardiomegaly',
                 1: 'Edema',
                 2: 'Consolidation',
                 3: 'Atelectasis',s
                 4: 'Pleural Effusion'
        """
        
        self.validation_history = [] #reset always
        self.loss_history = []
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        
        for epoch in range(self.config.num_epochs):
            print()
            print('-------------------------------')
            print(f"{self.condition} Model epoch {epoch + 1}/{self.config.num_epochs}")
            
            """
            TRAIN THE MODEL
            """    
            train_loss, train_acc, train_metric = self._train_epoch()
            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)
            
            print(f"Trainning loss: {train_loss:.4f} accuracy: {100*train_acc:.2f} %")
            
            """
            EVALUATE THE MODEL
            """
            dev_loss, dev_acc, dev_metric = self._eval_epoch()
            self.dev_loss_history.append(dev_loss)
            self.dev_acc_history.append(dev_acc)
            
            print(f"Validation loss: {dev_loss:.4f} accuracy: {100*dev_acc:.2f} %")   
    
            # deep copy the model
            if dev_acc > best_acc:
                best_acc = dev_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
            

        print()
        print('Best dev Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.best_model = self.model.load_state_dict(best_model_wts)
    
    
    def evaluate_model(self, model, loader, label_map):
        """A function to evaluate the final model performance on the development
        data
        
        TODO: check against: https://github.com/jfhealthcare/Chexpert/blob/4efbb4b251e7908cf855e4494ea6b9d2b8f4fbaa/bin/train.py#L231
        
        args:
            : model (torch.model): a trainned model for eval
            : loader (torch.utils.data.DataLoader): the data to evaluate
            : label_map (dict): the label map for the file
        
        returns:
            : results (pd.DataFrame): results reported on the development
            data
        """
        # get in the index of the condition
        cond_idx = label_map[self.condition]
        new_rows = []
        
        with torch.no_grad():

            for i, (inputs, labels) in enumerate(loader):
                output = self.model(inputs)
                
                
                _, y_pred = torch.max(output, 1)
                y_prob = F.softmax(output, dim=1)
                top_p, _ = y_prob.topk(1, dim=1)

                for j, _ in enumerate(inputs):
                    row = {
                        'y_prob': 1 - top_p[j].detach().numpy()[0],
                        'y_pred': y_pred[j].detach().numpy(),
                        'y_true': labels[j, cond_idx].detach().numpy()
                    }

                    new_rows.append(row)

        results = pd.DataFrame(new_rows)
        return results