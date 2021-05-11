import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split


def make_trainning_data(root_dir='/nfs/turbo/umms-indikar/shared/projects/CheXpert-v1.0-small/', 
                        output_dir="./", sample=None, return_frames=False, state=1729):
    """A function to create local copies of the data for trainning 
    
    args:
        : root_dir (str): the turbo volume location of the raw data
        : output_dir (str): the location to write new data to
        : sample (int or None): if `None' read all data. If sample=int
        return a subsample of the full dataset
        : return_frames (bool): if true returns the dataframes
        : state (int or None): a random seed 
        
    
    returns:
        : train (pd.DataFrame): trainning data
        : dev (pd.DataFrame): trainning data
    """
    
    df = pd.read_csv(f"{root_dir}train.csv")
    valid_df = pd.read_csv(f"{root_dir}valid.csv")
    
    # smaller samples possible
    if not sample is None:
        print(f"sampling {sample} records")
        sample = int(sample)
        df = df.sample(sample)
        
    # train test split with random seed
    train, dev = train_test_split(df, test_size=0.25, random_state=state)
    
    print(f"train.shape: {train.shape}")
    print(f"dev.shape: {dev.shape}")
    print(f"valid.shape: {valid_df.shape}")
    
    train_path = f"{output_dir}train.csv"
    dev_path = f"{output_dir}dev.csv"
    valid_path = f"{output_dir}valid.csv"
    
    train.to_csv(train_path, index=False)
    print(f"saved: {train_path}")
    dev.to_csv(dev_path, index=False)
    print(f"saved: {dev_path}")
    valid_df.to_csv(valid_path, index=False)
    print(f"saved: {valid_path}")
    
    if return_frames:
        return train, dev, valid_df
    


def imshow(inp, title=None):
    """Imshow for Tensor.
    
    from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    """
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    
    
    
def plot_sample(dataloader):
    """A wrapper around imshow.
    
    args:
        : dataloader (pytorch.dataloader)
    """
    inputs, classes = next(iter(dataloader))
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[x for x in classes])
    
    
def get_classification_metrics(results):
        """A function to 'pprint' classification metrics (binary)
        
        args:
            : results (pd.DataFrame): with columns `y_true`, `y_pred' and
            `prob_0`
            
        returns:
            : metrics (pd.DataFrame): performance metrics of the classifier
        """
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import roc_curve
        from sklearn.metrics import roc_auc_score

        y_proba = results['y_prob'].astype(float)
        y_test = results['y_true'].astype(int)

        # compute tpr/fpr at every thresh
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)

        # get optimal threshold by AUCROC - by Youden's J
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        aucroc = roc_auc_score(y_test, y_proba)

        # compute predictions based on optimal threshold
        y_pred = np.where(y_proba >= optimal_threshold, 1, 0)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # get precision/recall
        rate = y_test.mean()
        precision = tp / (tp + fn * (1 / rate - 1))
        recall = tp / (tp + fn * (1 / rate - 1))
        f1 = 2 * tp / (2*tp + fp + fn)

        res_dict = {
            'optimal_threshold': optimal_threshold,
            'true negatives': tn,
            'true positives': tp,
            'false positives': fp,
            'false negatives': fn,
            'sensitivity': tp / (tp + fn),
            'specificity': tn / (tn + fp),
            'F1-score' : f1,
            'precision': precision,
            'recall': recall,
            'AUCROC' : aucroc,
        }

        res = pd.DataFrame.from_dict(res_dict, orient='index')
        return res