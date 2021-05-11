import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_recall_fscore_support



def get_classification_metrics(model, X_test, y_test):
    """A function to 'pprint' classification metrics (binary)"""
    y_proba = model.predict(X_test)[:,1]
    y_test = y_test[:, 1]

    # compute tpr/fpr at every thresh
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    # get optimal threshold by AUCROC
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
        'optimal_threshold':optimal_threshold,
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