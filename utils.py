import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def save_features_and_labels(output_file:str, image_features_array:np.array, labels_array):
    with open(output_file, 'wb') as f:
        pickle.dump((image_features_array, labels_array), f)
    
    print(f"Saved image features and labels to {output_file}")


def load_features_and_labels(input_file:str,reshape_features:bool=True) -> np.array:
    with open(input_file, 'rb') as f:
        image_features_array, labels_array = pickle.load(f)
        if reshape_features:
            image_features_array = image_features_array.reshape(image_features_array.shape[0], -1)
    
    print(f"Loaded image features and labels from {input_file}")
    return image_features_array, labels_array


def get_percentage(
    features:np.array,
    labels:np.array,
    percentage:float,
    verbose:bool=True) ->np.array:
    """
    Returns a specified percentage of the data.
    
    Parameters:
    data (list or sequence): The input dataset.
    percentage (int or float): The percentage of the dataset to return (0-100).
    
    Returns:
    list: A subset of the data representing the specified percentage.
    """
    if not 0 <= percentage <= 1:
        raise ValueError("Percentage must be between 0 and 1")
    
    if percentage == 1:
        X_train, y_train = features,labels 

    else:
        _, X_train,_,y_train = train_test_split(
            features, 
            labels, 
            test_size=percentage, 
            random_state=42)

    if verbose: 
        print(f"Only using {X_train.shape}/{features.shape} or { (X_train.shape[0]/features.shape[0])*100} % of data" )
  
    return X_train,y_train


def compute_all_metrics(outputs, targets):
    n_classes = outputs.shape[1]
    fpr, tpr, aucs, precision, recall = {}, {}, {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _                       = roc_curve(targets[:,i], outputs[:,i])
        aucs[i] = auc(fpr[i], tpr[i])
        precision[i], recall[i], _              = precision_recall_curve(targets[:,i], outputs[:,i])
        fpr[i], tpr[i], precision[i], recall[i] = fpr[i].tolist(), tpr[i].tolist(), precision[i].tolist(), recall[i].tolist()

    metrics = {'fpr': fpr,
               'tpr': tpr,
               'aucs': aucs,
               'precision': precision,
               'recall': recall,
              }

    return metrics