import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

from tensorboardX import SummaryWriter
import numpy as np
from scipy import stats

from dataset import ChexpertSmall, ChexpertSmallInspect
from utils import load_features_and_labels, get_percentage, compute_all_metrics, set_random_seed
from chexpert import evaluate_single_model,  train_epoch,save_json,plot_roc,plot_roc_multiple_seeds


# Define a simple dataset class for PyTorch
class SimpleDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Define a simple neural network with a single linear layer
class SimpleNN(nn.Module):
    def __init__(
            self,
            input_size:int, 
            num_classes:int,
            dropout_prob:float=0.5,
            double_layer:bool = False):
        
        super(SimpleNN, self).__init__()
        self.double_layer = double_layer
        if self.double_layer:
            hidden_size = 2*input_size
        else:
            hidden_size = num_classes

        self.initial_dropout = nn.Dropout(dropout_prob)
        self.fc1             = nn.Linear(input_size, hidden_size)

        if self.double_layer:
            self.relu    = nn.ReLU()
            self.dropout = nn.Dropout(dropout_prob)
            self.fc2     = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.initial_dropout(x)
        x = self.fc1(x)
        
        if self.double_layer:
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
        return x

def run_experiment(
    args:dict,
    train_loader,
    test_loader,
    learning_rates:list[float]=[0.0001, 0.001, 0.01, 0.1]):

    best_metrics = None
    best_lr      = None
    best_epoch   = None
    best_model_state = None
    
    for learning_rate in learning_rates:
        model        = SimpleNN(args.input_size, args.num_classes)
        model.to(args.device)
        
        # Define loss function and optimizer
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = None
        writer    = SummaryWriter(logdir=args.output_dir)  # creates output_dir
    
        # Training loop
        model.train()
        for epoch in range(args.n_epochs):
            train_epoch(model, train_loader, test_loader, criterion, optimizer, scheduler, writer, epoch, args)
            # Evaluate model
            metrics = evaluate_single_model(model, test_loader, criterion, args.device)
            aurcs   = np.array(list(metrics["aucs"].values())).mean()
    

            # Choose a metric to determine the best model (e.g., accuracy, F1 score, ROC AUC)
            if best_metrics is None or aurcs > best_metrics:
                best_metrics = aurcs
                best_lr      = learning_rate
                best_epoch   = epoch 
                best_model_state = model

    metrics = evaluate_single_model(best_model_state, test_loader, criterion,args.device)

    return metrics,best_epoch,best_lr    

def main():
    parser = argparse.ArgumentParser(description='Evaluate a dataset using a PyTorch neural network')
    parser.add_argument('--model', type=str, default="biomedclip", help='Name of model')
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--output_dir', type=str, default='outputs',        help='Output directory to save evaluation results (default is "outputs")') 
    parser.add_argument("-p", '--percentage', type=float, default=0.1, help='Percentage to train Probe')
    parser.add_argument('--scale', action='store_true', help='Flag to scale data')
    parser.add_argument('--probe', type=str, default='pytorch', help='Probe')
    parser.add_argument('--pretrained', action='store_true', help='Use ImageNet pretrained model and normalize data mean and std.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Dataloaders batch size.')
    parser.add_argument('--n_epochs', type=int, default=60, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--lr_warmup_steps', type=float, default=0, help='Linear warmup of the learning rate for lr_warmup_steps number of steps.')
    parser.add_argument('--lr_decay_factor', type=float, default=0.97, help='Decay factor if exponential learning rate decay scheduler.')
    parser.add_argument('--step', type=int, default=0, help='Current step of training (number of minibatches processed).')
    parser.add_argument('--log_interval', type=int, default=40, help='Interval of num batches to show loss statistics.')
    parser.add_argument('--eval_interval', type=int, default=300, help='Interval of num epochs to evaluate, checkpoint, and save samples.')
    parser.add_argument('--label_set',  type=str, default='inspect',                   help='labels set to use ')
    parser.add_argument('--rseeds',  type=list,  default = [42, 123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627,
           282930, 313233, 343536, 373839, 404142, 434445, 464748, 495051, 525354, 555657,
           585960, 616263, 646566, 676869, 707172, 737475, 767778, 798081, 828384, 858687],  help='labels set to use ')
  

    args = parser.parse_args()
    

    if args.label_set == "inspect": 
        percentages = [0.01+0.26,0.1+0.26,1]
        percentages = [1]
        args.mask = True
        
    else:
        percentages = [0.01,0.1,1]
        args.mask = False
        percentages = [1]

    for percentage in percentages:
        # Load Data

        args.percentage = percentage 
        if args.label_set == "inspect":
            args.DATASET = ChexpertSmallInspect

        elif args.label_set == "chexpert": 
           args.DATASET = ChexpertSmall 
        else:
            raise f"Error: No support for  {args.label_set } "

        train_filepath = Path(args.output_dir,"embeddings",f"{args.label_set}_{args.model}_embeddings_train.pkl")
        test_filepath  = Path(args.output_dir,"embeddings",f"{args.label_set}_{args.model}_embeddings_test.pkl")
        valid_filepath = Path(args.output_dir,"embeddings",f"{args.label_set}_{args.model}_embeddings_valid.pkl")
     
        features, labels = load_features_and_labels(train_filepath)
        X_test, y_test   = load_features_and_labels(test_filepath)
        X_valid, y_valid = load_features_and_labels(test_filepath)
        X_train, y_train = get_percentage(features=features, labels=labels, percentage=args.percentage)
        #import pdb;pdb.set_trace()
        (args.input_size, args.num_classes)  = X_train.shape[1],y_train.shape[-1]  # Define size

        # Scale Data if required
        if args.scale:
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            X_valid = scaler.transform(X_valid)

        # Convert data to PyTorch DataLoader
        train_dataset = SimpleDataset(X_train, y_train)
        if args.label_set == "inspect":
            add_200:bool =True
            X_combined = np.concatenate((X_valid, X_test), axis=0)
            y_combined = np.concatenate((y_valid, y_test), axis=0)
            
            if add_200:
                X_combined = np.concatenate((X_combined, X_train[0:200,:]), axis=0)
                y_combined = np.concatenate((y_combined,y_train[0:200,:]), axis=0)
                X_valid    = X_train[200:400,:] 
                y_valid    = y_train[200:400,:]
                X_train = X_train[400::,:]
                y_train = y_train[400::,:] 
          
            test_dataset   = SimpleDataset(X_combined, y_combined)
            valid_dataset  = SimpleDataset(X_valid, y_valid) 
        else:
            test_dataset   = SimpleDataset(X_test, y_test)
            valid_dataset  = SimpleDataset(X_valid, y_valid)  
    
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    
        metrics = {}
        params  = {}
        bootstraps = {}
        for seed in args.rseeds:
            set_random_seed(seed)
            learning_rates= [0.0001, 0.001, 0.01, 0.1]
            metrics[seed] , params["best_epoch"], params["best_lr"]   = run_experiment(
                args,
                train_loader,
                valid_loader,
                learning_rates)
            metrics[seed]["params"] = params 
        
        aucs_ = collect_metric(metrics,'aucs')
        loss_ =  aucs = collect_metric(metrics,'loss')  
        
        bootstraps["aucs"] = bootstrap_statistics(aucs_)
        bootstraps["loss"]= bootstrap_statistics(loss_)

        output_name = f"{args.label_set}_{args.model}_train_percent_{args.percentage}"
        save_json(bootstraps, "results/"+output_name+"boostraps", args)
        save_json(metrics, "results/"+output_name, args)
        plot_roc_multiple_seeds(metrics, bootstraps["aucs"], args, filename=output_name, labels=args.DATASET.attr_names)
      
    
def bootstrap_statistics(data, num_bootstrap_samples=1000, confidence_level=0.95):
    boostrap_dict={}
    for key,values in data.items():
        data = np.array([v for v in values if not np.isnan(v)])  # Remove NaN values for the calculations
        if len(data) == 0:
            boostrap_dict[key] = None
        else:
            data = (data,)
            dump_ = {}
            boostrap_ci   = stats.bootstrap(data, np.mean, confidence_level=confidence_level,random_state=1, method='percentile')
            dump_["low"]  = boostrap_ci.confidence_interval.low
            dump_["high"] = boostrap_ci.confidence_interval.high
            dump_["se"]   = boostrap_ci.standard_error
            dump_["mean"] = boostrap_ci.bootstrap_distribution.mean()
            boostrap_dict[key] = dump_ 
         
    return boostrap_dict


def collect_metric(metrics,name) -> dict[int,list[int]]:
    collected_metric = {i: [] for i in metrics[42][name].keys()}  
    for seed, seed_metrics in metrics.items():
        for key in seed_metrics[name].keys():
            #ximport pdb;pdb.set_trace()
            collected_metric[key].append(seed_metrics[name][key])
    return collected_metric

if __name__ == "__main__":
    main()
