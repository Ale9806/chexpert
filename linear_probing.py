import argparse
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput  import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from utils    import load_features_and_labels,get_percentage,compute_all_metrics

from xgboost import XGBClassifier

def main():
    parser = argparse.ArgumentParser(description='Evaluate a dataset using a given model')
    parser.add_argument('--model',      type=str, default="biomedclip", help='Name of model')
    parser.add_argument('--output_dir', type=str, default='outputs',    help='Output directory to save evaluation results (default is "outputs")') 
    parser.add_argument("-p",'--percentage', type=float, default=0.1,   help='Percentage to train Probe')
    parser.add_argument("-m",'--max_iter', type=int, default=1000,      help='Percentage to train Probe')
    parser.add_argument('--scale',      action='store_true',            help='Flag to scale data')
    parser.add_argument('--muticlass',  action='store_true',            help='Flag to change method to multilabel to multiclass (OneVsALl)')
    parser.add_argument("-r",'--probe', type=str, default='xgboost',    help='Probe')

    args = parser.parse_args()
    
    ## Load Data ##
    train_filepath  = Path( args.output_dir ,f"{args.model}_embeddings_train.pkl")
    test_filepath   = Path( args.output_dir ,f"{args.model}_embeddings_valid.pkl")
    features,labels = load_features_and_labels(train_filepath)
    features,labels = features[0:1000],labels[0:1000]
    X_test,y_test   = load_features_and_labels(test_filepath)
    X_train,y_train = get_percentage(
        features    = features,
        labels      = labels,
        percentage  = args.percentage)

    scaler = preprocessing.StandardScaler().fit(X_train)
    if args.scale:
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    ## Initialize and train model ##
    #model = MultiOutputClassifier(LogisticRegression(max_iter=args.max_iter,))
    if args.probe == "xgboost":
        if args.muticlass:
            classifier = XGBClassifier(n_jobs=-1)
            model      = OneVsRestClassifier(classifier)
        else:
            model = XGBClassifier(tree_method="hist", multi_strategy="multi_output_tree")

         param_grid = {
            'estimator__max_depth': [3, 6, -1],
            'estimator__learning_rate': [0.02, 0.1, 0.5],
            'estimator__num_leaves' : [10, 25, 100],}

    else args.probe == "linear":
        param_grid = {
            "estimator__C": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2],
            "estimator__penalty": ['l2']
        }

   

    # GridSearchCV
    clf = GridSearchCV(
        model, 
        param_grid, 
        scoring='roc_auc',
        cv=3, 
        n_jobs=-1,
        verbose=0, 
        refit=False)
    clf.fit(X_train, y_train)
  
    # Best model
    best_params = clf.best_params_
    #best_base_estimator = LogisticRegression(
    #    C=best_params['estimator__C'], 
    #    penalty=best_params['estimator__penalty'])
    #best_model = MultiOutputClassifier(best_base_estimator)
    best_base_estimator = LogisticRegression(
        max_depth     = best_params['estimator__max_depth'],
        learning_rate = best_params['estimator__learning_rate'],
        num_leaves    =best_params['estimator__num_leaves'])
    best_model = OneVsRestClassifier(XGBClassifier(n_jobs=-1))
    best_model.fit(X_train, y_train)  
    y_pred = best_model.predict(X_test)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    metrics = compute_all_metrics(y_pred, y_test)
    import pdb; pdb.set_trace()

# Debugging line (optional)
if __name__ == "__main__":
    main()
    