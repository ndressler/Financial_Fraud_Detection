import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score


def load_data():
    '''Load the data'''

    # get the current working directory
    cwd = os.getcwd()

    # relative paths
    rel_path_test = '../../data/processed/test.pkl'
    rel_path_X_train = '../../data/processed/X_train.pkl'
    rel_path_X_val = '../../data/processed/X_val.pkl'
    rel_path_y_train = '../../data/processed/y_train.pkl'
    rel_path_y_val = '../../data/processed/y_val.pkl'

    # absolute paths
    abs_path_test = os.path.join(cwd, rel_path_test)
    abs_path_X_train = os.path.join(cwd, rel_path_X_train)
    abs_path_X_val = os.path.join(cwd, rel_path_X_val)
    abs_path_y_train = os.path.join(cwd, rel_path_y_train)
    abs_path_y_val = os.path.join(cwd, rel_path_y_val)

    # read files
    test = pd.read_pickle(abs_path_test)
    X_train = pd.read_pickle(abs_path_X_train)
    X_val = pd.read_pickle(abs_path_X_val)
    y_train = pd.read_pickle(abs_path_y_train)
    y_val = pd.read_pickle(abs_path_y_val)

    return test, X_train, X_val, y_train, y_val

def if_hypertune(param_grid, X_train, y_train):
    '''Hyperparameter tuning for an Isolation Forest model using GridSearchCV'''

    # Initialize the model
    clf = IsolationForest(random_state=42)

    # Initialize the grid search
    grid_search = GridSearchCV(clf, param_grid, scoring='roc_auc', cv=5, n_jobs=-1)

    # Fit the grid search on your training data
    grid_search.fit(X_train, y_train)

    # Print the best parameters and the best score
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best ROC AUC score: {grid_search.best_score_}")

    # Get the best parameters and the best estimator
    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_

    return best_params, best_estimator

def get_cm(model_predictions, y_val):
    y_val_converted = [-1 if i else 1 for i in y_val]
    tn, fp, fn, tp = confusion_matrix(y_val_converted, model_predictions).ravel()
    return tn, fp, fn, tp

def cm_inf(cm, model_name):
    '''Confusion Matrix for the model'''
    tn, fp, fn, tp = cm
    print(f"Confusion Matrix for model {model_name}:\n")
    print(f"True Positives: {tp}")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")

def plot_cm(cm, model_name):
    '''Plot the confusion matrix'''

    tn, fp, fn, tp = cm

    # Reconstruct the confusion matrix
    conf_mat = np.array([[tn, fp], [fn, tp]])

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10,7))

    # Plot the confusion matrix on the axes
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Anomaly', 'Normal'], yticklabels=['Anomaly', 'Normal'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    # Set the title
    ax.set_title(f'Confusion Matrix of {model_name}')

def get_metrics(y_val_converted, pred):
    '''Get the evaluation metrics'''

    # Calculate Precision, Recall, F1 Score, and AUC-ROC
    precision = precision_score(y_val_converted, pred)
    recall = recall_score(y_val_converted, pred)
    f1 = f1_score(y_val_converted, pred)
    roc_auc = roc_auc_score(y_val_converted, pred)

    return precision, recall, f1, roc_auc

def print_metrics(metrics, model_name):
    '''Get the evaluation metrics'''

    precision, recall, f1, roc_auc = metrics

    print(f"Metrics of {model_name}:\n")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"AUC-ROC: {roc_auc}")

def plot_metrics(metrics, model_name):
    '''Plot the evaluation metrics'''

    precision, recall, f1, roc_auc = metrics

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10,7))

    # Plot the metrics on the axes
    sns.barplot(x=['Precision', 'Recall', 'F1 Score', 'AUC-ROC'], y=[precision, recall, f1, roc_auc], ax=ax)

    # Set the title
    ax.set_title(f'Metrics of {model_name}')
