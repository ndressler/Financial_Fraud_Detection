import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import GridSearchCV
from scipy.stats import norm
from sklearn.metrics import make_scorer


def load_data():
    '''Load the data'''
    # get the current working directory
    cwd = os.getcwd()

    # relative paths
    rel_path_train_df = '../../data/processed/cleaned_train.csv'
    rel_path_test_df = '../../data/processed/cleaned_test.csv'

    # absolute paths
    abs_path_tran_df = os.path.join(cwd, rel_path_train_df)
    abs_path_test_df = os.path.join(cwd, rel_path_test_df)

    # read files
    X = pd.read_csv(abs_path_tran_df)
    x_test = pd.read_csv(abs_path_test_df)

    return X, x_test

def evaluate_model(model, X):
    ''' Evaluation Metrics'''

    # predict the anomalies
    pred_test = model.predict(X)

    # silhouette score
    sil_score = silhouette_score(X, pred_test)

    # Davies-Bouldin Score
    db_score = davies_bouldin_score(X, pred_test)

    # Calinski-Harabasz Score
    ch_score = calinski_harabasz_score(X, pred_test)

    # anomaly score distribution
    anomaly_scores = model.decision_function(X)

    # fit a normal distribution to the anomaly scores
    mu, std = norm.fit(anomaly_scores)

    # compute the distance of the mean of the anomaly scores from 1
    anomaly = abs(mu - 1)

    # compute the average of the silhouette score and the anomaly score distribution
    score = (sil_score + (1 - anomaly)) / 2

    return sil_score, db_score, ch_score, anomaly_scores, score

def evaluate_lof_model(model, X):
    ''' Evaluation Metrics for LocalOutlierFactor'''

    # Use fit_predict for LocalOutlierFactor
    pred_test = model.fit_predict(X)

    # silhouette score
    sil_score = silhouette_score(X, pred_test)

    # Davies-Bouldin Score
    db_score = davies_bouldin_score(X, pred_test)

    # Calinski-Harabasz Score
    ch_score = calinski_harabasz_score(X, pred_test)

    # anomaly score distribution
    anomaly_scores = model.negative_outlier_factor_

    # fit a normal distribution to the anomaly scores
    mu, std = norm.fit(anomaly_scores)

    # compute the distance of the mean of the anomaly scores from 1
    anomaly = abs(mu - 1)

    # compute the average of the silhouette score and the anomaly score distribution
    score = (sil_score + (1 - anomaly)) / 2

    return sil_score, db_score, ch_score, anomaly_scores, score

def hyper_tuning(algorithm, param_grid, X):
    '''Hyperparameter Tuning'''

    # Initialize the model
    model = algorithm()

    # Make a scorer for the custom scoring function
    scorer = make_scorer(make_custom_scorer(model), greater_is_better=True, needs_proba=False, needs_threshold=False)

    # Initialize the grid search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scorer, n_jobs=-1)

    # Fit the grid search
    grid_search.fit(X)

    # Get the best parameters
    best_params = grid_search.best_params_

    return best_params

def make_custom_scorer(model):
    def custom_scorer(X, y_pred):
        # Compute the silhouette score
        silhouette = silhouette_score(X, y_pred)
        # Compute the anomaly score distribution
        anomaly_scores = model.decision_function(X)
        # Fit a normal distribution to the anomaly scores
        mu, std = norm.fit(anomaly_scores)
        # Compute the distance of the mean of the anomaly scores from 1
        anomaly = abs(mu - 1)
        # Return the average of the silhouette score and the anomaly score distribution
        return (silhouette + (1 - anomaly)) / 2
    return custom_scorer

def show_metrics(sil_score, db_score, ch_score, score, model_name):
    '''Displaying the evaluation metrics'''

    # get silhouette score
    print(f'Silhouette Score of {model_name}: {sil_score}')

    # get Davies-Bouldin Score
    print(f'Davies-Bouldin Score of {model_name}: {db_score}')

    # get Calinski-Harabasz Score
    print(f'Calinski-Harabasz Score of {model_name}: {ch_score}')

    # get the score (average of the silhouette and anomaly score distribution)
    print(f'Score of {model_name}: {score}')

def plot_hist(algorithm, anomaly_scores, model_name):
    '''Plotting the histogram of the anomaly scores'''
    # style
    sns.set(style="whitegrid")

    # figure
    plt.figure(figsize=(10, 6))

    # plot histogram
    sns.histplot(anomaly_scores, bins=50, kde=True, color='blue')

    # add title and labels
    plt.title(f'Distribution of Anomaly Scores from {model_name}', fontsize=15)
    plt.xlabel('Anomaly Score', fontsize=12)
    plt.ylabel('Number of Instances', fontsize=12)

    # save figure as png
    plt.savefig(f'../visualization/{algorithm}_anomaly_scores_{model_name}.png')

    # show plot
    plt.show()
