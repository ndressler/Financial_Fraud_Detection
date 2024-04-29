import os
import pandas as pd
import numpy as np
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score
from sklearn.model_selection import cross_val_predict
from lime import lime_tabular
from scipy.special import expit

class Data:
    ''' Class to load and save data'''

    def __init__(self):
        pass

    def load_raw_data(self):
        '''Load raw data'''

        # directory for  files
        data_dir = os.path.join(os.getcwd(), '../data/raw/')

        # relative paths
        rel_path_train_tran = 'train_transaction.csv'
        rel_path_train_id = 'train_identity.csv'
        rel_path_test_tran = 'test_transaction.csv'
        rel_path_test_id = 'test_identity.csv'

        # absolute paths
        abs_path_train_tran = os.path.join(data_dir, rel_path_train_tran)
        abs_path_train_id = os.path.join(data_dir, rel_path_train_id)
        abs_path_test_tran = os.path.join(data_dir, rel_path_test_tran)
        abs_path_test_id = os.path.join(data_dir, rel_path_test_id)

        # read files
        train_tran = pd.read_csv(abs_path_train_tran)
        train_id = pd.read_csv(abs_path_train_id)
        test_tran = pd.read_csv(abs_path_test_tran)
        test_id = pd.read_csv(abs_path_test_id)

        return train_tran, train_id, test_tran, test_id

    def load_data(self, data_dir):
        '''Load processed data'''

        # relative paths
        rel_path_X_train = 'X_train.pkl'
        rel_path_X_val = 'X_val.pkl'
        rel_path_X_test = 'X_test.pkl'
        rel_path_y_train = 'y_train.pkl'
        rel_path_y_val = 'y_val.pkl'
        rel_path_y_test = 'y_test.pkl'
        rel_path_unlabeled_test = 'unlabeled_test.pkl'

        # absolute paths
        abs_path_X_train = os.path.join(data_dir, rel_path_X_train)
        abs_path_X_val = os.path.join(data_dir, rel_path_X_val)
        abs_path_X_test = os.path.join(data_dir, rel_path_X_test)
        abs_path_y_train = os.path.join(data_dir, rel_path_y_train)
        abs_path_y_val = os.path.join(data_dir, rel_path_y_val)
        abs_path_y_test = os.path.join(data_dir, rel_path_y_test)
        abs_path_unlabeled_test = os.path.join(data_dir, rel_path_unlabeled_test)

        # read files
        X_train = pd.read_pickle(abs_path_X_train)
        X_val = pd.read_pickle(abs_path_X_val)
        X_test = pd.read_pickle(abs_path_X_test)
        y_train = pd.read_pickle(abs_path_y_train)
        y_val = pd.read_pickle(abs_path_y_val)
        y_test = pd.read_pickle(abs_path_y_test)
        unlabeled_test = pd.read_pickle(abs_path_unlabeled_test)

        return X_train, X_val, X_test, y_train, y_val, y_test, unlabeled_test

    def save_processed(self, X_train, X_val, X_test, y_train, y_val, y_test, unlabeled_test):
        '''Save processed data'''

        # directory for output files
        data_dir = os.path.join(os.getcwd(), '../data/processed')

        # relative paths
        rel_path_X_train = 'X_train.pkl'
        rel_path_X_val = 'X_val.pkl'
        rel_path_X_test = 'X_test.pkl'
        rel_path_y_train = 'y_train.pkl'
        rel_path_y_val = 'y_val.pkl'
        rel_path_y_test = 'y_test.pkl'
        rel_path_unlabeled_test = 'unlabeled_test.pkl'

        # absolute paths
        abs_path_X_train = os.path.join(data_dir, rel_path_X_train)
        abs_path_X_val = os.path.join(data_dir, rel_path_X_val)
        abs_path_X_test = os.path.join(data_dir, rel_path_X_test)
        abs_path_y_train = os.path.join(data_dir, rel_path_y_train)
        abs_path_y_val = os.path.join(data_dir, rel_path_y_val)
        abs_path_y_test = os.path.join(data_dir, rel_path_y_test)
        abs_path_unlabeled_test = os.path.join(data_dir, rel_path_unlabeled_test)

        # save the DataFrames to pickle files
        X_train.to_pickle(abs_path_X_train)
        X_val.to_pickle(abs_path_X_val)
        X_test.to_pickle(abs_path_X_test)
        y_train.to_pickle(abs_path_y_train)
        y_val.to_pickle(abs_path_y_val)
        y_test.to_pickle(abs_path_y_test)
        unlabeled_test.to_pickle(abs_path_unlabeled_test)

    def save_ind_model(self, model, model_name, model_dir):
        '''Save individual model'''

        # relative path
        rel_path_model = f'{model_name}.pkl'

        # absolute path
        abs_path_model = os.path.join(model_dir, rel_path_model)

        # save the model
        joblib.dump(model, abs_path_model)

    def load_ind_models(self, models_dir):
        '''Load the individual models'''

        # relative paths
        rel_path_if = 'if_model.pkl'
        rel_path_lof = 'lof_model.pkl'
        rel_path_ocsvm = 'ocsvm_model.pkl'

        # absolute paths
        abs_path_if = os.path.join(models_dir, rel_path_if)
        abs_path_lof = os.path.join(models_dir, rel_path_lof)
        abs_path_ocsvm = os.path.join(models_dir, rel_path_ocsvm)

        # load the models
        if_model = joblib.load(abs_path_if)
        lof_model = joblib.load(abs_path_lof)
        ocsvm_model = joblib.load(abs_path_ocsvm)

        return if_model, lof_model, ocsvm_model

class Preprocessing:
    ''' Class to preprocess the data'''

    def __init__(self):
        pass

    def miss_75 (self, df):
        ''' Find number of columns with more than 75% missing values '''
        missing_percent = df.isnull().sum() / len(df)
        missing_columns = missing_percent[missing_percent > 0.75].index
        num_missing_columns = len(missing_columns)
        return num_missing_columns

    def drop_na_cols(self, df1, df2, threshold=75):
        ''' Drop columns with threshold 75% or more missing values in either dataset '''

        # calculate percentage of missing values
        missing_values_train = (df1.isnull().sum() / len(df1)) * 100
        missing_values_test = (df2.isnull().sum() / len(df2)) * 100

        # identify columns in both datasets
        columns_to_drop_train = missing_values_train[missing_values_train >= threshold].index
        columns_to_drop_test = missing_values_test[missing_values_test >= threshold].index

        # get all columns to drop
        all_columns_to_drop = columns_to_drop_train.union(columns_to_drop_test)

        # drop columns that exist in both dataframes
        common_columns_to_drop = df1.columns.intersection(df2.columns).intersection(all_columns_to_drop)
        df1 = df1.drop(columns=common_columns_to_drop)
        df2 = df2.drop(columns=common_columns_to_drop)

        return df1, df2

    def create_features(self, df):
        ''' Feature Engineering function'''

        epsilon = 1e-10  # small constant

        for feature in ['TransactionAmt', 'D15']:
            # Fill NaN values in the original features
            df[feature] = df[feature].fillna(0)

            for group in ['card1', 'card4', 'addr1', 'addr2']:
                # new feature that is the ratio of the original feature to its mean grouped by the current group
                group_mean = df.groupby([group])[feature].transform('mean')
                df[f'{feature}_to_mean_{group}'] = np.where(group_mean != 0, df[feature] / (group_mean + epsilon), 0)

                # new feature that is the ratio of the original feature to its standard deviation grouped by the current group
                group_std = df.groupby([group])[feature].transform('std').fillna(0)
                df[f'{feature}_to_std_{group}'] = np.where(group_std != 0, df[feature] / (group_std + epsilon), 0)

        return df

class Modeling:
    ''' Class to model the data'''

    def __init__(self):
        pass

    def if_hypertune(self, param_grid, X_train, y_train):
        '''Hyperparameter tuning for a model using GridSearchCV'''

        # Initialize the model
        clf = IsolationForest(random_state=42)

        # Initialize the grid search
        grid_search = GridSearchCV(clf, param_grid, scoring='recall', cv=5, n_jobs=-1, refit=True)

        # Fit the grid search on your training data
        grid_search.fit(X_train, y_train)

        # Print the best parameters and the best score
        print(f"Best parameters: {grid_search.best_params_}")
        best_recall = grid_search.best_score_
        print(f"Best recall: {best_recall}")

        # Get the best parameters and the best estimator
        best_params = grid_search.best_params_
        best_estimator = grid_search.best_estimator_

        # Predict the labels of the training set using cross-validation
        y_train_pred = cross_val_predict(best_estimator, X_train, y_train, cv=5)

        # Calculate the precision, F1 score, and PR AUC
        precision = precision_score(y_train, y_train_pred)
        f1 = f1_score(y_train, y_train_pred)
        pr_auc = average_precision_score(y_train, y_train_pred)

        # Return the best parameters, the best estimator, and the metrics
        return best_params, best_estimator, {'Recall': best_recall, 'Precision': precision, 'F1 Score': f1, 'R AUC': pr_auc}

    def lof_hypertune(self, param_grid, X_train, y_train):
        '''Hyperparameter tuning for a model using GridSearchCV'''

        # Initialize the model
        clf = LocalOutlierFactor(novelty=True)

        # Initialize the grid search
        grid_search = GridSearchCV(clf, param_grid, scoring='recall', cv=5, n_jobs=-1, refit=True)

        # Fit the grid search on your training data
        grid_search.fit(X_train, y_train)

        # Print the best parameters and the best recall score
        print(f"Best parameters: {grid_search.best_params_}")
        best_recall = grid_search.best_score_
        print(f"Best recall: {best_recall}")

        # Get the best parameters and the best estimator
        best_params = grid_search.best_params_
        best_estimator = grid_search.best_estimator_

        # Predict the labels of the training set using cross-validation
        y_train_pred = cross_val_predict(best_estimator, X_train, y_train, cv=5)

        # Calculate the precision, F1 score, and PR AUC
        precision = precision_score(y_train, y_train_pred)
        f1 = f1_score(y_train, y_train_pred)
        pr_auc = average_precision_score(y_train, y_train_pred)

        # Return the best parameters, the best estimator, and the metrics
        return best_params, best_estimator, {'Recall': best_recall, 'Precision': precision, 'F1 Score': f1, 'R AUC': pr_auc}

    def ocsvm_hypertune(self, param_grid, X_train, y_train):
        '''Hyperparameter tuning for a model using GridSearchCV'''

        # Initialize the model
        clf = OneClassSVM()

        # Initialize the grid search
        grid_search = GridSearchCV(clf, param_grid, scoring='recall', cv=5, n_jobs=-1, refit=True)

        # Fit the grid search on your training data
        grid_search.fit(X_train, y_train)

        # Print the best parameters and the best recall score
        print(f"Best parameters: {grid_search.best_params_}")
        best_recall = grid_search.best_score_
        print(f"Best recall: {best_recall}")

        # Get the best parameters and the best estimator
        best_params = grid_search.best_params_
        best_estimator = grid_search.best_estimator_

        # Predict the labels of the training set using cross-validation
        y_train_pred = cross_val_predict(best_estimator, X_train, y_train, cv=5)

        # Calculate the precision, F1 score, and PR AUC
        precision = precision_score(y_train, y_train_pred)
        f1 = f1_score(y_train, y_train_pred)
        pr_auc = average_precision_score(y_train, y_train_pred)

        # Return the best parameters, the best estimator, and the metrics
        return best_params, best_estimator, {'Recall': best_recall, 'Precision': precision, 'F1 Score': f1, 'R AUC': pr_auc}

class Evaluation:
    ''' Class to evaluate the models'''

    def __init__(self):
        pass

    def get_cm(self, model_predictions, y):
        '''Get Confusion Matrix for the model'''

            # Adjust labels
        y = [0 if label == 1 else 1 for label in y]
        model_predictions = [0 if prediction == 1 else 1 for prediction in model_predictions]

        tn, fp, fn, tp = confusion_matrix(y, model_predictions).ravel()
        return tn, fp, fn, tp

    def cm_inf(self, cm, model_name):
        '''Print Confusion Matrix for the model'''
        tn, fp, fn, tp = cm
        print(f"Confusion Matrix for model {model_name}:\n")
        print(f"True Positives: {tp}")
        print(f"True Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}\n\n")

    def plot_cm(self, cm, model_name):
        '''Plot the confusion matrix'''

        tn, fp, fn, tp = cm

        # Reconstruct the confusion matrix
        conf_mat = np.array([[tn, fp], [fn, tp]])

        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(10,7))

        # Plot the confusion matrix on the axes
        sns.heatmap(conf_mat, cmap='viridis', ax=ax)

        # Set the labels for the quadrants
        labels = np.array([['TN: ' + str(tn), 'FP: ' + str(fp)], ['FN: ' + str(fn), 'TP: ' + str(tp)]])
        for i in range(2):
            for j in range(2):
                ax.text(j+0.5, i+0.5, labels[i, j], horizontalalignment='center', verticalalignment='center')

        # Set the correct tick labels
        ax.set_xticklabels(['Normal', 'Anomaly'])
        ax.set_yticklabels(['Normal', 'Anomaly'])

        # Set the title
        ax.set_title(f'Confusion Matrix of {model_name}')

        # Show the plot
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

    def print_classreport(self, y_val, y_pred, model_name):
        '''Print the classification report'''

        print(f"Classification Report for {model_name}:\n")
        print(classification_report(y_val, y_pred))
        print("\n")

    def plot_classreport(self, y_val, y_pred, model_name):
        '''Plot the classification report'''

        # Convert report to dataframe
        df_report = pd.DataFrame(classification_report(y_val, y_pred, output_dict=True)).transpose()

        # Exclude 'support' row, 'accuracy' row, and 'macro avg' row
        df_report = df_report.loc[['-1', '1', 'weighted avg'], :]

        # Reset index to have metrics as a column
        df_report.reset_index(level=0, inplace=True)

        # Melt dataframe to have metrics and scores in separate columns
        df_report = pd.melt(df_report, id_vars='index', value_vars=['precision', 'recall', 'f1-score'])

        # Plot barplot
        plt.figure(figsize=(10, 7))
        bars = sns.barplot(x='index', y='value', hue='variable', data=df_report, palette='viridis')

        # Add the values on top of the bars
        for bar in bars.patches:
            bars.annotate(format(bar.get_height(), '.2f'),
                        (bar.get_x() + bar.get_width() / 2., bar.get_height()),
                        ha = 'center', va = 'center',
                        xytext = (0, 10),
                        textcoords = 'offset points')
        plt.title(f'Classification Report of {model_name}')
        plt.legend(title='Metrics')
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.show()

    def get_metrics(self, cm, y_true, y_pred):
        '''Get the evaluation metrics'''

        tn, fp, fn, tp = cm

        # Adjust labels
        y_true = 1 - ((y_true + 1) // 2)  # This will convert -1 to 1 and 1 to 0
        y_pred = 1 - ((y_pred + 1) // 2)  # This will convert -1 to 1 and 1 to 0

        pr_auc = round(average_precision_score(y_true, y_pred), 4)
        precision = round(precision_score(y_true, y_pred), 4)
        recall = round(recall_score(y_true, y_pred), 4)
        f1 = round(f1_score(y_true, y_pred), 4)
        auroc = round(roc_auc_score(y_true, y_pred), 4)
        specificity = round((tn / (tn+fp)), 4)

        return {"PR AUC": pr_auc, "Precision": precision, "Recall": recall, "F1 Score": f1, "AU ROC": auroc, "Specificity": specificity}

    def print_metrics(self, metrics, model_name):
        '''Print the evaluation metrics'''

        print(f"Metrics of {model_name}:\n")
        print(f"Recall: {metrics['Recall']}")
        print(f"Precision: {metrics['Precision']}")
        print(f"F1 Score: {metrics['F1 Score']}")
        print(f"PR AUC: {metrics['PR AUC']}")
        print(f"AU ROC: {metrics['AU ROC']}")
        print(f"Specificity: {metrics['Specificity']}\n\n")

    def plot_metrics(self, metrics, model_name):
        '''Plot the evaluation metrics'''

        # Convert the metrics dictionary to a DataFrame
        df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])

        # Add a column for the model name
        df['Model'] = model_name

        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(10,7))

        # Plot the metrics on the axes
        bars = sns.barplot(x='Metric', y='Value', hue='Metric', data=df, ax=ax, palette='viridis')

        # Add the values on top of the bars
        for bar in bars.patches:
            bars.annotate(format(bar.get_height(), '.2f'),
                        (bar.get_x() + bar.get_width() / 2., bar.get_height()),
                        ha = 'center', va = 'center',
                        xytext = (0, 10),
                        textcoords = 'offset points')

        # Set the title
        ax.set_title(f'Metrics of {model_name}')

        # Show the plot
        plt.show()

    def plot_anomalies(self, anomalies, model_name):
        '''Plot the anomalies detected by the model'''

        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(20,10))

        # Plot the anomalies detected by the model
        sns.boxplot(data=anomalies, palette='viridis')
        ax.set_title(f'Anomalies Detected by {model_name}', fontsize=20)
        ax.tick_params(labelsize=14)
        plt.xticks(rotation='vertical')
        plt.show()

    def plot_4_cm(self, ax, cm, model_name):
        '''Plot the confusion matrix'''

        tn, fp, fn, tp = cm

        # Reconstruct the confusion matrix
        conf_mat = np.array([[tn, fp], [fn, tp]])

        # Plot the confusion matrix on the axes
        sns.heatmap(conf_mat, cmap='viridis', ax=ax)

        # Set the labels for the quadrants
        labels = np.array([['TN: ' + str(tn), 'FP: ' + str(fp)], ['FN: ' + str(fn), 'TP: ' + str(tp)]])
        for i in range(2):
            for j in range(2):
                ax.text(j+0.5, i+0.5, labels[i, j], horizontalalignment='center', verticalalignment='center')

        # Set the correct tick labels
        ax.set_xticklabels(['Normal', 'Anomaly'])
        ax.set_yticklabels(['Normal', 'Anomaly'])

        # Set the title
        ax.set_title(f'Confusion Matrix of {model_name}')

        # Set the axis labels
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

    def plot_4_classreport(self, ax, y_val, y_pred, model_name):
        '''Plot the classification report'''

        # Convert report to dataframe
        df_report = pd.DataFrame(classification_report(y_val, y_pred, output_dict=True)).transpose()

        # Exclude 'support' row, 'accuracy' row, and 'macro avg' row
        df_report = df_report.loc[['-1', '1', 'weighted avg'], :]

        # Reset index to have metrics as a column
        df_report.reset_index(level=0, inplace=True)

        # Melt dataframe to have metrics and scores in separate columns
        df_report = pd.melt(df_report, id_vars='index', value_vars=['precision', 'recall', 'f1-score'])

        # Plot barplot
        bars = sns.barplot(x='index', y='value', hue='variable', data=df_report, palette='viridis', ax=ax)

        # Add the values on top of the bars
        for bar in bars.patches:
            bars.annotate(format(bar.get_height(), '.2f'),
                        (bar.get_x() + bar.get_width() / 2., bar.get_height()),
                        ha='center', va='center',
                        xytext=(0, 10),
                        textcoords='offset points')
        ax.set_title(f'Classification Report of {model_name}')
        ax.legend(title='Metrics')
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')

    def plot_4_metrics(self, ax, metrics, model_name):
        '''Plot the evaluation metrics'''

        # Convert the metrics dictionary to a DataFrame
        df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])

        # Add a column for the model name
        df['Model'] = model_name

        # Plot the metrics on the axes
        bars = sns.barplot(x='Metric', y='Value', hue='Metric', data=df, ax=ax, palette='viridis')

        # Add the values on top of the bars
        for bar in bars.patches:
            bars.annotate(format(bar.get_height(), '.2f'),
                        (bar.get_x() + bar.get_width() / 2., bar.get_height()),
                        ha='center', va='center',
                        xytext=(0, 10),
                        textcoords='offset points')

        # Set the title
        ax.set_title(f'Metrics of {model_name}')
        ax.set_xlabel('Metric')
        ax.set_ylabel('Value')

    def plot_4_anomalies(self, ax, labels, anomalies, model_name):
        '''Plot the anomalies detected by the model'''

        # Plot the anomalies detected by the model
        sns.boxplot(data=anomalies, palette='viridis', ax=ax)
        ax.set_title(f'Anomalies Detected by {model_name}', fontsize=20)
        ax.tick_params(labelsize=14)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_xlabel('Features')
        ax.set_ylabel('Values')

class Interpret:
    ''' Class to interpret the models '''

    def run_lime_explanation_if_ocsvm(self, model, X_test, feature_names, class_names, instance_index):
        ''' Run LIME explanation for Isolation Forest and One-Class SVM models '''

        # Create a Lime Explainer
        explainer = lime_tabular.LimeTabularExplainer(X_test.values,
                                                    feature_names=feature_names,
                                                    class_names=class_names,
                                                    discretize_continuous=True)

        # Define a predict_proba function for the model
        def predict_proba(X):
            # Convert X to a DataFrame with the correct feature names
            X = pd.DataFrame(X, columns=feature_names)
            decisions = model.decision_function(X)
            # Convert anomaly scores to probabilities using the sigmoid function
            proba_anomaly = expit(decisions)
            proba_normal = 1 - proba_anomaly
            return np.stack((proba_normal, proba_anomaly), axis=-1)

        # Use the provided instance index for explanation
        exp = explainer.explain_instance(X_test.values[instance_index], predict_proba)

        # Show the explanation
        exp.show_in_notebook(show_table=True, show_all=False)

        return exp

    def run_lime_explanation_lof(self, model, X_test, feature_names, class_names, instance_index):
        ''' Run LIME explanation for Local Outlier Factor model'''

        # Create a Lime Explainer
        explainer = lime_tabular.LimeTabularExplainer(X_test.values,
                                                    feature_names=feature_names,
                                                    class_names=class_names,
                                                    discretize_continuous=True)

        # Define a predict_proba function for the model
        def predict_proba(X):
            # Convert X to a DataFrame with the correct feature names
            X = pd.DataFrame(X, columns=feature_names)
            decisions = model.decision_function(X)
            # Convert normality scores to probabilities using the sigmoid function
            proba_normal = expit(decisions)
            proba_anomaly = 1 - proba_normal
            return np.stack((proba_normal, proba_anomaly), axis=-1)

        # Use the provided instance index for explanation
        exp = explainer.explain_instance(X_test.values[instance_index], predict_proba)

        # Show the explanation
        exp.show_in_notebook(show_table=True, show_all=False)

        return exp
