from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from preprocessing import *


def svm_model(df):
    X_hotels, y_hotels = split_data_into_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X_hotels, y_hotels,
                                                        test_size=0.15,
                                                        random_state=239,
                                                        shuffle=True,
                                                        stratify=y_hotels)
    svm = SVC(kernel='linear', random_state=239)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    return precision_recall_fscore_support(y_test, y_pred, average='weighted'), confusion_matrix(y_test, y_pred)



def grid_search(df, model, param_grid):
    param_grid = {
        'C': [1, 2, 5, 10, 20],
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [500, 1000, 1500]
    }

    grid = GridSearchCV(LogisticRegression(),
                        param_grid=param_grid,
                        n_jobs=-1,
                        cv=5,
                        scoring='accuracy')
    X_hotels, y_hotels = split_data_into_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X_hotels, y_hotels,
                                                        test_size=0.15,
                                                        random_state=239,
                                                        shuffle=True,
                                                        stratify=y_hotels)
    grid.fit(X_train, y_train)
    print(f"Best parameters: {grid.best_params_}")
    print(f"Best score: {grid.best_score_}")


def logistic_model_training(df):
    X_hotels, y_hotels = split_data_into_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X_hotels, y_hotels,
                                                        test_size=0.15,
                                                        random_state=239,
                                                        shuffle=True,
                                                        stratify=y_hotels)
    logistic = LogisticRegression(solver='lbfgs', max_iter=1000)
    logistic.fit(X_train, y_train)
    y_pred = logistic.predict(X_test)
    return precision_recall_fscore_support(y_test, y_pred, average='weighted'), confusion_matrix(y_test, y_pred)
