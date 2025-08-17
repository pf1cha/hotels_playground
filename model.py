from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from preprocessing import *


def feature_importance(df):
    X_hotels, y_hotels = split_data_into_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X_hotels, y_hotels,
                                                        test_size=0.15,
                                                        random_state=239,
                                                        shuffle=True,
                                                        stratify=y_hotels)
    rf = RandomForestClassifier(random_state=239, n_estimators=100)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    feature_names = X_hotels.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    y_pred = rf.predict(X_test)
    return feature_importance_df, precision_recall_fscore_support(y_test, y_pred, average='weighted'), confusion_matrix(
        y_test, y_pred)


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


def model_grid_search(df, model, param_grid):
    grid = GridSearchCV(model,
                        param_grid=param_grid,
                        cv=5,
                        scoring='recall'
                        )
    X_hotels, y_hotels = split_data_into_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X_hotels, y_hotels,
                                                        test_size=0.2,
                                                        random_state=239,
                                                        shuffle=True,
                                                        stratify=y_hotels)
    grid.fit(X_train, y_train)
    best_params = grid.best_params_
    best_score = grid.best_score_
    return best_params, best_score


def logistic_model_training(df):
    X_hotels, y_hotels = split_data_into_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X_hotels, y_hotels,
                                                        test_size=0.15,
                                                        random_state=239,
                                                        shuffle=True,
                                                        stratify=y_hotels)
    logistic = LogisticRegression(solver='liblinear', max_iter=500, penalty='l1', random_state=239, C=20)
    logistic.fit(X_train, y_train)
    y_pred = logistic.predict(X_test)
    return precision_recall_fscore_support(y_test, y_pred, average='weighted'), confusion_matrix(y_test, y_pred)
