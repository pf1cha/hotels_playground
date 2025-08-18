from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from preprocessing import *


def feature_importance(df):
    X_hotels, y_hotels = split_data_into_features_and_target(df)

    categorical_features = X_hotels.select_dtypes(include=['object']).columns
    numerical_features = X_hotels.select_dtypes(include=['int64', 'float64']).columns

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(drop='first'), categorical_features),
        ('num', 'passthrough', numerical_features),
    ])
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('rf', RandomForestClassifier(random_state=239, n_estimators=100))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X_hotels, y_hotels,
                                                        test_size=0.2,
                                                        random_state=239,
                                                        shuffle=True,
                                                        stratify=y_hotels)
    model.fit(X_train, y_train)
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    feature_map = []
    for f in feature_names:
        if f.startswith("cat__"):
            words = f.split("__")[1].split("_")[:-1]
            orig = "_".join(words)
        elif f.startswith("num__"):
            orig = f.split("__")[1]
        else:
            orig = f
        feature_map.append(orig)

    importance = model.named_steps['rf'].feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': feature_map, 'Importance': importance})
    feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    return feature_importance_df


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
                                                        test_size=0.2,
                                                        random_state=239,
                                                        shuffle=True,
                                                        stratify=y_hotels)
    logistic = LogisticRegression(solver='liblinear', max_iter=500, penalty='l1', random_state=239, C=20)
    logistic.fit(X_train, y_train)
    y_pred = logistic.predict(X_test)
    y_train_pred = logistic.predict(X_train)
    return precision_recall_fscore_support(y_test, y_pred, average='weighted'), confusion_matrix(y_test, y_pred)
