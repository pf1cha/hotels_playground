from sklearn.model_selection import train_test_split
from preprocessing import cleaned_data, split_data_into_features_and_target, correlation_heatmap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def train_model(df):
    correlation_heatmap(df)
    updated_hotels = cleaned_data(df)
    X_hotels, y_hotels = split_data_into_features_and_target(updated_hotels)
    X_train, X_test, y_train, y_test = train_test_split(X_hotels, y_hotels,
                                                        test_size=0.15,
                                                        random_state=239,
                                                        shuffle=True,
                                                        stratify=y_hotels)
    logistic = LogisticRegression(solver='lbfgs', max_iter=1000)
    logistic.fit(X_train, y_train)
    y_pred = logistic.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
