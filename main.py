from model import *
from preprocessing import *
from plot_results import *


def print_results(info_about_results, confusion_matrix, hotels_label, name=None):
    print(f"Precision: {info_about_results[0]}")
    print(f"Recall: {info_about_results[1]}")
    print(f"F1 Score: {info_about_results[2]}")
    plot_confusion_matrix(confusion_matrix, hotels_label, name=name)


def save_feature_importance_for_hotels(df):
    feature_importance_df = feature_importance(df)
    plot_feature_importance(feature_importance_df)
    the_most_important = feature_importance_df['Feature'].head(8).tolist()
    return the_most_important


def train_model(df):
    df_dummies = handling_categorical_features(df)
    scaled_df = scaling_features(df_dummies)
    info_about_results, confusion_matrix = logistic_model_training(scaled_df)
    hotels_label = ['Not Canceled', 'Canceled']
    print_results(info_about_results, confusion_matrix, hotels_label, name='logistic_model_cm.png')


if __name__ == "__main__":
    hotels = pd.read_csv('data/hotel_booking.csv')
    updated_hotels = basic_cleaning(hotels)
    important_features = save_feature_importance_for_hotels(updated_hotels) + ['is_canceled']
    important_hotels = drop_non_important_features(updated_hotels, important_features)
    train_model(important_hotels)
