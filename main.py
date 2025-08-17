from model import *
from preprocessing import *
from plot_results import *


def print_results(info_about_results, confusion_matrix, hotels_label):
    print(f"Precision: {info_about_results[0]}")
    print(f"Recall: {info_about_results[1]}")
    print(f"F1 Score: {info_about_results[2]}")
    plot_confusion_matrix(confusion_matrix, hotels_label)


if __name__ == "__main__":
    hotels = pd.read_csv('data/hotel_booking.csv')
    hotels_cleaned = cleaned_data(hotels)
    hotels_dummies = handling_categorical_features(hotels_cleaned)
    # hotels_scaled = scaling_features(hotels_dummies)
    hotels_label = ['Not Canceled', 'Canceled']
    feature_importance_df, info_about_results, confusion_matrix = feature_importance(hotels_dummies)
    print_results(info_about_results, confusion_matrix, hotels_label)
    plot_feature_importance(feature_importance_df)
