import pandas as pd
from model import logistic_model_training, grid_search, svm_model
from preprocessing import *
from plot_results import *

if __name__ == "__main__":
    hotels = pd.read_csv('data/hotel_booking.csv')
    hotels_cleaned = cleaned_data(hotels)
    # correlation_heatmap(hotels_cleaned)
    hotels_dummies = handling_categorical_features(hotels_cleaned)
    hotels_scaled = scaling_features(hotels_dummies)
    info_about_results, confusion_matrix = svm_model(hotels_scaled)
    hotels_label = ['Not Canceled', 'Canceled']
    plot_confusion_matrix(confusion_matrix, hotels_label)
    print(f"Precision: {info_about_results[0]}")
    print(f"Recall: {info_about_results[1]}")
    print(f"F1 Score: {info_about_results[2]}")

    # grid_search(hotels_scaled)
