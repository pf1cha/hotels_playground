from model import *
from preprocessing import *
from plot_results import *
from sklearnex import patch_sklearn

patch_sklearn()


def print_results_for_validation(info_about_results, hotels_label, confusion_matrix, name=None):
    print(f"Precision: {info_about_results[0]}.3f")
    print(f"Recall: {info_about_results[1]}.3f")
    print(f"F1 Score: {info_about_results[2]}.3f")
    plot_confusion_matrix(confusion_matrix, hotels_label, name=name)


def print_results_for_grid_search(best_params, best_score, info_about_test):
    print(f"Best Parameters: {best_params}")
    print(f"Best Score: {best_score}")
    print(f"Test Precision: {info_about_test[0]}")
    print(f"Test Recall: {info_about_test[1]}")
    print(f"Test F1 Score: {info_about_test[2]}")


def save_feature_importance_for_hotels(df):
    feature_importance_df = feature_importance(df)
    # plot_feature_importance(feature_importance_df)
    the_most_important = feature_importance_df['Feature'].head(8).tolist()
    return the_most_important


def train_model(df):
    df_dummies = handling_categorical_features(df)
    scaled_df = scaling_features(df_dummies)
    info_about_results, confusion_matrix = svm_model(scaled_df)
    hotels_label = ['Not Canceled', 'Canceled']
    print_results_for_validation(info_about_results, hotels_label, confusion_matrix, name='svm_model_cm.png')


if __name__ == "__main__":
    hotels = pd.read_csv('data/hotel_booking.csv')
    # small_hotels = hotels.sample(frac=0.2, random_state=239)
    updated_hotels = basic_cleaning(hotels)
    important_features = save_feature_importance_for_hotels(updated_hotels) + ['is_canceled']
    important_hotels = drop_non_important_features(updated_hotels, important_features)
    train_model(important_hotels)
