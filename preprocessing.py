import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

pd.set_option('display.max_columns', None)


def counting_null_for_each_column(df):
    print(df.isnull().sum())


def information_about_unique_values_for_categorical_data(df):
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        unique_values = df[column].unique()
        print(f"Column: {column}, Unique Values: {unique_values}, Count: {len(unique_values)}")
    print("\n")


def scaling_features(df):
    scaler = MinMaxScaler()
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    return df


def handling_categorical_features(df):
    categorical = df.select_dtypes(include=['object']).columns
    df_dummies = pd.get_dummies(df, columns=categorical, drop_first=True)
    df_dummies = scaling_features(df_dummies)
    return df_dummies


def cleaned_data(df):
    df.drop(['company', 'agent', 'name', 'country',
             'email', 'phone-number', 'credit_card',
             'reservation_status_date', 'reservation_status',
             'adr', 'arrival_date_year', 'arrival_date_week_number',
             'market_segment'], axis=1, inplace=True)

    df['arrival_date_month'] = df['arrival_date_month'].map({
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    })

    df[['meal', 'distribution_channel']] = df[['meal',
                                               'distribution_channel']].replace('Undefined', np.nan)
    df.dropna(subset=['meal', 'distribution_channel'], axis=0, inplace=True)

    df[['children']] = df[['children']].fillna(value=0)
    df['room_type_was_changed'] = df['reserved_room_type'] != df['assigned_room_type']
    df.drop(['reserved_room_type', 'assigned_room_type'], axis=1, inplace=True)
    df['stays_in_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df.drop(['stays_in_weekend_nights', 'stays_in_week_nights'], axis=1, inplace=True)
    df['children'] = df['children'].astype(int)
    return df


def split_data_into_features_and_target(df):
    X = df.drop('is_canceled', axis=1)
    y = df['is_canceled']
    return X, y


def basic_cleaning(df):
    # TODO replace inplace=True with df = df.dropna() to avoid modifying the original dataframe
    df.drop(columns=['agent', 'company'], axis=1, inplace=True)  # Due to high number of null values
    df['country'].fillna(df['country'].mode()[0], inplace=True)  # Fill nulls with mode
    df[['meal', 'distribution_channel']] = df[['meal',
                                               'distribution_channel']].replace('Undefined', np.nan)
    df.dropna(subset=['meal', 'distribution_channel'], axis=0, inplace=True)
    df = df.drop(columns=['country'], axis=1)  # Drop rows where children is NaN
    # Drop columns that are not useful for future prediction model
    df.drop(columns=['email', 'phone-number', 'credit_card', 'name'], axis=1, inplace=True)
    df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
    df['arrival_date'] = (df['arrival_date_year'].astype(str) + '-'
                          + df['arrival_date_month'].astype(str) + '-'
                          + df['arrival_date_day_of_month'].astype(str))
    df['arrival_date'] = pd.to_datetime(df['arrival_date'])
    df = df.drop(columns=['arrival_date_year', 'arrival_date_month',
                          'arrival_date_day_of_month', 'arrival_date_week_number'], axis=1)
    df['difference_in_days'] = (df['reservation_status_date'] - df['arrival_date']).dt.days
    df['difference_in_days'] = df['difference_in_days'].astype(int)
    df = df.drop(columns=['reservation_status_date', 'arrival_date'], axis=1)

    df['room_type_was_changed'] = df['reserved_room_type'] != df['assigned_room_type']
    df.drop(['reserved_room_type', 'assigned_room_type'], axis=1, inplace=True)

    return df


def drop_non_important_features(df, important_features):
    important_features = df[important_features].copy()
    return important_features
