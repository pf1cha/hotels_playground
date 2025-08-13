import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)


def counting_null_for_each_column(df):
    print(df.isnull().sum())


def scaling_features(df):
    scaler = MinMaxScaler()
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    return df


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
    df_dummies = pd.get_dummies(df, columns=['meal', 'distribution_channel',
                                             'customer_type', 'hotel', 'deposit_type'], drop_first=True)
    df_dummies = scaling_features(df_dummies)
    return df_dummies


def split_data_into_features_and_target(df):
    X = df.drop('is_canceled', axis=1)
    y = df['is_canceled']
    return X, y
