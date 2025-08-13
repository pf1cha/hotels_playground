import pandas as pd
from model import train_model


if __name__ == "__main__":
    hotels = pd.read_csv('data/hotel_booking.csv')
    train_model(hotels)

