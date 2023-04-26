import numpy as np


def make_hour_sin(seconds):
    seconds_in_day = 24 * 60 * 60
    return np.sin(2 * np.pi * seconds / seconds_in_day)


def make_hour_cos(seconds):
    seconds_in_day = 24 * 60 * 60
    return np.cos(2 * np.pi * seconds / seconds_in_day)


def feature_engineering(train):
    train['hour_sin'] = make_hour_cos(train['TransactionDT'])
    train['hour_cos'] = make_hour_cos(train['TransactionDT'])

    return train
