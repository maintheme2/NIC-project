import numpy as np


def make_hour_sin(TransactionDT):
    seconds_in_day = 24 * 60 * 60
    return np.sin(2 * np.pi * TransactionDT / seconds_in_day)


def make_hour_cos(TransactionDT):
    seconds_in_day = 24 * 60 * 60
    return np.cos(2 * np.pi * TransactionDT / seconds_in_day)


def get_Transaction_Day(TransactionDT):
    seconds_in_day = 24 * 60 * 60
    return TransactionDT / seconds_in_day


def get_DayBegan(dataset):
    return dataset['TransactionDay'] - dataset['D1']


def get_Last_Day_Transaction(dataset):
    return dataset['TransactionDay'] - dataset['D3']


def feature_engineering(data):
    data['hour_sin'] = make_hour_sin(data['TransactionDT'])
    data['hour_cos'] = make_hour_cos(data['TransactionDT'])

    data['day'] = get_Transaction_Day(data['TransactionDT'])

    data['uid1'] = (data.day - data.D1).astype(str) + '_' + \
                   data.P_emaildomain.astype(str)

    data['uid2'] = (data.card1.astype(str) + '_' +
                    data.addr1.astype(str) + '_' +
                    (data.day - data.D1).astype(str) + '_' +
                    data.P_emaildomain.astype(str))

    data = data_group_features(data, 'uid1')

    return data


def data_group_features(data, col):
    groupped_dataset = data.groupby(col)
    data[col + '_count_'] = groupped_dataset.TransactionID.transform('count').astype('int32')
    data[col + '_mean_amt'] = groupped_dataset.TransactionAmt.transform('mean').astype('float32')
    data[col + '_median_amt'] = groupped_dataset.TransactionAmt.transform('median').astype('float32')

    return data
