import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders.woe import WOEEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


def input_dataframe():
    train_transaction = pd.read_csv('train_transaction.csv')
    train_identity = pd.read_csv('train_identity.csv')
    return pd.merge(train_transaction, train_identity, on='TransactionID', how='left')


def make_hour_feature(f):
    hours = f / 3600
    encoded_hours = np.floor(hours) % 24
    return encoded_hours


def get_cat_features():
    return ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'P_emaildomain',
            'R_emaildomain', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'DeviceType', 'DeviceInfo',
            'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21',
            'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30',
            'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38']


def get_num_features(train, cat_features):
    exclude = ['TransactionID', 'TransactionDT', 'isFraud']
    return [f for f in train.columns if (f not in cat_features) & (f not in exclude)]


def drop_features(train, cat_features, num_features):
    col_na = train.isna().sum()
    to_drop = col_na[(col_na / train.shape[0]) > 0.5].index

    print("use_cols:")
    use_cols = [f for f in tqdm(train.columns) if f not in to_drop]

    print("cat_features:")
    cat_features = [f for f in tqdm(cat_features) if f not in to_drop]

    print("num_features:")
    num_features = [f for f in tqdm(num_features) if f not in to_drop]

    train[cat_features] = train[cat_features].astype(str)
    train[num_features] = train[num_features].astype(np.float)

    return train[use_cols], to_drop


def fill_empty_cells(train, cat_features, num_features):
    median_values = train[num_features].median()

    train[num_features] = train[num_features].fillna(median_values)
    train[cat_features] = train[cat_features].replace("nan", "missing")


def get_encoders(data, cat_features):
    to_ohe = []
    to_emb = []

    for c in cat_features:
        if data[c].nunique() < 5:
            to_ohe.append(c)
        else:
            to_emb.append(c)

    return to_ohe, to_emb


def get_test_train_data():
    train = input_dataframe()
    train['hour'] = make_hour_feature(train['TransactionDT'])

    cat_features = get_cat_features()
    num_featurs = get_num_features(train, cat_features)
    train, to_drop = drop_features(train, cat_features, num_featurs)
    fill_empty_cells(train, cat_features, num_featurs)

    data = train.drop(columns=['TransactionID', 'TransactionDT'])
    target = 'isFraud'

    num_features = data.select_dtypes(include=np.number).columns
    num_features = [f for f in tqdm(num_features) if f != target]
    cat_features = data.select_dtypes(exclude=np.number).columns

    x_train, x_test, y_train, y_test = train_test_split(data[num_features + list(cat_features)],
                                                        data['isFraud'],
                                                        train_size=0.2,
                                                        test_size=0.04)

    to_ohe, to_emb = get_encoders(x_train, cat_features)

    scaler = StandardScaler()
    ohe = OneHotEncoder(handle_unknown='ignore')
    woe = WOEEncoder()

    column_trans = ColumnTransformer(
        [('scaler', scaler, num_features),
         ('ohe', ohe, to_ohe),
         ('woe', woe, to_emb)], remainder='passthrough', n_jobs=-1)

    return (pd.DataFrame(column_trans.fit_transform(x_train, y_train)),
            pd.DataFrame(column_trans.transform(x_test)),
            y_train,
            y_test)


if __name__ == '__main__':
    preprocess()
