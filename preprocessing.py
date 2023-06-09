import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders.woe import WOEEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from feature_engineering import feature_engineering
from sklearn.impute import SimpleImputer


def input_dataframe(train_transaction_path, train_identity_path):
    train_transaction = pd.read_csv(train_transaction_path)
    train_identity = pd.read_csv(train_identity_path)
    return pd.merge(train_transaction, train_identity, on='TransactionID', how='left')


def get_cat_features(train):
    return train.select_dtypes(exclude=np.number).columns


def get_num_features(train, cat_features):
    exclude = ['TransactionID', 'TransactionDT', 'isFraud']
    exclude.extend(cat_features.tolist())
    num_features = train.drop(exclude, axis=1).columns
    return num_features


def drop_features(train, threshold):
    train = train.dropna(thresh=len(train) * threshold, axis=1)

    use_cols = train.columns
    cat_features = get_cat_features(train)
    num_features = get_num_features(train, cat_features)

    train[cat_features] = train[cat_features].astype(str)
    train[num_features] = train[num_features].astype(np.float32)

    return train[use_cols]


def fill_empty_cells(train, cat_features, num_features):
    median_values = train[num_features].median()

    train[num_features] = train[num_features].fillna(median_values)
    imputer = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
    for cat in cat_features:
        train[cat] = imputer.fit_transform(train[cat].values.reshape(-1,1))[:,0]
    return train


def get_encoders(data, cat_features):
    to_ohe = []
    to_emb = []

    for c in cat_features:
        if data[c].nunique() < 5:
            to_ohe.append(c)
        else:
            to_emb.append(c)

    return to_ohe, to_emb


def get_train_test_data(train_transaction_path, train_identity_path, threshold=0.5):
    train = input_dataframe(train_transaction_path, train_identity_path)
    train = drop_features(train, threshold)

    cat_features = get_cat_features(train)
    num_featurs = get_num_features(train, cat_features)

    train = fill_empty_cells(train, cat_features, num_featurs)

    # print("BEFORE TRAIN", (train.isnull().sum() > 0).sum())

    train = feature_engineering(train)

    data = train.drop(columns=['TransactionID', 'TransactionDT'])
    target = 'isFraud'

    # print("TRAIN", (train.isnull().sum() > 0).sum())
    # print(train.isna().any())
    num_features = data._get_numeric_data().columns.to_list()
    if target in num_features:
        num_features.remove(target)
    cat_features = [x for x in data.columns if x not in num_features and x != target]

    x_train, x_test, y_train, y_test = train_test_split(data.drop([target], axis=1),
                                                        data[target],
                                                        train_size=0.8)

    to_ohe, to_emb = get_encoders(x_train, cat_features)

    scaler = StandardScaler()
    ohe = OneHotEncoder(handle_unknown='ignore')
    woe = WOEEncoder()

    column_trans = ColumnTransformer(
        [('scaler', scaler, num_features),
         ('ohe', ohe, to_ohe),
         ('woe', woe, to_emb)], remainder='passthrough', n_jobs=-1)

    column_trans.fit(x_train, y_train)

    return (pd.DataFrame(column_trans.fit_transform(x_train, y_train)),
            pd.DataFrame(column_trans.transform(x_test)),
            y_train,
            y_test)
