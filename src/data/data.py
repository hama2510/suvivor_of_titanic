import pandas as pd
from sklearn.model_selection import train_test_split
from glob import glob
from utils.constants import DATA_TRAIN_PATH, DATA_TEST_PATH, DATA_COLUMNS
from utils.utils import get_time_string, get_id
import os
import pickle

def get_path(type):
    if type=='train':
        path =  DATA_TRAIN_PATH
    elif type=='test':
        path = DATA_TEST_PATH
    elif type=='raw':
        path = DATA_TEST_PATH
    else:
        raise ValueError(f'Type {type} of data is not supported.')
    return path

def check_data_structure(df):
    if len(df.columns)!=len(DATA_COLUMNS):
        raise ValueError('Data structure is incorrect. There are some missing columns')
    for col in df.columns:
        if not col in DATA_COLUMNS:
            raise ValueError(f'Column {col} is not supported')
    if len(df)==0:
        raise ValueError('File contains nothing')

def check_data_config(data_config):
    if len(data_config.target)!=1:
        raise ValueError('Too much target columns')
    if list(data_config.target.keys())[0] in data_config.features.keys():
        raise ValueError('Target columns exists in feature colummns')

def get_all_data_ids(type):
    path = get_path(type)
    files = glob(os.path.join(path, '**', '*.csv'), recursive=True)
    ids = []
    for f in files:
        id = f.split('/')[-2]
        filename = f.split('/')[-1]
        info = f.replace(filename, f'{id}.pkl')
        info = pickle.load(open(info, 'rb'))
        ids.append({'id':id, 'filename':filename, 'info':info})
    return ids

def get_data(type, ids=None):
    path = get_path(type)
    files = glob(os.path.join(path, '**', '*.csv'), recursive=True)
    data = []
    ret = {'ids':[]}
    for f in files:
        id = f.split('/')[-2]
        if not ids is None and len(ids)>0:
            if not id in ids:
                continue
        df = pd.read_csv(f, header=0)
        ret['ids'].append(id)
        data.append(df)
    df_data = pd.concat(data, ignore_index=True)
    ret['data'] = df_data
    return ret

def get_data_type(type):
    if type=='num':
        return 'float'
    elif type=='cat':
        return 'category'
    else:
        raise ValueError(f'Type {type} is not supported')

def data_to_feature_and_label(df, data_config):
    check_data_config(data_config)
    feature_cat_cols = []
    for col, data_type in data_config.features.items():
        data_type = get_data_type(data_type)
        if data_type=='category':
            feature_cat_cols.append(col)
        df[col] = df[col].astype(data_type, errors='ignore')
    for col, data_type in data_config.target.items():
        data_type = get_data_type(data_type)
        df[col] = df[col].astype(data_type, errors='ignore')
    X_cols = [col for col in data_config.features]
    Y_cols = [col for col in data_config.target]
    X = df[X_cols]
    y = df[Y_cols].values.reshape(-1)
    X_encoded = pd.get_dummies(X, columns=feature_cat_cols)
    return X_encoded, y

def preprocess_data(df):
    df = df.dropna(axis=0)
    return df

def data_to_feature(df, data_config):
    check_data_config(data_config)
    feature_cat_cols = []
    for col, data_type in data_config.features.items():
        data_type = get_data_type(data_type)
        if data_type=='category':
            feature_cat_cols.append(col)
        df[col] = df[col].astype(data_type, errors='ignore')
    X_cols = [col for col in data_config.features]
    X = df[X_cols]
    X_preprocessed = preprocess_data(X)
    if len(X_preprocessed)!=len(X):
        raise ValueError('Input data has missing value')
    X_encoded = pd.get_dummies(X_preprocessed, columns=feature_cat_cols)
    for col in data_config['encoded_features']:
        if not col in X_encoded.columns:
            X_encoded[col] = 0
    return X_encoded

def save_data(df, type, filename):
    path = get_path(type)
    id = get_id()
    file_path = os.path.join(path, id, f'{filename}')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    info = {'created_time':get_time_string(), 'src':type}
    info_path = file_path.replace(filename, f'{id}.pkl')
    pickle.dump(info, open(info_path, 'wb'))
    return id
