from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from utils.constants import MODEL_WEIGHT_PATH
from utils.utils import get_id, get_time_string
import os
import pickle
from omegaconf import OmegaConf
from data.data import get_data, data_to_feature_and_label, preprocess_data
from models.metrics import cal_metrics
import numpy as np
from glob import glob

def load_model(model_config):
    if not 'weights' in model_config.keys():
        if not model_config['name'] in model_factory.keys():
            raise ValueError(f'Only support model in list {list(model_factory.keys())}')
        model = model_factory[model_config['name']]
        config = model_config['config'] if 'config' in model_config.keys() else {}
        if not 'random_state' in config and model['random_state']:
            config['random_state'] = 42
        model = model['obj'](**config)
    else:
        model_path = os.path.join(MODEL_WEIGHT_PATH, model_config.weights, f'{model_config.weights}.pkl')
        config_path = os.path.join(MODEL_WEIGHT_PATH, model_config.weights, 'config.yaml')
        if not os.path.exists(model_path):
            raise ValueError('Model does not existed')
        if not os.path.exists(config_path):
            raise ValueError('Model config does not existed')
        model = pickle.load(open(model_path, 'rb'))
        model_config = OmegaConf.load(config_path)
        model_config['created_time'] = model['time']
        model = model['weight']
    return model, model_config

def save_model(model, config):
    id = get_id()
    data = {'weight':model, 'config':OmegaConf.to_container(config, resolve=True), 'id':id, 'time':get_time_string()}
    path = os.path.join(MODEL_WEIGHT_PATH, id)
    os.makedirs(path, exist_ok=True)
    pickle.dump(data, open(os.path.join(path, f'{id}.pkl'), 'wb'))
    OmegaConf.save(config, os.path.join(path, 'config.yaml'))
    return data

def check_config(config):
    assert 'model' in config, 'Config must contain keyword model'
    assert 'data' in config, 'Config must contain keyword data'

def train_model(config):
    check_config(config)
    model, _ = load_model(config.model)
    train_data = get_data(type='train', ids=config.data.train_id_list)
    train_data_ids = train_data['ids']
    train_data = preprocess_data(train_data['data'])
    X_train, y_train = data_to_feature_and_label(train_data, config.data)
    model.fit(X_train, y_train)
    ret = test_model(model, config)
    return {'model':model, 'encoded_features':X_train.columns.values.tolist(), 'test_data_ids':ret['test_data_ids'], 'train_data_ids':train_data_ids, 'metrics':ret['metrics']}

def test_model(model, config):
    check_config(config)
    if model is None:
        model, pretrained_config = load_model(config.model)
    else:
        pretrained_config = config
    test_data = get_data(type='test', ids=config.data.test_id_list)
    test_data_ids = test_data['ids']
    test_data = preprocess_data(test_data['data'])
    X_test, y_test = data_to_feature_and_label(test_data, pretrained_config.data)
    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred, decimals=0)
    ret = cal_metrics(y_test, y_pred)
    return {'model':model, 'config':pretrained_config, 'test_data_ids':test_data_ids, 'y_test':y_test, 'y_pred':y_pred, 'metrics':ret}

def model_predict(model, data):
    y_pred = model.predict(data)
    y_pred = np.round(y_pred, decimals=0)
    return y_pred

def get_all_models():
    ret = []
    files = glob(os.path.join(MODEL_WEIGHT_PATH, '**', '*.pkl'), recursive=True)
    for f in files:
        model = pickle.load(open(f, 'rb'))
        del model['weight']
        ret.append(model)
    return ret

model_factory = {
    'LogisticRegression': {'obj':LogisticRegression, 'random_state':False},
    'RandomForestClassifier': {'obj':RandomForestClassifier, 'random_state':True}
}
