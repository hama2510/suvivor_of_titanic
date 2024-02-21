import os

DATA_PATH = 'data'
DATA_TRAIN_PATH = os.path.join(DATA_PATH, 'train')
DATA_TEST_PATH = os.path.join(DATA_PATH, 'test')
MODEL_WEIGHT_PATH = 'weights'
CONFIG_PATH = 'config/server_config.yaml'
DATA_COLUMNS = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
PORT = 8000