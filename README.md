# Survivors of the Titanic prediction project
This project is an example for building a web server that train a model to predict the survivors of the Titanic

## How to run the project
1. Create environment with conda: <code>conda create -n env python=3.8</code>
2. Clone project and go to inside
3. Install requirement library: <code>pip install -r requirements.txt</code>
4. Start the server <code>python src/app.py</code> Default port is 8000. You can change it in src/utils/constants.py

## How to use the server

### Upload data into server
You need to request API <code>http://127.0.0.1:8000/upload</code> with method POST and csv file (using keyword file). You also need to add the source of the data by adding form with key <code>{src:YOUR_TARGET}</code>. YOUR_TARGET can either be train or test which will mark the data as the training data or testing data. If the data was uploaded succesfully, it will return status OK and the data id, otherwise the error will be notice.

### Check list of uploaded data
You can check the data you uploaded via API <code>http://127.0.0.1:8000/data?src=train</code> with method GET. It will return the list of uploaded files and some information. Replace <code>src=test</code> for checking testing data list.

### Train a model
***Before training a model, please make sure that you uploaded the training dataset and testing dataset***
To train a model you need to request API <code>http://127.0.0.1:8000/train</code> with method POST and config file (using keyword config).  After the training finish, it will return the model result and its id along with other information such as created time. You can provide a retraining with the same config to achieve the same model.

_Note that the training is assuming that it does not take so much time, in the future this function should be upgraded with asynchronous request._

Below is an example config file.

```
model:
  name: RandomForestClassifier      # Name of the algorithm. Currently only support RandomForestClassifier and LogisticRegression    
  config:                           # Config of the algorithm. It follow sklearn config
    n_estimators: 100
data:
  features: {'Pclass':'num', 'Sex':'cat', 'Age':'num', 'Fare':num}      # Mapping between features and data type. Model will be trained with only these feature. If a feature is cat, it will be transformed into one-hot encoding before training.
  target: {'Survived':'cat'}                                            # Target column in the data
  train_id_list: null                                                   # List of training id. Only use the uploaded data whose id in this list for the training. If the list is null or empty, the training will use all of uploaded training data.
  test_id_list: null                                                    # List of testing id. Only use the uploaded data whose id in this list for the testing. If the list is null or empty, the training will use all of uploaded testing data.
```

### Check all trained model
You can check the list of trained models via API <code>http://127.0.0.1:8000/models</code> with method GET. It will return the list of trained and its config.

### Run test models
You can run test a list of models via API <code>http://127.0.0.1:8000/test_models</code> with method POST and a config file (using keyword config). It will test all the model in list and return the evaluation metrics as well as the prediction.

Below is an example config

```
models:
  models: []            # List of models id to be tested. If the list is null or empty test all models
data:
  test_id_list: null    # List of testing id. Only use the uploaded data whose id in this list for the testing. If the list is null or empty, the training will use all of uploaded testing data.
```

### Assign a model
After checking the model result, you can select the default model for the predicting system. Use API <code>http://127.0.0.1:8000/setup_model</code> with method POST and params {model_id:MODEL_ID}. 

### Predict the survivors of the Titanic
You can get the prediction by upload a csv of data into the model. The csv should not have missing value in the features. The server will use the model registed above to give the prediction. Use API <code>http://127.0.0.1:8000/predict</code> with method POST and csv file (using keyword file).
***Before predicting a model, please make sure that you assigned the model as above***
