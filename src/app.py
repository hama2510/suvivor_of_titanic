from flask import Flask, request, jsonify
import pandas as pd
from data.data import save_data, check_data_structure, data_to_feature, get_all_data_ids
from models.model import train_model, save_model, test_model, load_model, model_predict, get_all_models
from omegaconf import OmegaConf
from io import StringIO
from utils.constants import CONFIG_PATH, PORT


app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_csv():
    # Check if the request contains a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not 'src' in request.form:
        return jsonify({'error': 'No source'}), 400
    elif not request.form['src'] in ['train', 'test']:
        return jsonify({'error': 'Source must be either train or test'}), 400
    else:
        src = request.form['src']
    # Check if the file is a CSV file
    if file and file.filename.endswith('.csv'):
        csv_content = file.read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content), sep=",", header=0)
        try:
            check_data_structure(df)
            id = save_data(df, src, file.filename)
            return jsonify({'message': 'File uploaded successfully', 'ret':{'filename':file.filename, 'id':id, 'src':src}}), 200
        except Exception as e:
            return jsonify({'error': repr(e)}), 400
    else:
        return jsonify({'error': 'Invalid file format. Please upload a CSV file'}), 400

@app.route('/data', methods=['GET'])
def get_all_data():
    if 'src' in request.args.keys():
        try:
            type = request.args['src']
            ids = get_all_data_ids(type)
            return jsonify({'results': ids}), 200
        except Exception as e:
            return jsonify({'error': repr(e)}), 400
    else:
        return jsonify({'error': 'Invalid request, please query either src=train or src=test'}), 400

@app.route('/train', methods=['POST'])
def train():
    # Check if the request contains a file
    if 'config' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['config']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Check if the file is a yaml file
    if file and file.filename.endswith('.yaml'):
        content = file.read().decode('utf-8')
        conf = OmegaConf.create(content)
        try:
            ret = train_model(conf)
            conf.data['encoded_features'] = ret['encoded_features']
            conf.data['train_id_list'] = ret['train_data_ids']
            conf.data['test_id_list'] = ret['test_data_ids']
            model_info = save_model(ret['model'], conf)
            del model_info['weight']
            return jsonify({'model': model_info, 'ret':ret['metrics']}), 200
        except Exception as e:
            return jsonify({'error': repr(e)}), 400 
    else:
        return jsonify({'error': 'Invalid file format. Please upload a yaml file'}), 400

@app.route('/models', methods=['GET'])
def get_models():
    ret = get_all_models()
    return jsonify({'results': ret}), 200

@app.route('/test_models', methods=['POST'])
def test_models():
    # Check if the request contains a file
    if 'config' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['config']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Check if the file is a yaml file
    if file and file.filename.endswith('.yaml'):
        content = file.read().decode('utf-8')
        conf = OmegaConf.create(content)
        try:
            results = []
            y_test = None
            for id in conf.models:
                config = conf.copy()
                config['model'] = {'weights':id}
                ret = test_model(model=None, config=config)
                results.append({'id':id, 'pretrained_config':OmegaConf.to_container(ret['config'], resolve=True), 
                'metrics':ret['metrics'], 'pred':ret['y_pred'].tolist()})
                y_test = ret['y_test'].tolist()
            test_id_list = OmegaConf.to_container(conf.data.test_id_list, resolve=True) if not conf.data.test_id_list is None else None
            return jsonify({'y_test':y_test, 'results': results, 'test_data_ids':test_id_list}), 200
        except Exception as e:
            return jsonify({'error': repr(e)}), 400 
    else:
        return jsonify({'error': 'Invalid file format. Please upload a yaml file'}), 400

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request contains a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Check if the file is a CSV file
    if file and file.filename.endswith('.csv'):
        csv_content = file.read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content), sep=",", header=0)
        try:
            check_data_structure(df)
            conf = OmegaConf.load(CONFIG_PATH)
            model, model_config = load_model(conf.model)
            data = data_to_feature(df, model_config['data'])
            ret = model_predict(model, data)
            return jsonify({'results': ret.tolist()}), 200
        except Exception as e:
            return jsonify({'error': repr(e)}), 400
    else:
        return jsonify({'error': 'Invalid file format. Please upload a CSV file'}), 400

@app.route('/setup_model', methods=['POST'])
def setup_model():
    # Check if the request contains a file
    if not 'model_id' in request.form:
        return jsonify({'error': 'No model is selected'}), 400
    else:
        try:
            model_id = request.form['model_id']
            conf = OmegaConf.load(CONFIG_PATH)
            conf.model.weights = model_id
            model, model_config = load_model(conf.model)
            OmegaConf.save(conf, CONFIG_PATH)
            return jsonify({'message': f'Sucessfully changed model into {model_id}'}), 200
        except Exception as e:
            return jsonify({'error': repr(e)}), 400

if __name__ == '__main__':
    app.run(debug=True,port=PORT)