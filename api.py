from flask import Flask, jsonify, request
import joblib
import lightgbm as ltb

app = Flask(__name__)


@app.route('/api/v1/prediction', methods=['POST'])
def get_prediction():
    modelname = '/Users/situ/Desktop/data-engineering-exercise/lgbr_cars.model'

    data = request.get_json()
    print(data.get('parameter_list'))

    loaded_model = joblib.load(modelname)
    value = loaded_model.predict([data.get('parameter_list')])
    print(value)
    return jsonify({'value': round(value[0], 2)})


if __name__ == '__main__':
    app.run(debug=True)