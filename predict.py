# ## Load the model


import pickle
import numpy as np

from flask import Flask, request, jsonify

def predict_single(person, dv, model):
    #Weight class
    weight_class_labels = {
        1: 'Underweight',
        2: 'Normal',
        3: 'Overweight',
        4: 'Obesity'
    }

    X = dv.transform([person])
    y_pred = model.predict(X)

    result = {'The weight class of this person is' : weight_class_labels[y_pred[0]]}

    return result



model_file = 'weight-class-model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)
    

app = Flask('weight_class')

@app.route('/predict', methods=['POST'])
def predict():
    person = request.get_json()

    prediction = predict_single(person, dv, model)

    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)   

# print(f'Input: {person}')
# print(f'The weight class is: {weight_class_labels[y_pred[0]]}')
