from flask import Flask, request, jsonify
from predict import predict_risk

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    #input_data = data['input_data']  # Expecting a list of lists for multiple samples

    predictions, probabilities = predict_risk([input_data])

    response = {
        'predictions': predictions,
        'probabilities': probabilities.tolist()
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)