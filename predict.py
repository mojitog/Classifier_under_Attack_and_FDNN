import joblib
import numpy as np

model_path = "model/model.pkl"

# Load the trained model
model = joblib.load(model_path)

def predict_risk(input_data):
    # Make predictions
    predictions_binary = model.predict(input_data)
    probabilities = model.predict_proba(input_data)[:, 1]

    predictions = ["Good" if pred == 1 else "Bad" for pred in predictions_binary]
    
    return predictions, probabilities

#test_data = [67,66,5,24,9,0,0,100,-7,7,8,9,4,44,0,4,4,53,66,4,2,1,86]
#preds, probs = predict([test_data])
#print("Predictions:", preds)
#print("Probabilities:", probs)
