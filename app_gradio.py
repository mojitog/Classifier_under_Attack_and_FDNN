import gradio as gr
from predict import predict_risk
import pandas as pd

dataset = "data/heloc_dataset.csv"

heloc_df = pd.read_csv(dataset)
features = heloc_df.columns.tolist() 

def gradio_predict(*input_data):
    feature_array = list(input_data)
    predictions, probabilities = predict_risk([feature_array])
    return predictions[0], probabilities[0]

input_components = [gr.Number(label=feature) for feature in features[1:]]
#print("input componenets: ",input_components)
output_components = [
    gr.Text(label="Prediction"),
    gr.Number(label="Probability of Good Risk")
]

interface = gr.Interface(
    fn=gradio_predict,
    inputs=input_components,
    outputs=output_components,
    title="HELOC Risk Prediction",
    description="Predict whether a Home Equity Line of Credit (HELOC) application is 'Good' or 'Bad' risk based on input features."
)

if __name__ == "__main__":
    interface.launch()