This project is designed and implemented to practice multiple Data Science and Machine Learning milestones, which the noticable points out of it is:

For the demo you can check out the hugging face space below:
https://huggingface.co/spaces/MojitoKap/Risk_Estimator

1- Implementing and banchmarking different machine learning model on dataset. For reaching this purpose, I used a decision Tree, a random Forest, and a neural network based model and compare their accuracy and ROC_AUC, to find the best working model over the HELOC dataset to predict risk performance.

2- Deployment and engineering, the practice of model serialization, model serving, containerizing, and running everything on a cloud server to have it working automatically. After serialization of the best working model (which in our case it's the random forest), I've served API to the model by Flask, this API makes the model response with the predict with the probability of the prediction. Moreover I wrote a script with gradio library for demo the model. Then I containerized the whole application with docker with two available ports, first to access API and the second is to use it with interface. In the end, I upload the model on a Hugging Face space as cloud working production.   

3- Robustness and security check against adversial attack, and the result of it over the predicting model. In the development phase of the project, I applied an adversial attack called peparnot attack to the test dataset and show it's impact on the model.

Quick Start

Requirments: Python 3, docker

Clone the repo.

Go to the directory, create a python virtual environment.(This can be optional)

Then install requirements by following terminal command:
pip install requirements.txt

You can first run the HELOC.py to see the output of training models, like accuracy and ROC_AUC by following command:
python HELOC.py

For check the API and Interface, you can use following commands to run their specific scripts:
python app_flask.py
python app_gradio.py

To build the dockerized version use the following command:
docker build -t HELOC .
This will build you a docker container based on the docker file in the project.

To run the docker use the following command:
docker run -p 7860:7860 HELOC

Now you can check the interface on localhost port 7860.

Thank you for your attention!
