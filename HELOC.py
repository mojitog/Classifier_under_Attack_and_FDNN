# import dependencies
import sys
import subprocess

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, recall_score, precision_score

from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeClassifier
from art.attacks.evasion import DecisionTreeAttack

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import joblib

dataset = "data/heloc_dataset.csv"
Model_path = "model/model.pkl"


# function to split dataset into train and test 
def data_spliter(dataset):
    heloc_df = pd.read_csv(dataset)
    y = heloc_df.iloc[:,0]
    x = heloc_df.iloc[:,1:]
    #Unnecessary preprocessing step!?
    yy = []
    for row in range(len(y)):
        if y[row] == "Bad":
            yy.append(0)
        elif y[row] == "Good":
            yy.append(1)
    x_train, x_test, y_train, y_test = train_test_split(x, yy, test_size=0.2, random_state=23)
    return x_train, x_test, y_train, y_test


# random forrest classifier
def random_forrest(dataset):
    x_train, x_test, y_train, y_test = data_spliter(dataset)
    rf_model = RandomForestClassifier(n_estimators=200, random_state=23)
    rf_model.fit(x_train, y_train)
    y_pred = rf_model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    print("Random Forest")
    print("Accuracy is equal to ", acc)
    print("Recall is equal to ", recall)
    print("Precision is equal to ", precision)
    y_prob = rf_model.predict_proba(x_test)[:, 1]
    ROC_AUC = roc_auc_score(y_test, y_prob)
    print ("ROC_AUC is equal to ", ROC_AUC,"\n")

    joblib.dump(rf_model, Model_path)


def decision_tree(dataset):
    x_train, x_test, y_train, y_test = data_spliter(dataset)
    dec_tree_mod = DecisionTreeClassifier(random_state=23)
    dec_tree_mod.fit(x_train, y_train)
    y_pred = dec_tree_mod.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print("Decision Tree")
    print("Accuracy is equal to ", acc)
    y_prob = dec_tree_mod.predict_proba(x_test)[:, 1]
    ROC_AUC = roc_auc_score(y_test, y_prob)
    print ("ROC_AUC is equal to ", ROC_AUC,"\n")


#adversial random forest 
def trees_under_adversial_attack(dataset_path):
    x_train, x_test, y_train, y_test = data_spliter(dataset_path)

    rf_model = RandomForestClassifier(n_estimators=200, random_state=23)
    rf_model.fit(x_train, y_train)

    x_train_np = x_train.to_numpy(dtype=np.float32)
    x_test_np  = x_test.to_numpy(dtype=np.float32)

    y_train_np = np.array(y_train) # or np.array(y_train)
    y_test_np  = np.array(y_test)

    dt_model = DecisionTreeClassifier(random_state=23)
    dt_model.fit(x_train_np, y_train_np)
    y_pred = dt_model.predict(x_test_np)
    acc = accuracy_score(y_test_np, y_pred)
    print("Decision Tree before attack")
    print("Accuracy is equal to ", acc)

    dt_art_wrap = ScikitlearnDecisionTreeClassifier(model=dt_model)
    attack = DecisionTreeAttack(classifier=dt_art_wrap, offset=0.001, verbose=True)
    x_test_adv = attack.generate(x=x_test_np)
    y_pred_adv = dt_model.predict(x_test_adv)
    acc_adv = accuracy_score(y_test_np, y_pred_adv)
    print("Decision Tree after attack")
    print("Accuracy on adversial example are equal to ", acc_adv )
    print("Prediction Difference is : ", np.mean(y_pred - y_pred_adv))


#Simple DNN model (secondary by pytorch)
class simpleDNN(nn.Module):
    def __init__(self):
        super(simpleDNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(23, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.network(x)


with open("outputs.txt", 'w') as output:
    sys.stdout = output
    random_forrest(dataset)
    
    decision_tree(dataset)
    trees_under_adversial_attack(dataset)
    #{ print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout) }
    #print(torch.cuda.is_available())

    # prepare data for DNN
    x_train, x_test, y_train, y_test = data_spliter(dataset)
    x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    test_data = TensorDataset(x_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)
        
    DNNmodel = simpleDNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(DNNmodel.parameters(), lr=0.001)

    #training loop
    num_epoch = 100
    for epoch in range(num_epoch):
        DNNmodel.train()
        running_loss = 0.0
        total = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()           
            outputs = DNNmodel(batch_x)    
            loss = criterion(outputs, batch_y)     
            loss.backward()              
            optimizer.step()             
            running_loss += loss.item() * batch_x.size(0)
            total += batch_x.size(0)
        train_loss = running_loss / total
        if (epoch % 10) == 0:
            print("Training loss is: ", train_loss)


    #evaluation loop
    DNNmodel.eval()
    with torch.inference_mode():
        running_loss = 0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []
        for batch_x, batch_y in test_loader:
            output = DNNmodel(batch_x)
            loss = criterion(output, batch_y)
            running_loss += loss.item() * batch_x.size(0)
            total += batch_x.size(0)
            preds = torch.argmax(output, dim=1)
            correct += (preds == batch_y).sum().item()

            probs = output.softmax(dim=1)
            p1 = probs[:, 1].numpy()
            all_probs.append(p1)
            all_labels.append(batch_y.numpy())
        test_loss = running_loss / total
        accuracy = correct / total
        print(f"DNN Test Accuracy: {accuracy}, test loss: {test_loss}")
        all_probs = np.concatenate(all_probs)
        all_labels =  np.concatenate(all_labels)
        print(f"all labels {all_labels}, all probs {all_probs}")
        AUC_ROC = roc_auc_score(all_labels,all_probs)
        print(f"DNN Test AUC_ROC: {AUC_ROC}")
