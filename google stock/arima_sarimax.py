# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')

training_set = dataset_train.iloc[:, 1:2].values
# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(dataset_train[i-60:i, 0])
    y_train.append(dataset_train[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

print(X_train)
print(y_train)








