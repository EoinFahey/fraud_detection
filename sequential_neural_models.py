import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

os.chdir("C:/Users/Eoin/Documents/Important/Work/Portfolio/Deep learning/A-Z/Part 8 - Deep Learning/Self_Organizing_Maps")
dataset = pd.read_csv('Credit_Card_Applications.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
x = sc.fit_transform(x)

from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(x)
som.train_random(data = x, num_iteration = 100)

from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(x):
    w = som.winner(x)
    plot(w[0]+ 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

mappings = som.win_map(x)
frauds = np.concatenate((mappings[(5, 3)], mappings[(8, 3)]), axis = 0)
frauds = sc.inverse_transform(frauds)


# Create dependent reference variable
customers = dataset.iloc[:, :-1].values # Array of all customers
is_fraud = np.zeros(len(dataset)) # Create vector for all of the customers, by default set to 0 (no fraud)

for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds: # Check if customer ID is in list of frauds
        is_fraud[i] = 1 # Give frauds a distinct value of 1 for training purposes
        
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15)) # Slight changes
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 2) # Few observations and few features = no need for excessive training

y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1) # First column for customer IDs, second column for probabilities
y_pred = y_pred[y_pred[:, 1].argsort()] # Sort probabilities from lowest to highest