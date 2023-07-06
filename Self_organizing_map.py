# Self-organizing map
# -------------------

# Setup
import numpy as np
import matplotlib as plt
import pandas as pd
import os
from minisom import MiniSom # Package for SOM
from pylab import bone, pcolor, colorbar, plot, show

os.chdir("C:/Users/Eoin/Documents/Important/Work/Portfolio/Deep learning/A-Z/Part 8 - Deep Learning/Self_Organizing_Maps")

data = pd.read_csv('Credit_Card_Applications.csv')

x = data.iloc[:, :-1].values # Applicant information
y = data.iloc[:, :1].values # Application approved/denied

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))  # Normalize features
x = sc.fit_transform(x) # Apply scaler from last line to x

# Train SOM
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5) # Rows, columns, variables, neuron radius, update weight influence
som.random_weights_init(x)
som.train_random(data = x, num_iteration = 100) # Training epochs

# Visualize results
bone() # Initialize map (empty)
pcolor(som.distance_map().T) # Matrix of distances between nodes, with distance represented as color on map
colorbar() # Map legend. Low distance = low color
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(x):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5, # Puts marker of winning node at centre of square
         markers[y[i]], # i (customer in for loop) marked with o or s
         markeredgecolor = colors[y[i]], # Colours customer red or green
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()


markers = ['o', 's']
colors = ['r', 'g']
for i in range(len(y)):
    w = som.winner(x[i])
    plot(w[0] + 0.5,
         w[1] + 0.5,  # Puts marker of winning node at centre of square
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)
show()

# Get list of customers with high potential of fraud
mappings = som.win_map(x)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0) # List of relevant customers from map coordinates. Vertical axis

