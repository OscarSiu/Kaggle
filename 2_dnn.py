#Deep Neural Network

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#Set up plotting
plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Read CSV
import pandas as pd

football_team = pd.read_csv('Football_teams.csv')
print(football_team.head())
#print(football_team.shape)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Define a linear model
model =  keras.Sequential([
    #Hidden layers
    layers.Dense(units=512, activation='relu', input_shape=[9]), # input_shape = dimension of input
    layers.Dense(units=512, activation='relu'),
    layers.Dense(units=512, activation='relu'),
    
    #linear output layer
    layers.Dense(units=1) #units = number of outputs
])
#print(model.weights)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Untrained Linear Model
x = tf.linspace(-1.0, 1.0, 100)
y = model.predict(x)
# y = activation_layer(x)

plt.figure(dpi=100)
plt.plot(x, y, 'k')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("Input: x")
plt.ylabel("Target y")
w, b = model.weights # you could also use model.get_weights() here
plt.title("Weight: {:0.2f}\nBias: {:0.2f}".format(w[0][0], b[0]))
plt.show()
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
activation_layer = layers.Activation('swish')     #Activation layer: 'relu', 'elu', 'selu', 'swish'

x = tf.linspace(-5.0, 5.0, 100)
y = activation_layer(x) # once created, a layer is callable just like a function

plt.figure(dpi=100)
plt.plot(x, y)
plt.xlim(-5, 5)
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()
