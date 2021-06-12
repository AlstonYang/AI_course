#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
tf.compat.v1.keras.backend.set_session(sess)


# In[3]:


def read(path):
    return pd.read_csv(path)


# In[4]:


def buildTrain(train, pastWeek=4, futureWeek=1, defaultWeek=1):
    X_train, Y_train = [], []
    for i in range(train.shape[0]-futureWeek-pastWeek):
        X = np.array(train.iloc[i:i+defaultWeek])
        X = np.append(X,train["CCSP"].iloc[i+defaultWeek:i+pastWeek])
        X_train.append(X.reshape(X.size))
        Y_train.append(np.array(train.iloc[i+pastWeek:i+pastWeek+futureWeek]["CCSP"]))
    return np.array(X_train), np.array(Y_train)


# In[5]:


sc = MinMaxScaler(feature_range = (0, 1))


# In[6]:


def get_data():
    
    ## Read weekly copper price data
    path = "WeeklyFinalData.csv"
    data = read(path)
    
    date = data["Date"]
    data.drop("Date", axis=1, inplace=True)
    
    ## Add time lag (pastWeek=4, futureWeek=1)
    x_data, y_data = buildTrain(data)
    
    ## Data split
    x_train = x_data[0:int(x_data.shape[0]*0.8)]
    x_test = x_data[int(x_data.shape[0]*0.8):]
    
    y_train = y_data[0:int(y_data.shape[0]*0.8)]
    y_test = y_data[int(y_data.shape[0]*0.8):]
    
    ## Normalize
    x_train_scaled = sc.fit_transform(x_train)
    y_train_scaled = sc.fit_transform(y_train)
    
    
    return (x_train_scaled, y_train_scaled)


# In[7]:


class network():
    
    def __init__(self, nb_neuro):
        
        x_train_scaled, y_train_scaled = get_data()
        
        # Stop criteria - threshold
        self.threshold_for_error = 1
        self.threshold_for_lr = 1e-2
        
        # Input data
        self.x = tf.convert_to_tensor(x_train_scaled, np.float32)
        self.y = tf.convert_to_tensor(y_train_scaled, np.float32)
        
        # Learning rate
        self.learning_rate = 1e-2
        
        # Optimizer
        self.optimizer = tf.optimizers.SGD(self.learning_rate)
        
         # Hidden layer I
        self.n_neurons_in_h1 = nb_neuro
        self.W1 = tf.Variable(tf.random.truncated_normal([self.x.shape[1], self.n_neurons_in_h1], mean=0, stddev=1), name='weights1')
        self.b1 = tf.Variable(tf.random.truncated_normal([self.n_neurons_in_h1], mean=0, stddev=1), name = "biases1")

        # Output layer
        self.Wo = tf.Variable(tf.random.truncated_normal([self.n_neurons_in_h1, self.y.shape[1]], mean=0, stddev=1), name='weightsOut')
        self.bo = tf.Variable(tf.random.truncated_normal([self.y.shape[1]], mean=0, stddev=1), name='biasesOut')

        # Whether the network is acceptable
        self.acceptable = False
        
        # forward operation
    def forward(self):
        with tf.GradientTape() as tape:

            y1 = tf.nn.relu((tf.matmul(self.x, self.W1)+self.b1), name='activationLayer1')
            yo = (tf.matmul(y1,self.Wo)+self.bo)

            # performance measure
            diff = yo-self.y
            loss = tf.reduce_mean(diff**2)

        return(yo, loss, tape)

    # backward operation
    def backward(self,tape,loss):

        gradients = tape.gradient(loss, [self.W1, self.Wo, self.b1, self.bo])
        self.optimizer.apply_gradients(zip(gradients, [self.W1, self.Wo, self.b1, self.bo]))



