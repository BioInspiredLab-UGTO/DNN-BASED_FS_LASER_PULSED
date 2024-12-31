# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 12:20:49 2024

@author: jrpar
"""

import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os
import csv
import time
###############################################################################
                            #GETTING DATA
###############################################################################
"""
The first 54 data correspond to the real data, and the remaining data
correspond to the interpolated data.
"""
f_i = './DB-Fluencia-Tiempo-Train.csv'
f_i_2 = './DB-Fluencia-Tiempo-Test.csv'

Datatrain = pd.read_csv(f_i)
Datatest = pd.read_csv(f_i_2)

Train = Datatrain.values
Test =  Datatest.values

#Import Columns
X_train_sp=Train[:,1:3]    #Columna 1 y 2
Y_train=Train[:,0]  #Columna 0

X_test_sp = Test[:,1:3]  
Y_test = Test[:,0]

mms = MinMaxScaler()
X_train = mms.fit_transform(X_train_sp)
X_test = mms.transform(X_test_sp)

###############################################################################
                           #MODEL SELECTION
###############################################################################

def build_model(optimizer,N_model):
    
    if N_model == 1:
    	model= keras.Sequential([
    		keras.layers.Dense(32, activation='relu', input_shape=[2]),
    		keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
    		keras.layers.Dense(1)		
    	]) 
    	model.compile(loss='mse', optimizer=optimizer,metrics=['mae', 'mse'])
        
    elif N_model == 2:
    	model= keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=[2]),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1)	
    	]) 
    	model.compile(loss='mse', optimizer=optimizer,metrics=['mae', 'mse'])
    elif N_model == 3:
    	model= keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=[2]),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1)	
    	])
    	model.compile(loss='mse', optimizer=optimizer,metrics=['mae', 'mse'])
    elif N_model ==  4:
    	model= keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=[2]),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1)	
    	]) 
    	model.compile(loss='mse', optimizer=optimizer,metrics=['mae', 'mse'])
    elif N_model ==  5:
    	model= keras.Sequential([
            keras.layers.Dense(256, activation='relu', input_shape=[2]),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(54, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1)	
    	])
    	model.compile(loss='mse', optimizer=optimizer,metrics=['mae', 'mse'])
    elif N_model ==  6:
    	model= keras.Sequential([
            keras.layers.Dense(350, activation='relu', input_shape=[2]),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1)	
    	]) 
    	model.compile(loss='mse', optimizer=optimizer,metrics=['mae', 'mse'])
    elif N_model ==  7:
    	model= keras.Sequential([
            keras.layers.Dense(700, activation='relu', input_shape=[2]),
            keras.layers.Dense(525, activation='relu'),
            keras.layers.Dense(350, activation='relu'),
            keras.layers.Dense(175, activation='relu'),
            keras.layers.Dense(88, activation='relu'),
            keras.layers.Dense(36, activation='relu'),
            keras.layers.Dense(1)	
    	]) 
    	model.compile(loss='mse', optimizer=optimizer,metrics=['mae', 'mse'])
    else:
       	model= keras.Sequential([
               keras.layers.Dense(700, activation='relu', input_shape=[2]),
               keras.layers.Dense(525, activation='relu'),
               keras.layers.Dense(350, activation='relu'),
               keras.layers.Dense(175, activation='relu'),
               keras.layers.Dense(88, activation='relu'),
               keras.layers.Dense(36, activation='relu'),
               keras.layers.Dense(18, activation='relu'),
               keras.layers.Dense(1)	
       	])
        model.compile(loss='mse', optimizer=optimizer,metrics=['mae', 'mse'])
        
    return model
#return model
###############################################################################
                                 #TRAINIG
###############################################################################        
 
###############################################################################
                                 #FIRST TRAINIG
###############################################################################
start = time.time()
for n_model in range(8):
    n_model = n_model + 1
    MSE_Train1 = []
    MAE_Train1 = []
    MSE_Test1 = []
    MAE_Test1 = []
    
    MSE_Train2 = []
    MAE_Train2 = []
    MSE_Test2 = []
    MAE_Test2 = []
    
    MSE_Train3 = []
    MAE_Train3 = []
    MSE_Test3 = []
    MAE_Test3 = []
    for n_train in  range(54):
        n_train = n_train +1        
        #optimizer selection 
        optimizer = tf.keras.optimizers.AdamW(learning_rate=0.01)
        #Build model
        model = build_model(optimizer,n_model)
        
        #keras.utils.plot_model(model, 'model.png', show_shapes=True)
        #model.summary()
        
        history = model.fit(X_train,Y_train,epochs=200,validation_split = 0.1,verbose=3)
        hist = pd.DataFrame(history.history)
        
        ###############################################################################
                                        #PREDICTION
        ###############################################################################        
        predictions = model.predict(X_test)
        mse = mean_squared_error(Y_test, predictions)
        print("MSE:", mse)
        mae = mean_absolute_error(Y_test, predictions)
        print("MAE:", mae)
        MSE_Train1.append(round(history.history['mse'][-1],4))
        MAE_Train1.append(round(history.history['mae'][-1],4))
        
        MSE_Test1.append(round(mse,4))
        MAE_Test1.append(round(mae,4))
        
        ###############################################################################
                                        #SAVE WEIGHTS
        ###############################################################################
        num_train = 'Train_1'
        target_dir = f'./WEIGHTS{num_train}/'
        os.makedirs(target_dir, exist_ok=True)
        model.save_weights(target_dir + num_train + '.weights.h5')	#pesos
        
        ######################################################R#########################
                                    #SECOND TRAIN
        ###############################################################################
        
        #We lower the optimizer training rate
        optimizer2 = tf.keras.optimizers.AdamW(learning_rate=0.001)
        
        # building the model again from cero
        model2 = build_model(optimizer2,n_model)
        
        #We assign the weights obtained from the previous training
        model2.load_weights(target_dir + num_train + '.weights.h5')
        
        
        history2 = model2.fit(X_train,Y_train,epochs=100,validation_split = 0.1,verbose=3)
        
        predictions = model2.predict(X_test)
        mse = mean_squared_error(Y_test, predictions)
        #print("MSE:", mse)
        mae = mean_absolute_error(Y_test, predictions)
        #print("MAE:", mae)
        
        MSE_Train2.append(round(history2.history['mse'][-1],4))
        MAE_Train2.append(round(history2.history['mae'][-1],4))
        
        MSE_Test2.append(round(mse,4))
        MAE_Test2.append(round(mae,4))
        
        num_train = 'Train_2'
        target_dir = f'./WEIGHTS{num_train}/'
        os.makedirs(target_dir, exist_ok=True)
        model2.save_weights(target_dir + num_train + '.weights.h5')	
        ###############################################################################
                                    #THIRD TRAIN
        ###############################################################################
        
        optimizer3 = tf.keras.optimizers.AdamW(learning_rate=0.00001)
        
        model3 = build_model(optimizer3,n_model)
        model3.load_weights(target_dir + num_train + '.weights.h5')
        history3 = model3.fit(X_train,Y_train,epochs=100,validation_split = 0.1,verbose=3)
        
        predictions = model3.predict(X_test)
        mse = mean_squared_error(Y_test, predictions)
        print("MSE:", mse)
        mae = mean_absolute_error(Y_test, predictions)
        print("MAE:", mae)
        
        MSE_Train3.append(round(history3.history['mse'][-1],4))
        MAE_Train3.append(round(history3.history['mae'][-1],4))
        
        MSE_Test3.append(round(mse,4))
        MAE_Test3.append(round(mae,4))
        
    file_csv = f"ERRORS_MODEL{n_model}.csv"
    
    # Write the lists to the file
    with open(file_csv, mode='w', newline='') as file:
        wr = csv.writer(file)
        
        
        wr.writerow(["MSE_TN1", "MAE_TN1", "MSE_TS1", "MAE_TS1",
                           "MSE_TN2", "MAE_TN2", "MSE_TS2", "MAE_TS2",
                           "MSE_TN3", "MAE_TRAIN3", "MSE_TS3", "MAE_TS3",])
        
        
        for row in zip(MSE_Train1, MAE_Train1, MSE_Test1, MAE_Test1, MSE_Train2, MAE_Train2, MSE_Test2, MAE_Test2 ,MSE_Train3, MAE_Train3, MSE_Test3, MAE_Test3):
            wr.writerow(row)
    print(f"Archivo '{file_csv}' creado con éxito.")
end_t = time.time()

tolal_time = end_t - start

print(f"The training took {tolal_time:.4f} seconds.")