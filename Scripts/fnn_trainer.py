##########################################################################################
# fnn_trainer.py - example code to create a dump AI engine to be transferred to Neblina
#
#  Created on: Feb. 18th, 2019
#      Author: Omid Sarbishei
#      Project: Motion Analysis and Activity Recognition
#      Company: Motsai
#########################################################################################

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import functools
import operator

from sklearn import model_selection
from sklearn import metrics
from sklearn.svm import LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from array import array
import struct

##############################################################################################################################################################################
##### Parameters ######
##############################################################################################################################################################################
input_filepath = 'D:/Projects/HAR/Recofit/input.csv'	## input CSV file containing the labeled IMU raw data with 9 columns => 0: timestamp, 1: subject ID (1,2,...), 2: ax (raw integer as read from Neblina), 3: ay, 4: az, 5: gx, 6: gy, 7: gz, 8: activity label (1,2,...)
output_filepath = 'D:/Projects/HAR/Recofit/dump.dat' 	## output dump file describing the AI engine
# input_filepath = 'D:/Projects/HAR/WISDM/WISDM_ar_latest/WISDM_at_v2.0/WISDM_ar_v2.0_raw.csv'	## input CSV file containing the labeled IMU raw data with 9 columns => 0: timestamp, 1: subject ID (1,2,...), 2: ax (raw integer as read from Neblina), 3: ay, 4: az, 5: gx, 6: gy, 7: gz, 8: activity label (1,2,...)
# output_filepath = 'D:/Projects/HAR/WISDM/WISDM_ar_latest/WISDM_at_v2.0/dump.dat' 	## output dump file describing the AI engine
nb_input_neurons = 60									## number of features or input neurons excluding the bias neurons => must be equal to ( "nb_histogram_bins" * 6 )
nb_hidden_neurons = 40									## number of hidden neurons excluding the bias neurons
nb_output_neurons = 6									## number of output neurons
nb_histogram_bins = 20									## number of histogram bins per sensor axis
accel_max = 4096										## maximum raw accelerometer value for histogram extraction. 4096 is equivalent to 2g under a 16g full-scale range
gyro_max = 8192											## maximum raw gyroscope value for histogram extraction. 8192 is equivalent to 500dps under a 2000dps full-scale range
hidden_trigger = 0										## Trigger function for the hidden layer => 0: Sigmoid, 1: Tanh, 2: Relu
output_trigger = 3										## Trigger function for the output layer => 0: Sigmoid, 1: Tanh, 2: Relu, 3: Softmax
window_size = 200										## Window size in terms of number of samples => 250 samples is equivalent to a 5s windows at 50Hz sampling rate
window_step = 200										## Sliding window step in terms of number of samples => 10 samples is equivalent to a 200ms sliding window step

nb_epochs = 5000											## number of epochs used for training the neural network
batch = 32 												## batch size for training
##############################################################################################################################################################################


#################################################################################################################################################################
##### Reading the raw input IMU data #####
#################################################################################################################################################################
print( "Reading the input CSV file..." )
df = pd.read_csv( input_filepath )
df.columns = list(range(0,9))
#################################################################################################################################################################


###################################################################################################################################################################################################################################################################################################################################
##### Segmentation and Extracting Histogram Bins => This process can be quite time consuming for large datasets. It is recommended that the extracted histograms "histogram_dataset" together with the labels "histogram_labels" are stored in another CSV file to avoid the re-execution of the histogram feature extraction #####
###################################################################################################################################################################################################################################################################################################################################
print( "Extracting histogram bins. This process can take many minutes for large datasets..." )
indx = window_size
count = 0
hist = np.zeros((1,nb_input_neurons)) # temporary array for histogram bins
histogram_dataset = np.zeros((1400000,nb_input_neurons)) # pre-allocation of the dataset to increase performance
histogram_labels = np.zeros((1400000,1)) # pre-allocation of output labels to increase performance
while( indx < len( df.index ) ): # segmentation and histogram feature extraction
	imu_window = df[indx - window_size:indx]
	label = imu_window[8] - 1
	if max(label) > min(label):
		indx = indx + 1
		continue
	'''ax = round( imu_window[2] * ( 32768 / 16 ) ) # converts the 16g-range ax accelerometer reading to the raw integer format. Remove this line, if ax is already in integer format
	ay = round( imu_window[3] * ( 32768 / 16 ) ) # converts the 16g-range ay accelerometer reading to the raw integer format. Remove this line, if ay is already in integer format
	az = round( imu_window[4] * ( 32768 / 16 ) ) # converts the 16g-range ax accelerometer reading to the raw integer format. Remove this line, if az is already in integer format
	gx = round( imu_window[5] * ( 32768 / 2000 ) ) # converts the 2000dps-range gx gyroscope reading to the raw integer format. Remove this line, if gx is already in integer format
	gy = round( imu_window[6] * ( 32768 / 2000 ) ) # converts the 2000dps-range gy gyroscope reading to the raw integer format. Remove this line, if gy is already in integer format
	gz = round( imu_window[7] * ( 32768 / 2000 ) ) # converts the 2000dps-range gz gyroscope reading to the raw integer format. Remove this line, if gz is already in integer format
	ax = ax.values
	ay = ay.values
	az = az.values
	gx = gx.values
	gy = gy.values
	gz = gz.values'''
	ax = imu_window[2].values # use this format only when the original raw IMU data in the CSV file is in integer format (compatible with Neblina readings)
	ay = imu_window[3].values # use this format only when the original raw IMU data in the CSV file is in integer format (compatible with Neblina readings)
	az = imu_window[4].values # use this format only when the original raw IMU data in the CSV file is in integer format (compatible with Neblina readings)
	gx = imu_window[5].values # use this format only when the original raw IMU data in the CSV file is in integer format (compatible with Neblina readings)
	gy = imu_window[6].values # use this format only when the original raw IMU data in the CSV file is in integer format (compatible with Neblina readings)
	gz = imu_window[7].values # use this format only when the original raw IMU data in the CSV file is in integer format (compatible with Neblina readings)
	
	hist_ax, bin_edges = np.histogram( ax, bins = nb_histogram_bins, range = (-accel_max,accel_max) ) # histograms for ax
	hist_ay, bin_edges = np.histogram( ay, bins = nb_histogram_bins, range = (-accel_max,accel_max) ) # histograms for ay
	hist_az, bin_edges = np.histogram( az, bins = nb_histogram_bins, range = (-accel_max,accel_max) ) # histograms for az
	hist_gx, bin_edges = np.histogram( gx, bins = nb_histogram_bins, range = (-gyro_max,gyro_max) ) # histograms for gx
	hist_gy, bin_edges = np.histogram( gy, bins = nb_histogram_bins, range = (-gyro_max,gyro_max) ) # histograms for gy
	hist_gz, bin_edges = np.histogram( gz, bins = nb_histogram_bins, range = (-gyro_max,gyro_max) ) # histograms for gz

	# append all histogram bins into a single dimensional array with "nb_input_neurons" elements
	hist[0,0:nb_histogram_bins] = hist_ax
	hist[0,nb_histogram_bins:2*nb_histogram_bins] = hist_ay
	hist[0,2*nb_histogram_bins:3*nb_histogram_bins] = hist_az
	hist[0,3*nb_histogram_bins:4*nb_histogram_bins] = hist_gx
	hist[0,4*nb_histogram_bins:5*nb_histogram_bins] = hist_gy
	hist[0,5*nb_histogram_bins:6*nb_histogram_bins] = hist_gz
	hist = hist / window_size # normalize the histogram bins
	histogram_dataset[count,0:nb_input_neurons] = hist
	histogram_labels[count,0] = label.values[0]
	count = count + 1
	indx = indx + window_step
	# if ( count % 10000 == 0 ):
	#	print( count )	
	#	print( max( histogram_labels ) )

histogram_dataset = histogram_dataset[0:count,0:nb_input_neurons]
histogram_labels = histogram_labels[0:count,0]
histogram_all = np.zeros((count,nb_input_neurons+2))
histogram_all[0:count,0:nb_input_neurons] = histogram_dataset
histogram_all[0:count,nb_input_neurons] = 0
histogram_all[0:count,nb_input_neurons+1] = histogram_labels
np.savetxt("D:/Projects/HAR/WISDM/WISDM_ar_latest/WISDM_at_v2.0/histograms_10s.txt", histogram_all, fmt = '%.4f', delimiter=" ")
#################################################################################################################################################################
	
	
################################################################################################################################################################
##### Training the Neural Network #####
################################################################################################################################################################
print( "Training the Neural Network..." )
model = keras.Sequential()
# add one hidden layer
if ( hidden_trigger == 0 ):
	model.add(keras.layers.Dense(nb_hidden_neurons, input_shape=(nb_input_neurons,), activation=tf.nn.sigmoid)) 
elif ( hidden_trigger == 1 ):
	model.add(keras.layers.Dense(nb_hidden_neurons, input_shape=(nb_input_neurons,), activation=tf.nn.tanh))
elif ( hidden_trigger == 2 ):
	model.add(keras.layers.Dense(nb_hidden_neurons, input_shape=(nb_input_neurons,), activation=tf.nn.relu)) 
else:
	model.add(keras.layers.Dense(nb_hidden_neurons, input_shape=(nb_input_neurons,), activation=tf.nn.sigmoid)) # softmax is not accepted for the hidden layer
	
# add the output layer
if ( output_trigger == 0 ):	
	model.add(keras.layers.Dense(nb_output_neurons, activation=tf.nn.sigmoid)) 
elif ( output_trigger == 1 ):
	model.add(keras.layers.Dense(nb_output_neurons, activation=tf.nn.tanh))
elif ( output_trigger == 2 ):
	model.add(keras.layers.Dense(nb_output_neurons, activation=tf.nn.relu))
else:
	model.add(keras.layers.Dense(nb_output_neurons, activation=tf.nn.softmax))
	
model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy']) # build an initial model and select a training algorithm
model.fit( histogram_dataset, histogram_labels, epochs = nb_epochs, batch_size = batch ) # training procedure
#################################################################################################################################################################


#################################################################################################################################################################
##### Neural Network Evaluation #####
#################################################################################################################################################################
print( "Evaluating the model on the trained dataset in terms of F1 measure, precision, recall and overall accuracy..." )
predictions = model.predict( histogram_dataset )
	
val_preds = np.argmax( predictions, axis = -1 )	
precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support( histogram_labels, val_preds ) # evaluating the model for precision, recall, and F1 score using the same training dataset
acc = np.sum( histogram_labels == val_preds ) / len( histogram_labels )
print('Overall accuracy:', acc)
print('Datapoints:', len( histogram_labels ))

print( 'Precision per Class: ' )
print( precisions )
print( 'Recall per Class: ' )
print( recall )
print( 'F1 Score per Class: ' )
print( f1_score )
#################################################################################################################################################################


#################################################################################################################################################################
##### Store the final AI engine to a dump file #####
#################################################################################################################################################################
print( 'Storing the AI engine to a dump file now...' ) 

## get the weights, perform transformations to flatten all the coefficients into a single one-dimensional array, and finally store the equivalent binary array to a dump file
input_neurons = model.get_layer( index = 0 ).get_weights() 
hidden_neurons = model.get_layer( index = 1 ).get_weights() 

input_weights = input_neurons[0].transpose()
input_bias = input_neurons[1].transpose()
hidden_weights = hidden_neurons[0].transpose()
hidden_bias = hidden_neurons[1].transpose()

hidden_all_weights = hidden_weights[0]
hidden_all_weights = np.append( hidden_all_weights, hidden_bias[0] )
for i in range(1,nb_output_neurons):
	hidden_all_weights = np.append( hidden_all_weights, hidden_weights[i] )
	hidden_all_weights = np.append( hidden_all_weights, hidden_bias[i] )

input_all_weights = input_weights[0]
input_all_weights = np.append( input_all_weights, input_bias[0] )
for i in range( 1, nb_hidden_neurons ):
	input_all_weights = np.append( input_all_weights, input_weights[i] )
	input_all_weights = np.append( input_all_weights, input_bias[i] )
	
all_weights = np.append( input_all_weights, hidden_all_weights )
output_file = open( output_filepath, 'wb' )
float_array = array( 'f', all_weights ) # single precision floating-point format is used

axis = 0
threshold = 0
hdr = struct.pack( '<BBHBBBBBHHBH', hidden_trigger, output_trigger, nb_input_neurons, nb_hidden_neurons, nb_output_neurons, nb_histogram_bins, window_size, window_step, accel_max, gyro_max, axis, threshold )	
output_file.write( hdr ) # write the header section
float_array.tofile( output_file ) # write the data section (FNN weights)
output_file.close()
#################################################################################################################################################################
