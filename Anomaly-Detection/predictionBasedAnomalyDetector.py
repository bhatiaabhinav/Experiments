from anomalyDetector import AnomalyDetector
import utils
import numpy as np
import random
import sys
import math
import csv
import os.path
import keras.models
from matplotlib import pyplot as plt

def PredictionBasedAnomalyDetector(AnomalyDetector):
	
	def __init__(self, config):
		super(AnomalyDetector, self).__init__(config)
		self.__stream = []
		self.__stream_prediction_error = []
		self.__stream_anomalies = []
		self.__stream_predicted = [np.zeros([1])]
		
	def create_new(self):
		print('Creating new model ... ')
		self.__model = __createModelCNN()

	def __createModelCNN(input_length):
		from keras.models import Sequential
		from keras.layers import Dense, Activation, Dropout, Convolution1D, MaxPooling1D, Flatten
		model = Sequential()
		model.add(Convolution1D(input_shape = (input_length,1), nb_filter=8,filter_length=4,border_mode='valid',activation='relu',subsample_length=1))
		model.add(MaxPooling1D(pool_length=2))
		model.add(Convolution1D(nb_filter=8,filter_length=8,border_mode='valid',activation='relu',subsample_length=1))
		model.add(MaxPooling1D(pool_length=2))
		model.add(Dropout(0.5))
		model.add(Flatten())
		model.add(Dense(256))
		model.add(Dropout(0.25))
		model.add(Activation('relu'))
		model.add(Dense(1))
		model.add(Activation('linear'))
		model.compile(loss='mean_squared_error', optimizer='adam')
		return model

	def __getMiniBatch(data, batch_size, input_length):
	    x_batch = []
	    y_batch = []
	    if len(data) >= input_length + 1:
	        for i in range(batch_size):
	            start_index = random.randint(0, len(data)-input_length-1)
	            end_index = start_index + input_length - 1
	            target_index = end_index + 1
	            x = data[start_index:end_index+1]
	            y = data[target_index:target_index+1]
	            # y_onehot = one_hot(y[0], 256)
	            x_batch.append(x)
	            y_batch.append(y)
	    x_batch = np.array(x_batch).reshape(batch_size, input_length, 1)
	    y_batch = np.array(y_batch).reshape(batch_size, 1)
	    return x_batch, y_batch

	def train(self, number_of_steps):
		# learn from data so far. Do number_of_steps SGD step:
		for i in range(number_of_steps):
			if (len(self.__stream) > self.__config.model_input_length + 1 + 100):
				x_train, y_train = __getMiniBatch(self.__stream, self.__config.training_batch_size, self.__config.model_input_length)
				self.__model.train_on_batch(x_train, y_train)

	def save(self, dir):
		self.__model.save(dir + 'model' if dir.endswith('/') else dir + '/model')

	def load(self, dir):
		print("Loading model ... ")
		self.__model = keras.models.load_model(dir + 'model' if dir.endswith('/') else dir + '/model')


	def consume(self, y):
		
		# transform and smooth the input
		log_y = utils.log_transform(y, self.__config.max_input_value)
		smooth_log_y = (1 - self.__config.input_smoothing) * log_y + self.__config.input_smoothing * (0 if len(self.__stream) == 0 else self.__stream[-1])
		self.__stream.append(smooth_log_y)

		# calculate prediction error
		self.__stream_prediction_error.append(abs(self.__stream[-1] - self.__stream_predicted[-1]))

		# check for anomalies:
		anomaly_results = self.__check_anomaly()
		self.__stream_anomalies.append(1 if anomaly_results.anomaly_exists else 0)

    	# make new prediction:
		predicted_next_point = self.__predict_next_point()
		self.__stream_predicted.append(predicted_next_point)

		return anomaly_results
		
	def __predict_next_point(self):
		# predict from data so far (partially from actual data points and partially by building on predictions):
	    if len(self.__stream) >= self.__config.model_input_length:
	        if len(__stream) >  self.__config.model_input_length + self.__config.prediction_latency:
	            no_predictions_availble = self.__config.prediction_latency
	        else:
	            no_predictions_availble = len(__stream) - self.__config.model_input_length

	        if no_predictions_availble > 0:
	            input = __stream[-self.__config.model_input_length : -no_predictions_availble]
	            input.extend(self.__stream_predicted[-no_predictions_availble : ])
	        else:
	            input = __stream

	        # print(input)
	        input_np = np.array(input).reshape(1, len(input), 1)
	        #print(input_np)
	        predicted = self.__model.predict(input_np, batch_size = 1)[0]
	    else:
	        predicted = 0

	    return predicted


	def __check_anomaly(self):
		results = {}
		results.anomaly_exists = False
		results.anomaly_started = False
		results.anomaly_endend = False
		results.current_anomaly_duration = 0
		results.current_anomaly_confidence = 0
		results.notes = 'Nothing interesting right now'
		return results

	def show_plots(self):
		n = 30000
		plt.plot(utils.downsample(self.__stream, n), color = 'green')
		plt.plot(utils.downsample(self.__stream_predicted, n), color = 'blue', alpha = 0.9)
		plt.plot(utils.downsample(self.__stream_prediction_error, n), color = 'red', alpha = 0.9)
		plt.plot(utils.downsample(self.__stream_anomalies, n), color = 'black', alpha = 0.6)
		plt.show()