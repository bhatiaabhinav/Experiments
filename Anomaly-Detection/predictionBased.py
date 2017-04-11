import numpy as np
import random
import sys
import math
import csv
import os.path
import keras.models
import dataLoader
from matplotlib import pyplot as plt


dummyData = False

def smooth(data, window_length):
    smoothed_data = []
    for i in range(len(data)):
        if i >= window_length/2 and i <= len(data) - window_length/2 - 1:
            sum = np.zeros((1,1))
            for j in range(window_length):
                sum += data[i - int(window_length/2) + j]
            sum = float(sum)/window_length
            smoothed_data.append(sum)
    return np.array(smoothed_data).reshape(len(smoothed_data), 1)

def one_hot(y, length):
    y_onehot = np.zeros([length], dtype='uint8')
    y_onehot[y] = 1
    return y_onehot

def getFullBatch(data, input_length):
    x_batch = []
    y_batch = []
    if len(data) >= input_length + 1:
        for i in range(len(data)-input_length):
            start_index = i
            end_index = start_index + input_length - 1
            target_index = end_index + 1
            x = data[start_index:end_index+1]
            y = data[target_index:target_index+1]
            x_batch.append(x)
            y_batch.append(y)
    x_batch = np.array(x_batch).reshape(len(data)-input_length, input_length, 1)
    y_batch = np.array(y_batch).reshape(len(data)-input_length, 1)
    return x_batch, y_batch

def getMiniBatch(data, batch_size, input_length):
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

def createModelMLP(input_length, output_length):
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout, Flatten
    model = Sequential()
    model.add(Flatten(input_shape=(input_length,1)))
    model.add(Dense(units=500, input_dim=input_length))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(units=500))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Dense(units=output_length))
    model.add(Activation('linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def createModelCNN(input_length, output_length):
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

def createModelCnnDilated(input_length, output_length):
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout, Convolution1D, MaxPooling1D, Flatten
    model = Sequential()
    model.add(Convolution1D(input_shape = (input_length,1), nb_filter=32, filter_length=4 ,border_mode='valid',activation='relu'))
    model.add(MaxPooling1D(pool_length=4))
    model.add(Convolution1D(nb_filter=32, filter_length=4, border_mode='valid',activation='relu'))
    # model.add(Convolution1D(input_shape = (input_length,1), nb_filter=16, filter_length=2, dilation_rate=64, border_mode='valid',activation='relu',subsample_length=1))
    # model.add(Convolution1D(input_shape = (input_length,1), nb_filter=32, filter_length=2, dilation_rate=8, border_mode='valid',activation='relu',subsample_length=1))
    model.add(MaxPooling1D(pool_length=4))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def extended_predict(model, x, prediction_length):
    current_x = list(x)
    predicted_y = []
    for i in range(prediction_length):
        y = model.predict(np.array(current_x).reshape(1,len(x),1), batch_size = 1)[0]
        predicted_y.append(y)
        current_x.pop(0)
        current_x.append(y)
    return predicted_y

model_input_length = 1440
stream_prediction_latency = 60
stream_data = []
raw_stream_data = []
stream_predicted = [np.zeros([1])]
stream_error = []
stream_anomalies = []
stream_anomalies_av = []
stream_anomalies_ul = []
stream_anomalies_ll = []
# the average of stream data over last model_input_length points
av_x = 0
max_x = 2000
model_version = "log-so-opm-v3"
#model_version = "sin"
model_name = "model-{0}".format(model_version)
skipTraining = True
anomaly_precision = 6

if os.path.isfile(model_name):
    print("Loading \model...")
    stream_model = keras.models.load_model(model_name)
else:
    print("Creating model")
    stream_model = createModelCNN(model_input_length, 1)

def sign(x):
    if x < 0:
        return -1
    if x > 0:
        return 1
    return 0

def transformInput(x, max_x, moving_av_x):
    u = 255
    x_clipped = min(x, max_x - 1) # between 0 and max_x (max_x excluded)
    # fx = 2 * (x_clipped / 500) - 1 # between 0 and 1 (1 excluded)
    fx = sign(x_clipped) * math.log1p(1 + abs(x_clipped)) / math.log1p(1 + max_x) # between 0 and 1 (1 excluded)
    # f = int(256 * f_scaled) # int between 0 and 256 (256 excluded)
    # f = int((256 * math.log1p(1+x_clipped)) / (math.log1p(max_x) + 1))
    # print("x: {0}\tfx: {1}".format(x, fx))
    smoothing = 0.6
    fx = (1 - smoothing) * fx + smoothing * moving_av_x
    return fx

anomaly_moving_av = 0
anomaly_moving_variance = 0
anomaly_smoothing = 0.95

def anomalyExists():
    error_window_length = 60 * 2
    if (len(stream_error) < error_window_length + 1):
        return False
    recent_errors = np.array(stream_error[-error_window_length - 1:-1])
    mean = np.mean(recent_errors)
    sd = math.sqrt(np.var(recent_errors))
    global anomaly_moving_av, anomaly_moving_variance
    anomaly_moving_av = mean
    anomaly_moving_variance = sd * sd
    current_point = stream_error[-1]
    # current_variance = (current_point - anomaly_moving_av) * (current_point - anomaly_moving_av)
    ans = abs(current_point - anomaly_moving_av) > anomaly_precision * math.sqrt(anomaly_moving_variance)
    # anomaly_moving_av = (1 - anomaly_smoothing) * current_point + anomaly_smoothing * anomaly_moving_av
    # anomaly_moving_variance = (1 - anomaly_smoothing) * current_variance + anomaly_smoothing * anomaly_moving_variance
    return ans

def observePointAndPredict(x):

    # save raw stream data
    raw_stream_data.append(x)

    # do preprocessing and save pre-processed stream:
    # global av_x
    # if len(raw_stream_data) == 0: av_x = x
    # else: av_x += 0.01 * (x - av_x)
    fx = transformInput(x, max_x, 0 if len(stream_data) == 0 else stream_data[-1])
    stream_data.append(fx)


    # if x==0:
    #     print("value of fx is " + str(fx))

    # right now data stream and predicted stream are of equal length

    stream_error.append(abs(fx - stream_predicted[-1]))
    stream_anomalies.append(1 if anomalyExists() else 0)
    stream_anomalies_av.append(anomaly_moving_av)
    stream_anomalies_ul.append(min(anomaly_moving_av + anomaly_precision * math.sqrt(anomaly_moving_variance), 1))
    stream_anomalies_ll.append(max(anomaly_moving_av - anomaly_precision * math.sqrt(anomaly_moving_variance), 0))

    # predict from data so far (partially from actual data points and partially by building on predictions):
    if len(stream_data) >= model_input_length:
        if len(stream_data) >  model_input_length + stream_prediction_latency:
            no_predictions_availble = stream_prediction_latency
        else:
            no_predictions_availble = len(stream_data) - model_input_length
        
        # print("No of predictions availble: {0}\tModel Input length: {1}".format(no_predictions_availble, model_input_length))

        if no_predictions_availble > 0:
            input = stream_data[-model_input_length:-no_predictions_availble]
            input.extend(stream_predicted[-no_predictions_availble:])
        else:
            input = stream_data

        # print(input)
        input_np = np.array(input).reshape(1, len(input), 1)
        #print(input_np)
        predicted = stream_model.predict(input_np, batch_size = 1)[0]
    else:
        predicted = 0

    #print(predicted)
    stream_predicted.append(predicted)

    # now learn from data so far. Do an SGD step:
    if (not skipTraining and len(stream_data) > model_input_length + 1 + 100 and len(stream_data) % 2 == 0):
        x_train, y_train = getMiniBatch(stream_data, 32, model_input_length)
        stream_model.train_on_batch(x_train, y_train)
        if (len(stream_data) % 500 == 0):
            stream_model.save(model_name)

def downsample(data, n_points):
    if n_points > len(data):
        return data[:]
    ans = []
    skip = int(len(data)/n_points)
    for i in range(len(data)):
        if i % skip == 0:
            ans.append(data[i])
    return data

def plot_all():
    n = 30000
    #plt.plot(raw_stream_data, color = 'black')
    plt.plot(downsample(stream_data, n), color = 'green')
    plt.plot(downsample(stream_predicted, n), color='blue', alpha = 0.9)
    plt.plot(downsample(stream_error, n), color='red', alpha = 0.9)
    plt.plot(downsample(stream_anomalies_ll, n), color='orange', alpha = 0.6)
    plt.plot(downsample(stream_anomalies_ul, n), color='orange', alpha = 0.6)
    plt.plot(downsample(stream_anomalies_av, n), color='orange', alpha = 1)
    plt.plot(downsample(stream_anomalies, n), color='black', alpha = 0.7)
    plt.show()

print("Initializing data ... ")
if dummyData:
    dataLoader.initializeDummyData()
else:
    raw_data = dataLoader.initializeFromFile('archive_opm.csv')
print("Done")
print("")

i = 0
x = dataLoader.getNextPoint()
while x is not None:
    observePointAndPredict(x)
    print('{0}'.format(i))
    i += 1
    x = dataLoader.getNextPoint()

plot_all()
