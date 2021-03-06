import numpy as np
import random
import sys
import math
import csv
from matplotlib import pyplot as plt

def load_data():
    return np.array([250 + 250 * math.sin(float(i)/100) for i in range(30000)]).reshape(30000,1)
    #return np.array([(i%300) * math.sin(float(i)/100) for i in range(10000)]).reshape(10000,1)

def load_data_opm():
    data = []
    with open('archive_opm.csv') as csvfile:
        opmCounts = csv.reader(csvfile, quotechar='"')
        for row in opmCounts:
            for i in range(len(row)):
                row[i] = row[i].strip()
            data.append(float(row[0]))
    return np.array(data).reshape(len(data),1)

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
            y_onehot = one_hot(y[0], 256)
            x_batch.append(x)
            y_batch.append(y_onehot)
    x_batch = np.array(x_batch).reshape(batch_size, input_length, 1)
    y_batch = np.array(y_batch).reshape(batch_size, 256)
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
    model.add(Convolution1D(input_shape = (input_length,1), nb_filter=32,filter_length=4,border_mode='valid',activation='relu',subsample_length=1))
    model.add(MaxPooling1D(pool_length=4))
    model.add(Convolution1D(input_shape = (input_length,1), nb_filter=32,filter_length=4,border_mode='valid',activation='relu',subsample_length=1))
    model.add(MaxPooling1D(pool_length=4))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
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

# input_length = 400
# prediction_length = 900
# raw_data = load_data()
# #raw_data = smooth(load_data_opm(), 31)
# print("raw_data len: " + str(len(raw_data)))
# x_batch, y_batch = getFullBatch(raw_data, input_length)
# print("full batch len: " + str(len(y_batch)))
# train_upto_index = int(5.*len(x_batch)/8.)
# x_train, y_train = x_batch[0:train_upto_index], y_batch[0:train_upto_index]
# x_test, y_test = x_batch[train_upto_index:], y_batch[train_upto_index:]


# print("creating model..")
# model = createModelCNN(input_length, 1)

# print("Fitting model..")
# model.fit(x_train, y_train, epochs=10, batch_size=32)

# print("Evaluating model..")
# eval = model.evaluate(x_test, y_test, batch_size=32)
# print()
# print(eval)
# sys.stdout.flush()

# print("Prediting for given points..")
# #y_predicted = model.predict(x_test, batch_size=len(x_test))
# y_predicted = extended_predict(model, x_test[0], prediction_length)
# plt.plot(list(raw_data), color='green')
# plt.plot([input_length + train_upto_index + x for x in range(prediction_length)] ,list(y_predicted[0:prediction_length]), color='red')
# plt.show()

model_input_length = 300
stream_prediction_latency = 100
stream_data = []
raw_stream_data = []
stream_predicted = [np.zeros([1])]
stream_model = createModelCNN(model_input_length, 1)
stream_error_probability = []
# the average of stream data over last model_input_length points
av_x = 0
max_x = 2000

def sign(x):
    if x < 0:
        return -1
    if x > 0:
        return 1
    return 0

def transformInput(x, max_x):
    u = 255
    x_clipped = min(x, max_x - 1) # between 0 and max_x (max_x excluded)
    # x_scaled = x_clipped / max_x # between 0 and 1 (1 excluded)
    f_scaled = sign(x_clipped) * math.log1p(1 + abs(x_clipped)) / math.log1p(1 + max_x) # between 0 and 1 (1 excluded)
    f = int(256 * f_scaled) # int between 0 and 256 (256 excluded)
    # f = int((256 * math.log1p(1+x_clipped)) / (math.log1p(max_x) + 1))
    return int(256 * x_clipped / max_x)

def observePointAndPredict(x):

    # save raw stream data
    raw_stream_data.append(x)

    # do preprocessing and save pre-processed stream:
    # global av_x
    # if len(raw_stream_data) == 0: av_x = x
    # else: av_x += 0.01 * (x - av_x)
    fx = transformInput(x, max_x)
    stream_data.append(fx)

    # if x==0:
    #     print("value of fx is " + str(fx))

    # right now data stream and predicted stream are of equal length

    error_probability = 
    #stream_error_probability.

    # predict from data so far (partially from actual data points and partially by building on predictions):
    if len(stream_data) >= model_input_length:
        if len(stream_data) >  model_input_length + stream_prediction_latency:
            no_predictions_availble = stream_prediction_latency
        else:
            no_predictions_availble = len(stream_data) - model_input_length
        
        if no_predictions_availble > 0:
            input = stream_data[-model_input_length:-no_predictions_availble]
            input.extend(stream_predicted[-no_predictions_availble:])
        else:
            input = stream_data

        input_np = np.array(input).reshape(1, len(input), 1)
        #print(input_np)
        predicted = np.argmax(stream_model.predict(input_np, batch_size = 1))
    else:
        predicted = 0

    #print(predicted)
    stream_predicted.append(predicted)

    # now learn from data so far. Do an SGD step:
    if (len(stream_data) > model_input_length + 1 + 100 and len(stream_data) % 10 == 0):
        x_train, y_train = getMiniBatch(stream_data, 32, model_input_length)
        stream_model.train_on_batch(x_train, y_train)

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
    plt.plot(raw_stream_data, color = 'black')
    plt.plot(stream_data, color = 'green')
    plt.plot(downsample(stream_predicted, n), color='blue')
    #plt.plot(downsample(stream_error, n), color='red')
    plt.show()

raw_data = load_data()
print("Loadind and smoothing data ... ")
#raw_data = smooth(load_data_opm(), 31)[0:10000]
#raw_data = [np]
print("Data loaded")
print("")

for i in range(len(raw_data)):
    x = raw_data[i]
    # print("x is " + str(x))
    observePointAndPredict(x)
    print(str((i*100)/len(raw_data)) + "%")

plot_all()

# plt.plot(downsample(raw_data, 5000), color='green')
# plt.show()