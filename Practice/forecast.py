import numpy as np
import random
import sys
import math
from matplotlib import pyplot as plt

def load_data():
    #return np.array([math.sin(float(i)/100) for i in range(10000)]).reshape(10000,1)
    return np.array([float((i%300)-150)/150 + math.sin(float(i)/100) for i in range(10000)]).reshape(10000,1)

def getFullBatch(data, input_length):
    x_batch = []
    y_batch = []
    for i in range(len(raw_data)-input_length):
        start_index = i
        end_index = start_index + input_length - 1
        target_index = end_index + 1
        x = raw_data[start_index:end_index+1]
        y = raw_data[target_index:target_index+1][0]
        x_batch.append(x)
        y_batch.append(y)
    x_batch = np.array(x_batch)
    y_batch = np.array(y_batch)
    return x_batch, y_batch

def getMiniBatch(data, batch_size, input_length):
    x_batch = []
    y_batch = []
    for i in range(batch_size):
        start_index = random.randint(0, len(raw_data)-input_length-1)
        end_index = start_index + input_length - 1
        target_index = end_index + 1
        x = raw_data[start_index:end_index+1]
        y = raw_data[target_index:target_index+1][0]
        x_batch.append(x)
        y_batch.append(y)
    x_batch = np.array(x_batch)
    y_batch = np.array(y_batch)
    return x_batch, y_batch

def createModelMLP(input_length, output_length):
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout, Flatten
    model = Sequential()
    model.add(Flatten(input_shape=(input_length,1)))
    model.add(Dense(units=500, input_dim=input_length))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=250))
    model.add(Activation('relu'))
    model.add(Dense(units=output_length))
    model.add(Activation('linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def createModelCNN(input_length, output_length):
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout, Convolution1D, MaxPooling1D, Flatten
    model = Sequential()
    model.add(Convolution1D(input_shape = (input_length,1), nb_filter=32,filter_length=2,border_mode='valid',activation='relu',subsample_length=1))
    model.add(MaxPooling1D(pool_length=4))
    model.add(Convolution1D(input_shape = (input_length,1), nb_filter=64,filter_length=2,border_mode='valid',activation='relu',subsample_length=1))
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

def extended_predict(model, x, prediction_length):
    current_x = list(x)
    predicted_y = []
    for i in range(prediction_length):
        y = model.predict(np.array(current_x).reshape(1,len(x),1), batch_size = 1)[0]
        predicted_y.append(y)
        current_x.pop(0)
        current_x.append(y)
    return predicted_y

input_length = 500
prediction_length = 2000
raw_data = load_data()
x_batch, y_batch = getFullBatch(raw_data, input_length)
train_upto_index = int(0.5*len(x_batch))
x_train, y_train = x_batch[0:train_upto_index], y_batch[0:train_upto_index]
x_test, y_test = x_batch[train_upto_index:], y_batch[train_upto_index:]


print("creating model..")
model = createModelCNN(input_length, 1)

print("Fitting model..")
model.fit(x_train, y_train, epochs=10, batch_size=32)

print("Evaluating model..")
eval = model.evaluate(x_test, y_test, batch_size=32)
print()
print(eval)
sys.stdout.flush()

print("Prediting for given points..")
#y_predicted = model.predict(x_test, batch_size=len(x_test))
y_predicted = extended_predict(model, x_test[0], prediction_length)
plt.plot(list(y_test[0:prediction_length]), color='green')
plt.plot(list(y_predicted[0:prediction_length]), color='red')
plt.show()