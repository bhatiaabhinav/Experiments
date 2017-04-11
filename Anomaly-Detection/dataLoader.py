import numpy as np
import csv
import os.path
import math

def load_dummy_data():
    return np.array([250 + 250 * math.sin(float(i)/100) for i in range(10000)]).reshape(10000,1)
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


data = []
index = 0

def initializeDummyData():
    global data, index
    data = load_dummy_data()[-5000:]

def initializeFromFile(filepath):
    global data, index
    if not os.path.isfile(filepath):
        print('File not found. Loading dummy data.')
        data = load_dummy_data()
    else:
        data = load_data_opm()[-10000:]
    index = 0

def getNextPoint():
    global index
    if index >= len(data):
        return None
    p = data[index]
    index += 1
    return p