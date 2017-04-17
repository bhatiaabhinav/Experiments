import numpy as np
import csv
import os.path
import math

class DataLoader:
    # abstract class DataLoader

    def __init__(self):
        pass

    def getNextPoint(self):
        return None


class DummyDataLoader(DataLoader):
    __index = 0

    def __init__(self):
        super(DataLoader, self).__init__()

    def getNextPoint(self):
        y = np.array(250 + 250 * math.sin(float(__index)/100))
        __index += 1
        return y


class FileDataLoader(DataLoader):

    def __init__(self, filename):
        super(DataLoader, self).__init__()
        if os.path.isfile(filename):
            self.__filename = filename
        else:
            print('File not found. Loading dummy data.')
        self.load_data()

    def load_data(self):
        data = []
        print('Loading from {0}'.format(self.__filename))
        with open(self.__filename) as csvfile:
            counts = csv.reader(csvfile, quotechar='"')
            for row in counts:
                for i in range(len(row)):
                    row[i] = row[i].strip()
                data.append(float(row[0]))
        self.__data = np.array(data).reshape(len(data), 1)
        self.__index = 0

    def getNextPoint(self):
        if self.__index >= len(self.__data):
            return None
        y = self.__data[self.__index]
        self.__index += 1
        return y


class KafkaDataLoader(DataLoader):
    # Ishabh to implment this
    pass