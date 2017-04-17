import numpy as np


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

def downsample(data, n_points):
    if n_points > len(data):
        return data[:]
    ans = []
    skip = int(len(data)/n_points)
    for i in range(len(data)):
        if i % skip == 0:
            ans.append(data[i])
    return data

def log_transform(x, max_x):
    u = 255
    x_clipped = min(x, max_x - 1) # between 0 and max_x (max_x excluded)
    fx = sign(x_clipped) * math.log1p(1 + abs(x_clipped)) / math.log1p(1 + max_x) # between 0 and 1 (1 excluded)
    return fx