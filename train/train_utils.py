import numpy as np

class RunningAvgQueue(object):
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.data=[]

    def __str__(self):
        return str(self.data)

    def add(self, x):
        self.data.append(x)
        if len(self.data) > self.maxsize:
            self.data.pop(0)

    def mean(self):
        return np.mean(self.data)