import math
import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    """ Sigmoid function.
    This function accepts any shape of np.ndarray object as input and perform sigmoid operation.
    """
    return 1 / (1 + np.exp(-x))

def der_sigmoid(y):
    """ First derivative of Sigmoid function.
    The input to this function should be the value that output from sigmoid function.
    """
    return y * (1 - y)

def gen_linear(n=100):
    """ Data generation (Linear)

    Args:
        n (int):    the number of data points generated in total.

    Returns:
        data (np.ndarray, np.float):    the generated data with shape (n, 2). Each row represents
            a data point in 2d space.
        labels (np.ndarray, np.int):    the labels that correspond to the data with shape (n, 1).
            Each row represents a corresponding label (0 or 1).
    """
    data = np.random.uniform(0, 1, (n, 2))

    inputs = []
    labels = []

    for point in data:
        inputs.append([point[0], point[1]])

        if point[0] > point[1]:
            labels.append(0)
        else:
            labels.append(1)

    return np.array(inputs), np.array(labels).reshape((-1, 1))

def gen_xor(n=100):
    """ Data generation (XOR)

    Args:
        n (int):    the number of data points generated in total.

    Returns:
        data (np.ndarray, np.float):    the generated data with shape (n, 2). Each row represents
            a data point in 2d space.
        labels (np.ndarray, np.int):    the labels that correspond to the data with shape (n, 1).
            Each row represents a corresponding label (0 or 1).
    """
    data_x = np.linspace(0, 1, n // 2)

    inputs = []
    labels = []

    for x in data_x:
        inputs.append([x, x])
        labels.append(0)

        if x == 1 - x:
            continue

        inputs.append([x, 1 - x])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape((-1, 1))

class LinearNet():
    def __init__(self,hiddensize,lr=0.01):
        self.inputsize=2
        self.EPS=1e-9
        
        '''
        Input->W1->W_O->A1->Output
        '''
        self.input=np.zeros((2,1))
        self.W=[np.random.rand(hiddensize[0],2)]
        self.W_O=[np.zeros(hiddensize[0],1)]
        self.a=[np.zeros(hiddensize[0],1)]
        
    
    def forward(self,input):
        self.input = input
        self.W_O[0]=np.dot(self.W,self.input)
        self.a[0]=sigmoid(self.W_O[0])
        return self.a[0]
    
    def backward(self,output,label):
        self.cost=-(output-label)*(output-label) #RMSE
        
        grad_C_a0=-(label/(output+self.EPS)-(1-label)/(1-output+self.EPS))
        grad_a0_WO0=np.diag(der_sigmoid(self.a[2]).reshape(-1))
        grad_WO0_W0=self.a[0].T
        # get grad_C_W2
        grad_C_W0=np.dot(grad_WO0_W0,)
        # update W0,W1,W2
        self.W[0]-=self.lr*grad_C_W0
    def train(self,input,label):   
        n = input.shape[0]
        for epochs in range(3000):
            for idx in range(n):
                output=self.forward(n[idx:idx+1, :].T)
                
                
        