import math
import numpy as np
import matplotlib.pyplot as plt
def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n,1)

def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i,0.1*i])
        labels.append(0)
        
        if 0.1*i == 0.5:
            continue
        
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
        
    return np.array(inputs), np.array(labels).reshape(21 ,1)

def show_result(x, y, pred_y):
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i]==0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
            
    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i]==0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()

class linear():
    def __init__(self,input_feature,output_feature):
        self.X=np.zeros((2,1))
        self.layer = [np.random.rand(output_feature,2),np.random.rand(output_feature,output_feature),np.random.rand(1,output_feature)]
        self.z=[np.zeros((output_feature,1)),np.zeros((output_feature,output_feature)),np.zeros((1,1))]
        self.a=[np.zeros((output_feature,1)),np.zeros((output_feature,output_feature)),np.zeros((1,1))]
        self.lr=1e-1
    def forward(self,x):
        self.X=x
        self.z[0]=self.layer[0]@self.X
        self.a[0]=self.sigmoid(self.z[0])
        #self.a[0]=self.ReLU(self.z[0])
        self.z[1]=self.layer[1]@self.a[0]
        self.a[1]=self.sigmoid(self.z[1])
        #self.a[1]=self.ReLU(self.z[1])
        self.z[2]=self.layer[2]@self.a[1]
        self.a[2]=self.sigmoid(self.z[2])
        #self.a[2]=self.ReLU(self.z[2])
        return self.a[2]
    def backprop(self,pred_Y,gt_Y):
        self.loss=-(gt_Y-pred_Y)*(gt_Y-pred_Y) #RMSE
        grad_C_a2=2*(pred_Y-gt_Y)
        grad_a2_z2=np.diag(self.derivative_sigmoid(self.a[2]).reshape(-1))
        grad_z2_L2=self.a[1].T
        grad_C_z2=grad_C_a2@grad_a2_z2
        grad_C_L2=grad_C_z2@grad_z2_L2
        grad_z2_a1=self.layer[2]
        grad_a1_z1=np.diag(self.derivative_sigmoid(self.a[1]).reshape(-1))
        grad_z1_L1=np.zeros((self.z[1].shape[0],self.layer[1].shape[0]*self.layer[1].shape[1]))
        n,m=self.z[1].shape[0],self.layer[1].shape[1]
        for i in range(n):
            grad_z1_L1[i,i*m:(i+1)*m]=self.a[0].reshape(-1)
        grad_C_z1=grad_C_z2@grad_z2_a1@grad_a1_z1
        grad_C_L1=grad_C_z1@grad_z1_L1
        grad_C_L1=grad_C_L1.reshape(n,m)
        
        grad_z1_a0=self.layer[1]
        grad_a0_z0=np.diag(self.derivative_sigmoid(self.a[0]).reshape(-1))
        grad_z0_L0=np.zeros((self.z[0].shape[0],self.layer[0].shape[0]*self.layer[0].shape[1]))
        n,m=self.z[0].shape[0],self.layer[0].shape[1]
        #print(n,m)
        for i in range(n):
            grad_z0_L0[i,i*m:(i+1)*m]=self.X.reshape(-1)
        #print(grad_z0_L0)
        grad_C_z0=grad_C_z1@grad_z1_a0@grad_a0_z0
        #print(grad_C_z0)
        grad_C_L0=grad_C_z0@grad_z0_L0
        #print("C:",grad_C_L0)
        grad_C_L0=grad_C_L0.reshape(n,m)
        #print(grad_W0)
        #print(grad_W1)
        self.layer[0]-=self.lr*grad_C_L0
        #print(self.layer[1],grad_C_L1)
        self.layer[1]-=self.lr*grad_C_L1
        self.layer[2]-=self.lr*grad_C_L2
        return self.loss        
       
    def sigmoid(self,x):
        return 1.0/(1.0+np.exp(-x))
    def derivative_sigmoid(self,x):
        return np.multiply(x, 1.0-x)
    def ReLU(self,x):
        re=np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                re[i][j] = x[i][j] if x[i][j]>=0 else 0
        return re
    def derivate_ReLU(self,x):
        re=np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                re[i][j] = 1 if x[i][j]>=0 else 0
        return re
n=100
x, y = generate_linear(n)
#x, y = generate_XOR_easy()
model=linear(2,10)
lastLoss = 0
inputShape=x.shape[0]
linearOutput=np.zeros(100,dtype=float)
epochs = 10001
totalloss=np.zeros(epochs-1,dtype=float)
for epoch in range(1,epochs):
    loss = np.zeros(n,dtype=float)
    finalloss = 0
    for idx in range(inputShape):
        linearOutput[idx] = model.forward(x[idx:idx+1, :].T)
        #print(linearOutput)
        #finalOutput = model.output(activateOutput)
        
        loss[idx]=model.backprop(linearOutput[idx],y[idx])

    #print("epoch",epoch,"loss",finalloss)
    #loss = np.zeros(n, dtype=float)
    totalloss[epoch-1]=np.sum(loss)
    if epoch%1000==0:
        print("epoch:",epoch,"loss:",(-1*np.sum(loss)))
    lastLoss = finalloss
testOutput=np.zeros(inputShape,dtype=float)
testloss=np.zeros(inputShape,dtype=float)
for idx in range(inputShape):
    testOutput[idx]=model.forward(x[idx:idx+1,:].T)
    testloss[idx]=-(y[idx]-testOutput[idx])*(y[idx]-testOutput[idx])
testOutput=np.round(testOutput)
acc = 0
for i in range(testOutput.shape[0]):
    if y[i] == testOutput[i]:
       acc +=1 
plt1=plt
plt1.plot(-1*totalloss)
plt1.show()
print("testloss:",np.sum(testloss))
print("testAccuracy:",(acc/y.shape[0])*100,"%")
show_result(x, y, testOutput)

    
      

