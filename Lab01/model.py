import numpy as np
import random
class linear_Unit():
    def __init__(self):
        self.w=random.random()
        self.b=random.random()
       # print("w:",self.w,"b",self.b)
    
    def forward(self,x):
        return self.w*x+self.b/10
    
    def allocateW(self,G,lr):
      #  print(self.w)
        self.w=self.w-lr*G
class linear():
    def __init__(self,input_feature,output_feature):
        self.X=np.zeros((2,1))
        self.IF=input_feature
        self.OF=output_feature #layer_unit_number
        self.layer = []
        self.OU = linear_Unit()
        self.EPS = 1e-9
        self.lr=1e-1
        for i in range(0,self.OF):
            LU=linear_Unit()
            self.layer.append(LU)
    def forward(self,x):
        self.X=x
        sum = np.zeros(self.OF, dtype=float)
        for i in range(0,self.OF):
            sum[i]=sum[i]+self.layer[i].forward(x[0])
            sum[i]=sum[i]+self.layer[i].forward(x[1])
        return sum
    def output(self,x):
        sum = 0
        for i in range(0,x.size):
            sum = sum + self.OU.forward(x[i])
        return sum      
    def num_gradient(self,f, x):
        h=1e-4
        tmp_x = x

        x = tmp_x + h
        fxh1 = f(x)
    

        x = tmp_x - h
        fxh2 = f(x)
        
        grad = (fxh1 - fxh2) / (2 * h)
        x = tmp_x
        return grad  
    def num_gradients(self,x):
        grad = np.zeros_like(x,dtype=float)
        for i,X in enumerate(x):
            grad[i] = self.num_gradient(self.layer[i].forward,X)
        return grad
    def allocateWeight(self,grad):
        for idx,G in enumerate(grad):
            self.layer[idx].allocateW(G,self.lr)
        self.lr = self.lr/1.2
