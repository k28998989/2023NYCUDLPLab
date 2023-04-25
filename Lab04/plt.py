import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pretrainedACC = pd.read_csv("resnet50 pretrained HF= true.csv")
unpretrainedACC = pd.read_csv("resnet50 pretrained HF= false.csv")
averages = []
avg = 0
print(pretrainedACC[['acc_test']])

plt.ylabel('Accuracy (%)')
plt.xlabel('epoch')
plt.plot(pretrainedACC[['acc_test']],color='red',label='test Pretrained')
plt.plot(pretrainedACC[['acc_train']],color='blue',label='train Pretrained')
plt.plot(unpretrainedACC[['acc_test']],color='green',label='test unPretrained')
plt.plot(unpretrainedACC[['acc_train']],color='black',label='train unPretrained')
plt.title('resnet50')
plt.legend()
#plt.show()
plt.savefig('result_resnet50.png')
