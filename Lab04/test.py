import pandas as pd
import numpy as np
import math
label = pd.read_csv('train_label.csv')
label=np.squeeze(label)
test=[0,0,0,0,0]
for i in label:
    test[i]=test[i]+1
print(test)
print(test[0]/sum(test))