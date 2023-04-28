import pandas as pd
from torch.utils import data
import numpy as np
import os
from PIL import Image
from torchvision import transforms
def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
       
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)
    def transform(self,img):
        shortSize=min(img.size)
        script = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop((shortSize,shortSize)),
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

        return script(img)
    def __getitem__(self, index):
        
        single_img_name = os.path.join(self.root,self.img_name[index]+'.jpeg')
        img=Image.open(single_img_name)
        label=self.label[index]
        try:
            return self.transform(img), label
        except:
            single_img_name = os.path.join(self.root,self.img_name[index+1]+'.jpeg')
            img=Image.open(single_img_name)
            label=self.label[index+1]
            return self.transform(img), label

