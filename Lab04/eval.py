import os
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models
print(torch.__version__)
print(torch.cuda.is_available())
from torchvision.models import resnet18,resnet50,ResNet18_Weights,ResNet50_Weights
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import copy
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataloader import RetinopathyLoader
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def getmodels(modelType,pretrained=False):
    if pretrained:
        if modelType=='resnet18':
            model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(in_features=512, out_features = 5)
        else:
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(in_features=2048, out_features = 5)       
    else:
        if modelType=='resnet18':
            model = models.resnet18()
            model.fc = nn.Linear(in_features=512, out_features = 5)
        else:
            model = models.resnet50()
            model.fc = nn.Linear(in_features=2048, out_features = 5)
    return model

def training(model,modelType,loader_train,loader_test,num_class):
    
    
    if modelType=='resnet18':
        epochs=10
    else:
        epochs=5
    optimizer=optim.SGD(model.parameters(),momentum=momentum,lr=lr,weight_decay=5e-4)
    df=pd.DataFrame()
    df['epoch']=range(1,epochs+1)
    best_model_wts=None
    best_evaluated_acc=0
    
    model.to(device)
    acc_train=list()
    acc_test=list()
    for epoch in range(1,epochs+1):
        """
        train
        """
        with torch.set_grad_enabled(True):
            model.train()
            total_loss=0
            correct=0
            for images,targets in tqdm(loader_train, desc=f'epoch: {epoch} (train)'):
                images,targets=images.to(device),targets.to(device,dtype=torch.long)
                #print(targets)
                predict=model(images)
                #print(predict)
                loss=Loss(predict,targets)
                total_loss+=loss.item()
                correct+=predict.max(dim=1)[1].eq(targets).sum().item()
                """
                update
                """
                optimizer.zero_grad()
                loss.backward()  # bp
                optimizer.step()
            total_loss/=len(loader_train.dataset)
            acc=100.*correct/len(loader_train.dataset)
            acc_train.append(acc)
            print(f'epoch{epoch:>2d} loss:{total_loss:.4f} acc:{acc:.2f}%')
        """
        evaluate
        """
        _,acc=evaluate(model,loader_test,num_class)
        acc_test.append(acc)
        # update best_model_wts
        if acc>best_evaluated_acc:
            best_evaluated_acc=acc
            best_model_wts=copy.deepcopy(model.state_dict())
    
    df['acc_train']=acc_train
    df['acc_test']=acc_test
    p='without'
    if pretrained:
        p='with'
    
    # save model
    torch.save(best_model_wts,os.path.join('./models',modelType+p+' pretrained.pt'))
    
    return df
def evaluate(model,loader_test,num_class):
    """
    Args:
        model: resnet model
        loader_test: testing dataloader
        device: gpu/cpu
        num_class: #target class
    Returns:
        confusion_matrix: (num_class,num_class) ndarray
        acc: accuracy rate
    """
    
    confusion_matrix=np.zeros((num_class,num_class))
    model.to(device)
    with torch.set_grad_enabled(False):
        model.eval()
        correct=0
        for images,targets in tqdm(loader_test,desc=f'(test)'):
            images,targets=images.to(device),targets.to(device,dtype=torch.long)
            predict=model(images)
            predict_class=predict.max(dim=1)[1]
            correct+=predict_class.eq(targets).sum().item()
            for i in range(len(targets)):
                confusion_matrix[int(targets[i])][int(predict_class[i])]+=1
        acc=100.*correct/len(loader_test.dataset)
    print(f'acc:{acc:.2f}%')
    # normalize confusion_matrix
    confusion_matrix=confusion_matrix/confusion_matrix.sum(axis=1).reshape(num_class,1)
    
    return confusion_matrix,acc       


def plot(dataframe1,title,p):
    """
    Arguments:
        dataframe1: dataframe with 'epoch','acc_train','acc_test' columns of with pretrained weights model 
        title: figure's title
    Returns:
        figure: an figure
    """
    fig=plt.figure(figsize=(10,6))
    for name in dataframe1.columns[1:]:
        plt.plot(range(1,1+len(dataframe1)),name,data=dataframe1,label=name[4:]+'('+p+' pretraining)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy(%)')
    plt.title(title)
    plt.legend()
    return fig

def plot_confusion_matrix(confusion_matrix):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
    ax.xaxis.set_label_position('top')
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(i, j, '{:.2f}'.format(confusion_matrix[j, i]), va='center', ha='center')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    return fig    
batch=4
lr=1e-3
momentum=0.9
num_classes=5
Loss=nn.CrossEntropyLoss()            
dataset_train=RetinopathyLoader('./data/new_train','train')
loader_train=DataLoader(dataset=dataset_train,batch_size=batch,shuffle=True,num_workers=4)
dataset_test=RetinopathyLoader('./data/new_test','test')
loader_test=DataLoader(dataset=dataset_test,batch_size=batch,shuffle=False,num_workers=4)
print('resnet18 unpretrained')
model_type='resnet18'
#pretrained=True
#model=getmodels(model_type,pretrained)
model=models.resnet18()
model.fc =nn.Linear(in_features=512,out_features=5)
model.load_state_dict(torch.load(os.path.join('models','resnet18without pretrained.pt')))
#df=training(model,model_type,loader_train,loader_test,num_classes)
##testing(model_type,loader_test,num_classes)
confusion_matrix,_=evaluate(model,loader_test,num_classes)

#figure=plot_confusion_matrix(confusion_matrix)
#p='without'
#if pretrained:
#    p='with'
#figure.savefig(model_type+' (with '+p+' pretrained weights HF).png')

#"""
#plot accuracy figure
#"""

#figure=plot(df,'Result Comparison('+p+' pretrained)',p)
#figure.savefig('Result Comparison('+p+' pretrained HF.png')
#p='false'
#if pretrained:
#    p='true'
#path=model_type+' pretrained HF= '+ p +'.csv'
#df.to_csv(path)