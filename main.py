import json
import numpy as np
import random
import argparse
import os 
import warnings
from utils import *
from load_data import load_data
import torch
from model import MyModel
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable as V
from sklearn.model_selection import train_test_split
import model

criterion = torch.nn.CrossEntropyLoss()
windowSize = model.ws 
max_epoch=200
best_score=0
random_state=345
save=True
load=False
test_size=0.933
random.seed(69)
output_model = '/tmp/output/model.pth'

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", default=1e-3, type=float,help="learning_rate")
parser.add_argument("--train_bs", default=64, type=int,help="train_bs")

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten() # [3, 5, 8, 1, 2, ....]
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def create_dataloader(x,y,bs):
    x=np.transpose(x, axes=(0, 3, 1, 2))  
    
    x=V(torch.FloatTensor(x))
    y=V(torch.LongTensor(y))

    dataset=TensorDataset(x,y)
    data_loader = DataLoader(dataset=dataset, batch_size=bs, shuffle=True, num_workers=8,) 
    return data_loader

def save(model, optimizer):
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, output_model)

def eval(model, optimizer,criterion,data_loader,save=save):
    print('\033[1;35m----Evaluating----\033[0m')
    model.eval()
    eval_loss, eval_accuracy, nb_eval_steps = 0, 0, 0

    for i,(batch_x,batch_y) in enumerate(tqdm(data_loader)):
        batch_x=batch_x.to(device)
        batch_y=batch_y.to(device)
        with torch.no_grad():
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            logits = logits.detach().cpu().numpy()
            eval_loss += loss.item()
            label_ids = batch_y.cpu().numpy()
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

    avg_loss = eval_loss / nb_eval_steps 
    avg_accuracy = eval_accuracy / nb_eval_steps
    print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {avg_accuracy:.4f}")  

    global best_score
    if best_score < eval_accuracy / nb_eval_steps:
        best_score = eval_accuracy / nb_eval_steps
        if save:
            save(model, optimizer)
            print('\033[1;31m Better model saved.\033[0m')

    model.train()
    return avg_accuracy
    
def train(model,optimizer,criterion,train_loader,test_loader):
    print('-----------------training------------')
    for epoch in range(max_epoch):
        print('【Epoch:{}】'.format(epoch+1))
        for i,(batch_x,batch_y) in enumerate(tqdm(train_loader)):
            model.train()
            batch_x=batch_x.to(device)
            batch_y=batch_y.to(device)
            logits=model(batch_x)

            loss=criterion(logits,batch_y)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        eval(model, optimizer,criterion,test_loader)

import shutil  
import torch.nn.init as init  

if __name__ == '__main__':
    print('start')

    data_path = 'IP'  
      
    data, gt = load_data(data_path)  
 
    data = Standardize_data(data)

    X_coh, y = createComplexImageCubes(data, gt, windowSize)

    del data, gt
    X_train, X_test, y_train, y_test = train_test_split(X_coh, y, test_size=test_size, random_state=random_state,stratify=y)
    del X_coh, y
    train_loader=create_dataloader(X_train,y_train,args.train_bs)
    del X_train,y_train
    test_loader=create_dataloader(X_test,y_test,args.train_bs)
    del X_test,y_test
    model=MyModel(batch_size=args.train_bs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,weight_decay=0.01)
    
    warnings.filterwarnings("ignore", message="Applied workaround for CuDNN issue, install nvrtc.so")

    if not os.path.exists('/tmp/output'):  
        os.makedirs('/tmp/output')  
    if not os.path.exists('pretrain'):  
        os.makedirs('pretrain')  
    model.to(device)
    
    if load:
        checkpoint = torch.load(output_model)  
        model.load_state_dict(checkpoint['model_state_dict'])  
        optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
        print('Pretrained model loaded.')

    eval(model, optimizer,criterion,test_loader,save=False)
    train(model,optimizer,criterion,train_loader,test_loader)
    print(f'Best score:{best_score:.4f}')