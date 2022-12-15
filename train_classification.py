import csv
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from visdom import Visdom

import librosa.display
import librosa
import os
from torchvision.models import resnet18, mobilenet_v2

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():

        maxk = max(topk)

        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))


        res = []
        for k in topk:

            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)

            res.append(correct_k.mul_(100.0 / batch_size))

        return res, pred

def setlr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def lr_decay(optimizer, epoch, step):

    if (epoch+1)%step==0:
        new_lr = learning_rate / 10
        optimizer = setlr(optimizer, new_lr)
        print(f'Changed learning rate to {new_lr}')

    return optimizer

def read_csv_file(filename):
    with open(filename) as csvfile:
        spamreader = csv.reader(csvfile)
        input_data = np.empty([1, 1001])
        for num, row in enumerate(spamreader):
            if num > 19:
                index = num - 20
                input_data[0, index] = row[3]
        return input_data

def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled

class SpecData(Dataset):
    def __init__(self, input_data, datasize=1000, n_fft=64):

        data = input_data[:,:datasize]
        label = input_data[:,datasize]
        
        self.data = []
        self.labels = []

        for i, c in enumerate(data):
            
            p = c
            c -= c.min()
            c = ((c / (c.max() + 1e-6)) - 0.5)
            wav = c.cpu().numpy()

            X = librosa.stft(wav, n_fft=n_fft)
            Xdb = librosa.amplitude_to_db(abs(X))

            out = torch.from_numpy(spec_to_image(Xdb)).cuda()
            self.data.append(out[np.newaxis,...])
            self.labels.append(label[i])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


if __name__ == '__main__':

    visualization = False
    if visualization:
        viz = VisdomLinePlotter()
    
    # Load dataset (used cross validation)
    root_path = 'volunteers'
    A = torch.load(os.path.join(root_path, 'data_A.pth'))
    B = torch.load(os.path.join(root_path, 'data_B.pth'))
    C = torch.load(os.path.join(root_path, 'data_C.pth'))
    D = torch.load(os.path.join(root_path, 'data_D.pth'))
    print(A.shape, B.shape, C.shape, D.shape)
    data_train = torch.cat((A, B, C), 0)
    data_test = D

    print('loading data...')
    print('training set:', len(data_train))
    print('testing set:', len(data_test))
    
    data_input = SpecData(data_train)
    test = SpecData(data_test)
    
    train_loader = DataLoader(data_input, batch_size=240, shuffle=True)
    test_loader = DataLoader(test, batch_size=240, shuffle=True)
    
    if torch.cuda.is_available():
        device=torch.device('cuda:0')
    else:
        device=torch.device('cpu')
    
    # initialize the model (ResNet/mobileNetv2)
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(512,5)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    # model = mobilenet_v2(pretrained=True)
    # model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    # model.classifier[1] = nn.Linear(1280,5)
    
    model = model.to(device)
    model.train()
    
    # Setting training loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 60
    train_losses = []
    valid_losses = []
    change_lr = lr_decay
    best_acc = 0

    for epoch in range(1,epochs+1):
        model.train()
        batch_losses=[]
        optimizer = change_lr(optimizer, epoch, 70)
        
        # Start Training
        tbar = tqdm(train_loader)
        for i, data in enumerate(tbar):

            x,y = data
            optimizer.zero_grad()
            x = torch.tensor(x, dtype=torch.float32).to(device)
            y = torch.tensor(y, dtype=torch.long).to(device)
            
            y_hat = model(x)
            
            loss = loss_fn(y_hat, y)
            loss.backward()
            batch_losses.append(loss.item())
            optimizer.step()
        train_losses.append(batch_losses)
        print(f'Epoch - {epoch} Train-Loss : {np.mean(train_losses[-1])}')
        if visualization:
            viz.plot('Loss','Train', 'Class Loss', epoch, np.mean(train_losses[-1]))

        # Start Testing
        model.eval()
        batch_losses=[]
        trace_y = []
        trace_yhat = []
        top0 = AverageMeter()
        tbar = tqdm(test_loader)
        for i, data in enumerate(tbar):
            
            x, y = data
            x = torch.tensor(x, dtype=torch.float32).to(device)
            target = torch.tensor(y, dtype=torch.long).to(device)
            
            y_hat = model(x)
            loss = loss_fn(y_hat, target)    
            batch_losses.append(loss.item())
            
            prec0, pred = accuracy(y_hat, target, topk=(1,))
            
            top0.update(prec0[0], x.size(0))
            tbar.set_description('\r[%d/%d][%d/%d] [Acc: %.3f] [best: %.3f]' % \
                (epoch, 60, i, len(test_loader), top0.avg[0].item(), best_acc))

            if i == 0:
                y_target = target
                y_pred = pred
            else:
                y_target = torch.cat((y_target, target), dim=0)
                y_pred = torch.cat((y_pred, pred), dim=1)
        
        valid_losses.append(batch_losses)
        y_target = y_target.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        y_pred = np.squeeze(y_pred)
        cnf_matrix = confusion_matrix(y_target, y_pred)
        np.set_printoptions(precision=2)
        print('\n')
        print(cnf_matrix)
        print(f'Epoch - {epoch} Val-Loss : {np.mean(valid_losses[-1])}')
        if top0.avg[0].item() >= best_acc:
            torch.save(model.state_dict(), './models/best.pth')
            best_acc = top0.avg[0].item()
            script_model = torch.jit.trace(model, x)
            script_model.save('model.ptl')
            best_cnf = cnf_matrix
        if visualization:
            viz.plot('Accuracy','Val', 'Accuracy', epoch, top0.avg[0].item())
    
    print('Best confusion matrix: ')
    print(best_cnf)