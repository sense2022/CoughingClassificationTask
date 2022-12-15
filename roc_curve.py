
import csv
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import label_binarize
from torchvision.models import resnet18, mobilenet_v2
import librosa.display
import librosa

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

def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled

class SpecData(Dataset):
    def __init__(self, input_data):

        data = input_data[:,:1000]
        label = input_data[:,1000]
        
        self.data = []
        self.labels = []
        
        for i, c in enumerate(data):
            # normalize
            p = c
            c -= c.min()
            c = ((c / (c.max() + 1e-6)) - 0.5)

            wav = c.cpu().numpy()

            X = librosa.stft(wav, n_fft=64)
            Xdb = librosa.amplitude_to_db(abs(X))

            out = torch.from_numpy(spec_to_image(Xdb)).cuda()
            
            self.data.append(out[np.newaxis,...])
            self.labels.append(label[i])
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def read_csv_file(filename):
    with open(filename) as csvfile:
        spamreader = csv.reader(csvfile)
        input_data = np.empty([1, 1001])
        for num, row in enumerate(spamreader):
            if num > 19:
                index = num - 20
                input_data[0, index] = row[3]
        return input_data

if __name__ == '__main__':

    root_path = 'volunteers'

    A = torch.load(os.path.join(root_path, 'data_A.pth'))
    B = torch.load(os.path.join(root_path, 'data_B.pth'))
    C = torch.load(os.path.join(root_path, 'data_C.pth'))
    D = torch.load(os.path.join(root_path, 'data_D.pth'))
    
    data_test = A

    print('loading data...')
    print('testing set:', len(data_test))
    
    test = SpecData(data_test)
    
    test_loader = DataLoader(test, batch_size=240, shuffle=True)

    if torch.cuda.is_available():
        device=torch.device('cuda:0')
    else:
        device=torch.device('cpu')
    
    model = resnet18()
    model.fc = nn.Linear(512,5)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    loaded_dict_enc = torch.load('./models/best.pth', map_location=device)
    model.load_state_dict(loaded_dict_enc)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
   
    model.eval()
    batch_losses=[]
    trace_y = []
    trace_yhat = []
    top0 = AverageMeter()
    tbar = tqdm(test_loader)
    epoch = 0
    for i, data in enumerate(tbar):
        
        x, y = data
        x = torch.tensor(x, dtype=torch.float32).to(device)
        target = torch.tensor(y, dtype=torch.long).to(device)
        
        y_hat = model(x)

        loss = loss_fn(y_hat, target)    
        batch_losses.append(loss.item())
        
        prec0, pred = accuracy(y_hat, target, topk=(1,))
        
        top0.update(prec0[0], x.size(0))
        tbar.set_description('\r[%d/%d][%d/%d] [Acc: %.3f]' % \
            (epoch, 60, i, len(test_loader), top0.avg[0].item()))

        if i == 0:
            y_target = target
            y_pred = pred
            y_score = F.softmax(y_hat)
            
        else:
            y_target = torch.cat((y_target, target), dim=0)
            y_pred = torch.cat((y_pred, pred), dim=1)
            y_score = torch.cat((y_score, F.softmax(y_hat)), dim=0)

    
    y_target = y_target.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    y_score = y_score.detach().cpu().numpy()
    y_pred = np.squeeze(y_pred)
    cnf_matrix = confusion_matrix(y_target, y_pred)
    print(cnf_matrix)

    
    y_test = label_binarize(y_target, classes=[0, 1, 2,3,4])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(5):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot of a ROC curve for a specific class
    plt.figure(figsize=(12, 10))
    lw = 4

    plt.plot(
        fpr[0],
        tpr[0],
        color="darkorange",
        lw=lw,
        label="Coughing(area = %0.3f)" % roc_auc[0],
    )

    plt.plot(
        fpr[1],
        tpr[1],
        color="darkgreen",
        lw=lw,
        label="Laughing (area = %0.3f)" % roc_auc[1],
    )

    plt.plot(
        fpr[2],
        tpr[2],
        color="darkred",
        lw=lw,
        label="Throat Cleaning (area = %0.3f)" % roc_auc[2],
    )

    plt.plot(
        fpr[3],
        tpr[3],
        color="darkblue",
        lw=lw,
        label="Speaking (area = %0.3f)" % roc_auc[3],
    )

    plt.plot(
        fpr[4],
        tpr[4],
        color="purple",
        lw=lw,
        label="Walking (area = %0.3f)" % roc_auc[4],
    )

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        color="Red",
        lw=lw,
        label="Average (area = %0.3f)" % roc_auc["micro"],
    )

    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    # plt.xlim([0.0, 1.0])
    plt.ylim([0.9, 1.01])
    plt.xlabel("False Positive Rate", size=45, fontweight="bold")
    plt.ylabel("True Positive Rate", size=45, fontweight="bold")
    plt.xticks(fontsize=40, fontweight="bold")
    plt.yticks(fontsize=40, fontweight="bold")
    plt.title("ROC Curve of Fold 1", size=45, fontweight="bold")
    

    ax = plt.gca()
    ax.spines['top'].set_linewidth(lw)
    ax.spines['left'].set_linewidth(lw)
    ax.spines['bottom'].set_linewidth(lw)
    ax.spines['right'].set_linewidth(lw)

    plt.rcParams["font.weight"] = "bold"
    plt.tick_params(width=lw)

    plt.legend(loc="lower right", fontsize=30)

    plt.savefig('ROC_fold1.png', bbox_inches = "tight")
