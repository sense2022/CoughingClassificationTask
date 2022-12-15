import csv
import numpy as np
import os
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18

class SpecData(Dataset):
    """
    transfer the input data to Spectrogram
    input_data : numpy array [1, 1000]
    """
    def __init__(self, input_data, datasize=1000, n_fft=64):
        data = input_data[:,:datasize]
        self.data = []
        for i, c in enumerate(data):
            # normalize
            c -= c.min()
            c = ((c / (c.max() + 1e-6)) - 0.5)
            wav = c
            X = librosa.stft(wav, n_fft=n_fft)
            Xdb = librosa.amplitude_to_db(abs(X))
            out = torch.from_numpy(spec_to_image(Xdb)).cuda()
            self.data.append(out[np.newaxis,...])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def spec_to_image(spec, eps=1e-6):
    """
    transfer the Spectrogram to Image
    spec : numpy array
    """
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled

def read_csv_file(filename):
    """
    read csv file.
    filename : path of the file
    """
    with open(filename) as csvfile:
        spamreader = csv.reader(csvfile)
        input_data = np.empty([1, 1000])
        for num, row in enumerate(spamreader):
            if num > 19:
                index = num - 20
                input_data[0, index] = row[3]
        return input_data

if __name__ == '__main__':

    # read csv file
    filename = os.path.join('test_data', 'walking', 'Sheet1.csv')
    model_path = os.path.join('models', 'best.pth')
    classes = ['coughing', 'laughing', 'throat_cleaning', 'speaking', 'walking']
    data = read_csv_file(filename)

    # Transfer data to spectrogram
    data_input = SpecData(data)
    test_loader = DataLoader(data_input, batch_size=1, shuffle=True)

    # CPU or GPU Mode
    if torch.cuda.is_available():
        device=torch.device('cuda:0')
    else:
        device=torch.device('cpu')
    
    # Network Loading
    model = resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512,5)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    loaded_dict_enc = torch.load(model_path)
    model.load_state_dict(loaded_dict_enc)
    model = model.to(device)
    model.eval()
    
    for i, data in enumerate(test_loader):
        x = data
        x = x.type(torch.float32).to(device)
        
        y_hat = model(x)
        pred = y_hat.topk(max((1,)), 1, True, True)

        # Print prediction result
        print(f'Prediction Result: {classes[pred[1].squeeze()]}')
