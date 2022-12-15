import os
import numpy as np
import csv
import random
import torch

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

    classes = ['coughing', 'laughing', 'throat_cleaning', 'speaking', 'walking']
    
    root_path = ['volunteer1', 'volunteer2', 'volunteer3', 'volunteer4', 'volunteer5']
    save_root = 'volunteers'
    
    # Split into train and test sets, customize the dataset
    train = np.empty([1, 1001])
    test = np.empty([1, 1001])
    data_range = [range(0,10), range(10, 55)]

    for v_root in root_path:
        
        for c in range(len(classes)):
            data_temp = np.empty([1, 1001])

            root = os.path.join(v_root, classes[c])
            for idx, filename in enumerate(os.listdir(root)):
                
                data_file = os.path.join(root, filename)
                data = read_csv_file(data_file)
                data[:,-1] = c
                data_temp = np.concatenate((data_temp, data))

            data_temp = data_temp[1:,]
            random.shuffle(data_temp)

            # Split randomly into train and test sets
            test = np.concatenate((test, data_temp[data_range[0]]))
            data_range[1] = range(10, len(data_temp))
            train = np.concatenate((train, data_temp[data_range[1]]))

    test = test[1:,]
    train = train[1:,]

    test = torch.from_numpy(test.copy()).cuda()
    train = torch.from_numpy(train.copy()).cuda()
    
    train_name = save_root + '/train.pth'
    test_name = save_root + '/test.pth'
    
    torch.save(train, train_name)
    torch.save(test, test_name)