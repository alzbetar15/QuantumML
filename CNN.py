import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt 
from complex_wave_generator import complex_wave_gen
from sklearn.model_selection import train_test_split
import pickle
import datetime
import os


def get_n_params(model):
    np = 0
    for p in list(model.parameters()):
        np += p.nelement()
    return np

def accuracy_test(predictions, labels):
    acc = 0
    for (p,l) in zip(predictions, labels):
        if p[0] >= p[1]:
            pred = 0 
        else:
            pred = 1

        if pred == l:
            acc = acc + 1
    acc = acc / len(labels)
    return acc

class complex_ReLU(nn.Module):
    def forward(self, x):
        return nn.ReLU()(x.real) + 1j * nn.ReLU()(x.imag)
        
class complex_MaxPool1d(nn.Module):
    def forward(self, x):
        return nn.MaxPool1d(kernel_size = 2)(x.real) + 1j * nn.MaxPool1d(kernel_size = 2)(x.imag)
        
class complex_CrossEntropyLoss(nn.Module):
    def forward(self, inputs, targets):

        real_loss = nn.CrossEntropyLoss(inputs.real, targets.real)
        imag_loss = nn.CrossEntropyLoss(inputs.imag, targets.imag)

        return (real_loss + imag_loss)/2


#counter =0
steps = 300
#for binary classification
n_feature = 2
batch_size = 10

def Benchmarking_CNN(dataset,filename, input_size, optimizer,smallest):

    final_layer_size = int(input_size / 4)
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10

    X_train, X_test, Y_train, Y_test = train_test_split(dataset[0], dataset[1], test_size = 1 - train_ratio)
    #X_train, X_test, Y_train, Y_test = train_test_split(dataset[0], dataset[1], test_size=0.2, random_state=0, shuffle=True)
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 
    currentData = (X_train, X_val, X_test, Y_train, Y_val, Y_test)
    
    currentfile = "CNN_Data\data"+str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.') +".pkl"
    print("Saving current data:",currentfile)

    path = "CNN_Models/"+str(datetime.datetime.now().date())
    try:
        os.mkdir(path)
    except:
        print("File "+path+"already created")

    pickle.dump(currentData, open(currentfile,'wb'))


    CNN = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=n_feature, kernel_size=2, padding=1, dtype=torch.cfloat),
        complex_ReLU(),
        complex_MaxPool1d(),
        nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=2, padding=1, dtype=torch.cfloat),
        complex_ReLU(),
        complex_MaxPool1d(),
        nn.Flatten(),
        nn.Linear(n_feature * final_layer_size, 2, dtype=torch.cfloat)
        )

    loss_history = []
    for it in range(steps):
        batch_idx = np.random.randint(0, len(X_train), batch_size)
        #batch_idy = np.random.randint(0, len(Y_train), batch_size)
        X_train_batch = np.array([X_train[i] for i in batch_idx])
        Y_train_batch = np.array([Y_train[i] for i in batch_idx])

        X_train_batch_torch = torch.tensor(X_train_batch, dtype=torch.cfloat)
        X_train_batch_torch.resize_(batch_size, 1, input_size)
        Y_train_batch_torch_real = torch.tensor(Y_train_batch.real, dtype=torch.long)
        Y_train_batch_torch_imag = torch.tensor(Y_train_batch.imag, dtype=torch.long)

        criterion = nn.CrossEntropyLoss()

        if optimizer == 'adam':
            opt = torch.optim.Adam(CNN.parameters(), lr=0.01, betas=(0.9, 0.999))
        elif optimizer == 'nesterov':
            opt = torch.optim.SGD(CNN.parameters(), lr=0.1, momentum=0.9, nesterov=True)

        Y_pred_batch_torch = CNN(X_train_batch_torch)
        real_loss = criterion(Y_pred_batch_torch.real, Y_train_batch_torch_real)
        imag_loss = criterion(Y_pred_batch_torch.imag, Y_train_batch_torch_imag)
        loss = (real_loss + imag_loss)/2
        loss_history.append(loss.item())

        if it % 10 == 0:
            print("[iteration]: %i, [LOSS]: %.10f" % (it, loss.item()))
            if loss.item() < smallest:
                currentfile = path+"\model"+str(it)+"C"+str(loss.item())+".pkl"
                #currentfile = "CNN_Models\model"+str()+"C"+str(loss.item())+".pkl"
                print("Saving current parameters:",currentfile)
                pickle.dump(CNN, open(currentfile,'wb'))
                smallest = loss.item()   
        opt.zero_grad()
        loss.backward()
        print(loss.item().type)
        opt.step()

        X_test_torch = torch.tensor(X_val, dtype=torch.cfloat)
        X_test_torch.resize_(len(X_val), 1, input_size)
        Y_pred = CNN(X_test_torch).detach().numpy()
        accuracy = accuracy_test(Y_pred, Y_val)
        N_params = get_n_params(CNN)
 

    filename1 = filename+'.txt'
    with open(filename1, 'a') as f:
        f.write("Loss History for CNN: " )
        f.write("\n")
        f.write(str(loss_history))
        f.write("\n")
        f.write("Accuracy for CNN with " +optimizer + ": " + str(accuracy))
        f.write("\n")
        f.write("Number of Parameters used to train CNN: " + str(N_params))
        f.write("\n")
        f.write("\n")

    plt.plot(loss_history)
    plt.title('CNN Loss History with '+ optimizer+ ' Optimiser')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.savefig(filename+'CNN Loss History with qdata'+optimizer+' Optimiser.png')

if __name__ == "__main__":
    p = 0.8
    freq = [10,30]
    filename = "CNN_Results/CNN_Result"+str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')

    dataset = complex_wave_gen(p,freq,100)
    smallest = 20
    print('running')
        
    Benchmarking_CNN(dataset, filename, input_size = 256, optimizer='nesterov', smallest=smallest)

