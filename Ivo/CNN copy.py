#import data
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt 
import sin_generator
from sklearn.model_selection import train_test_split

def get_n_params(model):
    np=0
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

steps = 200
n_feature = 2
batch_size = 25
def Benchmarking_CNN(dataset,filename, input_size, optimizer):
    final_layer_size = int(input_size / 4)
    X_train, X_test, Y_train, Y_test = train_test_split(dataset[0], dataset[1], test_size=0.2, random_state=0, shuffle=True)

    CNN = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=n_feature, kernel_size=2, padding=1),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2),
        nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=2, padding=1),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(n_feature * final_layer_size, 2))

    loss_history = []
    for it in range(steps):
        batch_idx = np.random.randint(0, len(X_train), batch_size)
        X_train_batch = np.array([X_train[i] for i in batch_idx])
        Y_train_batch = np.array([Y_train[i] for i in batch_idx])

        X_train_batch_torch = torch.tensor(X_train_batch, dtype=torch.float32)
        X_train_batch_torch.resize_(batch_size, 1, input_size)
        Y_train_batch_torch = torch.tensor(Y_train_batch, dtype=torch.long)

        criterion = nn.CrossEntropyLoss()
        if optimizer == 'adam':
            opt = torch.optim.Adam(CNN.parameters(), lr=0.01, betas=(0.9, 0.999))
        elif optimizer == 'nesterov':
            opt = torch.optim.SGD(CNN.parameters(), lr=0.01, momentum=0.9, nesterov=True)

        Y_pred_batch_torch = CNN(X_train_batch_torch)

        loss = criterion(Y_pred_batch_torch, Y_train_batch_torch)
        loss_history.append(loss.item())
        if it % 10 == 0:
            print("[iteration]: %i, [LOSS]: %.6f" % (it, loss.item()))

        opt.zero_grad()
        loss.backward()
        opt.step()

        X_test_torch = torch.tensor(X_test, dtype=torch.float32)
        X_test_torch.resize_(len(X_test), 1, input_size)
        Y_pred = CNN(X_test_torch).detach().numpy()
        accuracy = accuracy_test(Y_pred, Y_test)
        N_params = get_n_params(CNN)

    filename1=filename+'.txt'
    f = open(filename1, 'a')
    f.write("Loss History for CNN: " )
    f.write("\n")
    f.write(str(loss_history))
    f.write("\n")
    f.write("Accuracy for CNN with " +optimizer + ": " + str(accuracy))
    f.write("\n")
    f.write("Number of Parameters used to train CNN: " + str(N_params))
    f.write("\n")
    f.write("\n")

    f.close()
    plt.plot(loss_history)
    plt.title('CNN Loss History with '+ optimizer+ ' Optimiser')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.savefig(filename+'CNN Loss History with qdata'+optimizer+' Optimiser.png')
steps = 200


filename=r'C:\Users\ivoll\OneDrive\Documents\qml_sin-3 copy\Final_Results_CNN\CNN_qdata_compare'
for i in range(0,20):
    print('running',i)
    dataset = sin_generator.sin_gen3(i,10000)



    
    Benchmarking_CNN(dataset,filename ,input_size = 256 ,optimizer='adam')
    #Benchmarking_CNN(dataset=dataset, classes=classes, Encodings=Encodings, Encodings_size=Encodings_size,
    #                 binary=binary, optimizer='nesterov')

