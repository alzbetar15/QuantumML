
import Training
import QCNN_circuit
import numpy as np
import pennylane as qml
from pennylane.templates.embeddings import AmplitudeEmbedding
#pip3 install scikit-learn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from complex_wave_generator import complex_wave_gen
import pickle
import datetime
import os

# This is an implementation of data_embedding function used for 8 qubits Quantum Convolutional Neural Network (QCNN)
# and Hierarchical Quantum Classifier circuit.

def data_load_and_process(dataset):
    '''
    args:
    dataset-[outputs, labels] labelled data (either noise or sin wave)

    returns:
    randomly shuffled training and test data ot feed into QCNN/CNN
    '''
    #split dataset into 80% training / 20%test after randomly shuffling
    #need to randomly shuffle as first half of dataset is just noise and second is noisy sin waves
    
    #x_train, x_test, y_train, y_test = train_test_split(dataset[0], dataset[1], test_size=0.2, random_state=0, shuffle=True)    
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10

    # train is now 75% of the entire data set
    x_train, x_test, y_train, y_test = train_test_split(dataset[0], dataset[1], test_size=1 - train_ratio)

    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 

    #print(x_train, x_val, x_test)

    return (x_train,x_val, x_test, y_train,y_val, y_test)

def data_load_and_process1(dataset):
    '''
    args:
    dataset-[outputs, labels] labelled data (either noise or sin wave)

    returns:
    randomly shuffled training and test data ot feed into QCNN/CNN
    '''
    #split dataset into 80% training / 20%test after randomly shuffling
    #need to randomly shuffle as first half of dataset is just noise and second is noisy sin waves

    #QE CHANGE THIS TO HAVE THREE DATA SETS, test train and validate
    x_train, x_test, y_train, y_test = train_test_split(dataset[0], dataset[1], test_size=0.8, random_state=0, shuffle=True)    
    
    return (x_train, x_test, y_train, y_test)

def data_embedding(X): #encode input X with 8 qubits with amplitude encoding
    '''
    Embeds 256 data points into 8 qubit and normalises the data, with L2 normalisation

    args: X - the data
    '''
    AmplitudeEmbedding(X, wires=range(8), normalize=True)

#Double check this is correct
def accuracy_test(predictions, labels, binary = True):
    '''
    This functions calculates the accuracy of the preedicitons
    args: predictions - is the label outputed by neural network (QCNN/CNN)
          labels - Y_test/Y_train datta
          binary True/Flase (not used)
    '''
    #QE add meaningful varaible names
    acc = 0
    for l,p in zip(labels, predictions):
        if p[0] > p[1]: 
            P = 0
        else:
            P = 1
        if P == l:
            acc = acc + 1
    return acc / len(labels)





def Benchmarking(dataset, Unitaries, U_num_params, filename,testName ,circuit, steps, snr, binary=True):
    '''
    This function benchmarks the QCNN
    '''
    #Make this actually epochs
    print('Unitaries',Unitaries)
    I = len(Unitaries) # Number of Quantum circuits try
    try:
        path = "Result/"+str(datetime.datetime.now().date())+str(testName)
        os.mkdir(path)
    except:
         print("File "+path+"already created")
    for i in range(I):
        start = time.time()

        f = open(path+"/"+filename+'.txt', 'a')
        U = Unitaries[i]
        U_params = U_num_params[i]
        Embedding='Amplitude'

            #get data
        #calls function in this file

        X_train, X_val, X_test, Y_train, Y_val, Y_test = data_load_and_process(dataset)
        
        currentData = (X_train, X_val, X_test, Y_train, Y_val, Y_test)
        #look at difference between the two functions
        try:
            datapath = "Data/"+str(datetime.datetime.now().date())+str(testName)
            os.mkdir(datapath)
        except:
            print("File "+datapath+"already created")
        currentfile = str(datapath) +"/data"+str(testName)+".pkl"
        print("Saving current Data:",currentfile)
        pickle.dump(currentData, open(currentfile,'wb'))
        #Xn_train, Xn_test, Yn_train, Yn_test = data_load_and_process1(sin_generator.sin_gen3(snr,256))

        print("\n")
        lend=str(len(dataset[0]))
        #QE Here add in a laoding bar look at obsiden for notes
        print("Loss History for " + circuit + " circuits, " + U + " Amplitude with " +'cross entropy' + ' trained with: ' + lend + ' with snr: ' +str(snr))
        #calls the training function, work out where are the hyper parameters
        #Save the Paramaters
        #if its just parameters dont even really need to pickel
        for i in range(2):
            loss_history, trained_params = Training.circuit_training(X_train,X_val, Y_train,Y_val, U, U_params, steps,testName)
            Valpredictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params) for x in X_val]
            Trainingpredictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params) for x in X_train]
            #calculate accuray
            accuracy = accuracy_test(Trainingpredictions, Y_train, binary)
            #for that epoch looks at the accuracy for the model and looks at the current accuracy for both seen (training)
            print("Training Dataset Accuracy for " + U + " Amplitude :" + str(accuracy))
            #and partially seen data (validation)
            accuracy = accuracy_test(Valpredictions, Y_val, binary)
            print("Validation Dataset Accuracy for " + U + " Amplitude :" + str(accuracy))

        print("trained parameters",trained_params)
        #pltos the graph that is outputted QCNN loss.png
        plt.plot(loss_history, label=U)
        plt.xlabel('Epochs')
        plt.ylabel('Training Loss')
        plt.title('Loss History across '+ str(steps) + 'epochs.')
        plt.savefig(path+'\QCNN_Loss'+str(snr)+'.png')

        #makes predictions of test set with trained parameters
        #Now training Off Validation data sets 
        testPredictions =[QCNN_circuit.QCNN(x, trained_params, U, U_params) for x in X_test]
        testAccuracy = accuracy_test(Trainingpredictions, Y_test, binary)
        print("test Accuracy for " + U + " Amplitude :" + str(accuracy))
        #Writes file
        f.write("Loss History for " + circuit + " circuits, " + U + " Amplitude with " +'cross entropy' + ' trained with: ' + lend + ' with snr: ' +str(snr))
        f.write("\n")
        f.write(str(loss_history))
        f.write("\n")
        f.write("Total time: "+ str(time.time() - start)+ "seconds.cc")
        f.write("\n")
        f.write("Training Accuracy for " + U + " Test Acuracy :" + str(testAccuracy))
        f.write("Test Accuracy for " + U + " Test Acuracy :" + str(testAccuracy))
        f.write("\n")
        f.write("Accuracy for " + U + " Test Acuracy :" + str(testAccuracy))
        f.write("\n")
        f.write("\n")
    f.close()


def Benchmarking_new(dataset, Unitaries, U_num_params, filename, circuit, steps, snr, binary=True):
    '''
    This function benchmarks the QCNN
    '''
    #Is this where its deciding epochs
    I = len(Unitaries) # Number of Quantum circuits try

    for i in range(I):
        start = time.time()
        f = open('Result/'+filename+'.txt', 'a')
        U = Unitaries[i]
        U_params = U_num_params[i]
        Embedding='Amplitude'

            #get data
        X_train, X_test, Y_train, Y_test = data_load_and_process(dataset)


        print("\n")
        lend=str(len(dataset[0]))
        print("Loss History for " + circuit + " circuits, " + U + " Amplitude with " +'cross entropy' + ' trained with: ' + lend + ' with snr: ' +str(snr))
        loss_history, trained_params = Training.circuit_training(X_train, Y_train, U, U_params, steps)

        plt.plot(loss_history, label=U)
        plt.xlabel('Steps')
        plt.ylabel('Training Loss')
        plt.title('Loss History across '+ str(steps) + 'Steps.')
        plt.savefig('QCNN Loss')

            #makes predictions of test set with trained parameters
        lsnr=[5,4,3,2,1,0.5,0.25,0.1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01,0.009,0.008,0.007,0.006,0.005,0.004,0.003,0.002,0.001,0.0009,0.0005,0.0001,0.00009,0.00001]
        for nsnr in lsnr:
            Xn_train, Xn_test, Yn_train, Yn_test = data_load_and_process1(complex_wave_gen(nsnr,256))
            predictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params) for x in Xn_test]
                
                #calculate accuray
            accuracy = accuracy_test(predictions, Yn_test, binary)
            print("Accuracy for " + U + " Amplitude :" + str(accuracy))

            f.write("Loss History for " + circuit + " circuits, " + U + " Amplitude with " +'cross entropy' + ' trained with: ' + lend + ' with snr: ' +str(snr))
            f.write("\n")
            f.write(str(loss_history))
            f.write("\n")
            f.write("Total time: "+ str(time.time() - start)+ "seconds.cc")
            f.write("\n")
            f.write("Accuracy for " + U + " Amplitude :" + str(accuracy))
            f.write("\n")
            f.write("\n")
    f.close()

