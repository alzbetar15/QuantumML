
import Training
import QCNN_circuit
import numpy as np
import pennylane as qml
from pennylane.templates.embeddings import AmplitudeEmbedding
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import sin_generator

# This is an implementation of data_embedding function used for 8 qubits Quantum Convolutional Neural Network (QCNN)
# and Hierarchical Quantum Classifier circuit.

def data_load_and_process1(dataset):
    '''
    args:
    dataset-[outputs, labels] labelled data (either noise or sin wave)

    returns:
    randomly shuffled training and test data ot feed into QCNN/CNN
    '''
    #split dataset into 80% training / 20%test after randomly shuffling
    #need to randomly shuffle as first half of dataset is just noise and second is noisy sin waves
    
    x_train, x_test, y_train, y_test = train_test_split(dataset[0], dataset[1], test_size=0.2, random_state=0, shuffle=True)    
    
    return (x_train, x_test, y_train, y_test)

def data_load_and_process(dataset):
    '''
    args:
    dataset-[outputs, labels] labelled data (either noise or sin wave)

    returns:
    randomly shuffled training and test data ot feed into QCNN/CNN
    '''
    #split dataset into 80% training / 20%test after randomly shuffling
    #need to randomly shuffle as first half of dataset is just noise and second is noisy sin waves
    
    x_train, x_test, y_train, y_test = train_test_split(dataset[0], dataset[1], test_size=0.2, random_state=0, shuffle=True)    
    
    return (x_train, x_test, y_train, y_test)

def data_embedding(X): #encode input X with 8 qubits with amplitude encoding
    '''
    Embeds 256 data points into 8 qubit and normalises the data, with L2 normalisation

    args: X - the data
    '''
    AmplitudeEmbedding(X, wires=range(8), normalize=True)


def accuracy_test(predictions, labels, binary = True):
    '''
    This functions calculates the accuracy of the preedicitons
    args: predictions - is the label outputed by neural network (QCNN/CNN)
          labels - Y_test/Y_train datta
          binary True/Flase (not used)
    '''
    acc = 0
    for l,p in zip(labels, predictions):
        if p[0] > p[1]: 
            P = 0
        else:
            P = 1
        if P == l:
            acc = acc + 1
    return acc / len(labels)





def Benchmarking(dataset, Unitaries, U_num_params, filename, circuit, steps, snr, binary=True):
    '''
    This function benchmarks the QCNN
    '''

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
        plt.xlabel('Epochs')
        plt.ylabel('Training Loss')
        plt.title('Loss History across '+ str(steps) + 'epochs.')
        plt.savefig('QCNN Loss')

            #makes predictions of test set with trained parameters
        predictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params) for x in X_test]
            
            #calculate accuray
        accuracy = accuracy_test(predictions, Y_test, binary)
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
        ll=[10000]#,100000]
        for i in range(0,20):
            
            for length in ll:
                gdataset = sin_generator.sin_gen3(i, length)
                gX_train, gX_test, gY_train, gY_test = data_load_and_process1(gdataset)
                print('gX_test',len(gX_test))
                predictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params) for x in gX_test]
                
                #calculate accuray
                accuracy = accuracy_test(predictions, gY_test, binary)
                print("Accuracy for " + 'Pnoise test with Gnoise' +'SNR:'+str(i)+ ' length: '+str(length)+" Amplitude :" + str(accuracy))
                f.write("Accuracy for " + 'Pnoise test with Gnoise' + " Amplitude :" + str(accuracy))
                f.write("\n")
                f.write("\n")
            #dataset = sin_generator.sin_genn(i, 10000)
            #X_train, X_test, Y_train, Y_test = data_load_and_process1(dataset)

            #predictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params) for x in X_test]
            
            #calculate accuray
            #accuracy = accuracy_test(predictions, Y_test, binary)
            #print("just noise Accuracy for " + 'Pnoise test with Gnoise' +'SNR:'+str(i)+ " Amplitude :" + str(accuracy))
            #f.write("just noise Accuracy for " + 'Pnoise test with Gnoise' + " Amplitude :" + str(accuracy))

            #dataset = sin_generator.sin_gen3m(i, 10000)
            #X_train, X_test, Y_train, Y_test = data_load_and_process1(dataset)

            #predictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params) for x in X_test]
            
            #calculate accuray
            #accuracy = accuracy_test(predictions, Y_test, binary)
            #print("mix noise Accuracy for " + 'Pnoise test with Gnoise' +'SNR:'+str(i)+ " Amplitude :" + str(accuracy))
            #f.write("mix noise Accuracy for " + 'Pnoise test with Gnoise' + " Amplitude :" + str(accuracy))
        #f.write("Loss History for " + circuit + " circuits, " + U + " Amplitude with " +'cross entropy' + ' trained with: ' + lend + ' with snr: ' +str(snr))
        #f.write("\n")
        #f.write(str(loss_history))
        #f.write("\n")
        #f.write("Total time: "+ str(time.time() - start)+ "seconds.cc")
        #f.write("\n")
        #f.write("Accuracy for " + U + " Amplitude :" + str(accuracy))
            
    f.close()



def Benchmarking_vs(dataset, Unitaries, U_num_params, filename, circuit, steps, snr, binary=True):
    '''
    This function benchmarks the QCNN
    '''

    I = len(Unitaries) # Number of Quantum circuits try
    gdataset = sin_generator.sin_gen3(1, 10000) #snr of 2
    gX_train, gX_test, gY_train, gY_test = data_load_and_process1(gdataset)

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
        for s in range(10,210,10):
            loss_history, trained_params = Training.circuit_training(X_train, Y_train, U, U_params, s)

            plt.plot(loss_history, label=U)
            plt.xlabel('Epochs')
            plt.ylabel('Training Loss')
            plt.title('Loss History across '+ str(s) + 'epochs.')
            plt.savefig('QCNN Loss')

                #makes predictions of test set with trained parameters
            predictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params) for x in X_test]
                
                #calculate accuray
            accuracy = accuracy_test(predictions, Y_test, binary)
            print("Accuracy for " + U + " Amplitude :" + str(accuracy))

            f.write("Loss History for " + circuit + " circuits, " + U + " Amplitude with " +'cross entropy' + ' trained with: ' + lend + ' with snr: ' +str(snr))
            f.write("\n")
            f.write(str(loss_history))
            f.write("\n")
            print("Total time: "+ str(time.time() - start)+ "seconds.cc")
            f.write("Total time: "+ str(time.time() - start)+ "seconds.cc")
            f.write("\n")
            f.write("Accuracy for " + U + " Amplitude :" + str(accuracy))
            f.write("\n")
            f.write("\n")
            #ll=[10000]#,100000]
            #j=2
            #for i in range(0,20):
                
            #for length in ll:
                #gdataset = sin_generator.sin_gen3(j, length)
                #gX_train, gX_test, gY_train, gY_test = data_load_and_process1(gdataset)
            print('gX_test',len(gX_test))
            predictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params) for x in gX_test]
            
            #calculate accuray
            accuracy = accuracy_test(predictions, gY_test, binary)
            print("Accuracy for " + 'Pnoise test with Gnoise ' +'SNR:'+str(i)+ ' length: '+str(1000)+" Amplitude :" + str(accuracy))
            f.write("Accuracy for " + 'Pnoise test with Gnoise ' + " Amplitude :" + str(accuracy))
            f.write("\n")
            f.write("\n")
            #dataset = sin_generator.sin_genn(i, 10000)
            #X_train, X_test, Y_train, Y_test = data_load_and_process1(dataset)

            #predictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params) for x in X_test]
            
            #calculate accuray
            #accuracy = accuracy_test(predictions, Y_test, binary)
            #print("just noise Accuracy for " + 'Pnoise test with Gnoise' +'SNR:'+str(i)+ " Amplitude :" + str(accuracy))
            #f.write("just noise Accuracy for " + 'Pnoise test with Gnoise' + " Amplitude :" + str(accuracy))

            #dataset = sin_generator.sin_gen3m(i, 10000)
            #X_train, X_test, Y_train, Y_test = data_load_and_process1(dataset)

            #predictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params) for x in X_test]
            
            #calculate accuray
            #accuracy = accuracy_test(predictions, Y_test, binary)
            #print("mix noise Accuracy for " + 'Pnoise test with Gnoise' +'SNR:'+str(i)+ " Amplitude :" + str(accuracy))
            #f.write("mix noise Accuracy for " + 'Pnoise test with Gnoise' + " Amplitude :" + str(accuracy))
        #f.write("Loss History for " + circuit + " circuits, " + U + " Amplitude with " +'cross entropy' + ' trained with: ' + lend + ' with snr: ' +str(snr))
        #f.write("\n")
        #f.write(str(loss_history))
        #f.write("\n")
        #f.write("Total time: "+ str(time.time() - start)+ "seconds.cc")
        #f.write("\n")
        #f.write("Accuracy for " + U + " Amplitude :" + str(accuracy))
            
    f.close()



def Benchmarking_vs1(dataset, Unitaries, U_num_params, filename, circuit, steps, snr, binary=True):
    '''
    This function benchmarks the QCNN
    '''

    I = len(Unitaries) # Number of Quantum circuits try
    gdataset = sin_generator.sin_gen3(1, 10000) #snr of 2
    gX_train, gX_test, gY_train, gY_test = data_load_and_process1(gdataset)

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
        for s in range(10,210,10):
            loss_history, trained_params = Training.circuit_training(X_train, Y_train, U, U_params, s)

            plt.plot(loss_history, label=U)
            plt.xlabel('Epochs')
            plt.ylabel('Training Loss')
            plt.title('Loss History across '+ str(s) + 'epochs.')
            plt.savefig('QCNN Loss')

                #makes predictions of test set with trained parameters
            predictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params) for x in X_test]
                
                #calculate accuray
            accuracy = accuracy_test(predictions, Y_test, binary)
            print("Accuracy for " + U + " Amplitude :" + str(accuracy))

            f.write("Loss History for " + circuit + " circuits, " + U + " Amplitude with " +'cross entropy' + ' trained with: ' + lend + ' with snr: ' +str(snr))
            f.write("\n")
            f.write(str(loss_history))
            f.write("\n")
            print("Total time: "+ str(time.time() - start)+ "seconds.cc")
            f.write("Total time: "+ str(time.time() - start)+ "seconds.cc")
            f.write("\n")
            f.write("Accuracy for " + U + " Amplitude :" + str(accuracy))
            f.write("\n")
            f.write("\n")
            #ll=[10000]#,100000]
            #j=2
            #for i in range(0,20):
                
            #for length in ll:
                #gdataset = sin_generator.sin_gen3(j, length)
                #gX_train, gX_test, gY_train, gY_test = data_load_and_process1(gdataset)
            print('gX_test',len(gX_test))
            predictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params) for x in gX_test]
            
            #calculate accuray
            accuracy = accuracy_test(predictions, gY_test, binary)
            print("Accuracy for " + 'Gnoise test with Gnoise ' +'SNR:'+str(i)+ ' length: '+str(1000)+" Amplitude :" + str(accuracy))
            f.write("Accuracy for " + 'Gnoise test with Gnoise ' + " Amplitude :" + str(accuracy))
            f.write("\n")
            f.write("\n")
            #dataset = sin_generator.sin_genn(i, 10000)
            #X_train, X_test, Y_train, Y_test = data_load_and_process1(dataset)

            #predictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params) for x in X_test]
            
            #calculate accuray
            #accuracy = accuracy_test(predictions, Y_test, binary)
            #print("just noise Accuracy for " + 'Pnoise test with Gnoise' +'SNR:'+str(i)+ " Amplitude :" + str(accuracy))
            #f.write("just noise Accuracy for " + 'Pnoise test with Gnoise' + " Amplitude :" + str(accuracy))

            #dataset = sin_generator.sin_gen3m(i, 10000)
            #X_train, X_test, Y_train, Y_test = data_load_and_process1(dataset)

            #predictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params) for x in X_test]
            
            #calculate accuray
            #accuracy = accuracy_test(predictions, Y_test, binary)
            #print("mix noise Accuracy for " + 'Pnoise test with Gnoise' +'SNR:'+str(i)+ " Amplitude :" + str(accuracy))
            #f.write("mix noise Accuracy for " + 'Pnoise test with Gnoise' + " Amplitude :" + str(accuracy))
        #f.write("Loss History for " + circuit + " circuits, " + U + " Amplitude with " +'cross entropy' + ' trained with: ' + lend + ' with snr: ' +str(snr))
        #f.write("\n")
        #f.write(str(loss_history))
        #f.write("\n")
        #f.write("Total time: "+ str(time.time() - start)+ "seconds.cc")
        #f.write("\n")
        #f.write("Accuracy for " + U + " Amplitude :" + str(accuracy))
            
    f.close()

