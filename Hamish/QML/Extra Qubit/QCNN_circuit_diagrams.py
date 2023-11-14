
import Benchmarking
import Training
import QCNN_circuit
import numpy as np
import pennylane as qml
from pennylane.templates.embeddings import AmplitudeEmbedding
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import sin_generator


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

U='U_6'
U_params=10
US=['U_6']
U_num_params = [10]
#US=['U_SU4_no_pooling', 'U_SU4_1D', 'U_9_1D']
#U_num_params = [15, 15, 2]
steps=20
dataset = sin_generator.sin_gen2(10,10,0.024,10,20)
X_train, X_test, Y_train, Y_test = data_load_and_process(dataset)
loss_history, trained_params = Training.circuit_training(X_train, Y_train, U, U_params, steps)

for x in X_test[:1]:
    for U, U_params in zip(US, U_num_params):
        print(U, U_params)
        dev = qml.device('default.qubit', wires = 9)
        @qml.qnode(dev)
        def QCNN(X, params, U, U_params):

            Benchmarking.data_embedding(X)#, embedding_type=embedding_type)
            
            if U == 'U_TTN':
                QCNN_circuit.QCNN_structure(QCNN_circuit.U_TTN, params, U_params)
            elif U == 'U_5':
                QCNN_circuit.QCNN_structure(QCNN_circuit.U_5, params, U_params)
            elif U == 'U_6':
                QCNN_circuit.QCNN_structure(QCNN_circuit.U_6, params, U_params)
            elif U == 'U_9':
                QCNN_circuit.QCNN_structure(QCNN_circuit.U_9, params, U_params)
            elif U == 'U_13':
                QCNN_circuit.QCNN_structure(QCNN_circuit.U_13, params, U_params)
            elif U == 'U_14':
                QCNN_circuit.QCNN_structure(QCNN_circuit.U_14, params, U_params)
            elif U == 'U_15':
                QCNN_circuit.QCNN_structure(QCNN_circuit.U_15, params, U_params)
            elif U == 'U_SO4':
                QCNN_circuit.QCNN_structure(QCNN_circuit.U_SO4, params, U_params)
            elif U == 'U_SU4':
                QCNN_circuit.QCNN_structure(QCNN_circuit.U_SU4, params, U_params)
            elif U == 'U_SU4_no_pooling':
                QCNN_circuit.QCNN_structure_without_pooling(QCNN_circuit.U_SU4, params, U_params)
            elif U == 'U_SU4_1D':
                QCNN_circuit.QCNN_1D_circuit(QCNN_circuit.U_SU4, params, U_params)
            elif U == 'U_9_1D':
                QCNN_circuit.QCNN_1D_circuit(QCNN_circuit.U_9, params, U_params)
            else:
                print("Invalid Unitary Ansatz")
                return False

            result = qml.probs(wires=5)
            return result
        fig, ax = qml.draw_mpl(QCNN)(x, trained_params, U, U_params)

        name=U+'3.png'
        plt.savefig(name)
        #fig.show()
        #plt.ioff()  # Use non-interactive mode.
        #plt.plot([0, 1])  # You won't see the figure yet.
        #plt.show()