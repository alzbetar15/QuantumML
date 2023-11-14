# Implementation of Quantum circuit training procedure
import QCNN_circuit
import Benchmarking
import pennylane as qml
from pennylane import numpy as np
import autograd.numpy as anp
from pennylane.templates.embeddings import AmplitudeEmbedding

# This is an implementation of data_embedding function used for 8 qubits Quantum Convolutional Neural Network (QCNN)
# and Hierarchical Quantum Classifier circuit.




#this is a cross entropy function
def cross_entropy(labels, predictions):
    '''
    Finds cross entropy loss for one example
    ARGS: labels- Y_train and Y_test datasets
          predictions- Ouput from QCNN
    RETURNS: cross entropy loss
    '''
    loss = 0
    for l, p in zip(labels, predictions):
        c_entropy = l * (anp.log(p[l])) + (1 - l) * anp.log(1 - p[1 - l])
        loss = loss + c_entropy
    return -1 * loss

def cost(params, X, Y, U, U_params):
    '''
    computes total cost of QCNN over training set
    ARGS: params- tunable parameters of QCNN
          X - embedded input
          Y - labeled output
          U - Unitaries used in qcnn
          U_params - unitary cirucit tunable parameters
    RETURNS: Total loss 
    '''
    predictions = [QCNN_circuit.QCNN(x, params, U, U_params) for x in X]
    loss = cross_entropy(Y, predictions)
    return loss

# Circuit training parameters
learning_rate = 0.01
batch_size = 25
def circuit_training(X_train, Y_train, U, U_params, steps):
    '''
    trains qcnn on training data
    ARGS: X_train- training data
          Y_train- training labels
          U- circuit architecture of qcnn
          U_params- tunable parameters of qcnn to be learned
    RETURNS: loss history and learned parameters
    '''
    if U == 'U_SU4_no_pooling' or U == 'U_SU4_1D' or U == 'U_9_1D':
        total_params = U_params * 3
    else:
        total_params = U_params * 3 + 2 * 3

    params = np.random.randn(total_params, requires_grad=True)  #randomly initialises circuit parameters
    opt = qml.NesterovMomentumOptimizer(stepsize=learning_rate)  #defines optimizer
    loss_history = []

    for it in range(steps):
        '''
        calculate loss for each epoch- traing set split into
        '''
        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch = [X_train[i] for i in batch_index]
        Y_batch = [Y_train[i] for i in batch_index]
        params, cost_new = opt.step_and_cost(lambda v: cost(v, X_batch, Y_batch, U, U_params),
                                                     params)
        loss_history.append(cost_new)
        if it % 10 == 0:
            print("iteration: ", it, " cost: ", cost_new)


    return loss_history, params


