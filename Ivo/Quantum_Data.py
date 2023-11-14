import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2664)
import random
from sklearn.preprocessing import normalize

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer, IBMQ
from qiskit.quantum_info import Pauli
from qiskit.tools.visualization import circuit_drawer
import itertools as it
#import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import Benchmarking

#matplotlib.rcParams['mathtext.fontset'] = 'stix'
#matplotlib.rcParams['font.family'] = 'STIXGeneral'
width=0.75
color='black'
fontsize=28
ticksize=22
figsize=(14,10)


def sin_gen(snr, length): #input snr as power ratio
    data_length = 256
    x = np.linspace(0,2*np.pi,data_length) #create "time series" over period 2pi with data_length points 
    outputs = [] # initialize empty example and label arrays
    labels = []

    for i in range(0,length): #10000 total examples (training and test)

        f=2         #random freq and phase to give sin wave
        fnoise = 0.5
        phase = np.random.randint(0,256)
        signal = np.sin((f*x)+phase)
        sig_avg_power = np.mean(signal**2)
        #noise_power = sig_avg_power / snr
        if snr == 0:
            noise = [1e-20] * data_length
        else:
            noise_power = sig_avg_power / snr
            noise = np.random.normal(0,np.sqrt(noise_power),data_length)   #normal dist. mean 0, st.dev. given by noise power
        
        if i < (length/2): #half of examples to be signal with noise
            output = signal
            #output=[float(i)/max(output) for i in output] #normalise datas
            label = 1 
        else:       #half of examples only noise
            output = np.sin((fnoise*x)+phase)
            #output = noise
            #output = [0.00000001] * data_length
            #output=[float(i)/max(output) for i in output] #normalise noise
            label = 0 #label corresponding to just noise is 0
    
        outputs.append(output) #add example to array
        labels.append(label) #add corresponding label to array
    
    dataset = [outputs, labels] #entire dataset contains all examples and their corresponding labels
    return(dataset)


def depolarisation_channel(circ, qreg, p, wrap=False, inverse=False, label='depol_channel'):
    """
    Given a quantum circuit, applies a depoloarisation channel to a given register with probability p.
    
    circ:    Qiskit circuit object.
    qreg:    Qiskit circuit quantum register object.
    p:       Probability of applying Pauli operation to a given qubit, otherwise apply identity (float).
    wrap:    Wrap operation into a single gate (bool) default=False.
    inverse: Apply the inverse operation (bool) default=False.
    label:   Name given to wrapped operation (str) default='depol_channel'.
    """
    
    n = qreg.size

    if inverse:
        wrap = True

    if wrap:
        qreg = QuantumRegister(n, 'q_reg')
        circ = QuantumCircuit(qreg)

    num_terms = 4**n
    max_param = num_terms / (num_terms - 1)

    if p < 0 or p > max_param:
        raise NoiseError("Depolarizing parameter must be in between 0 "
                         "and {}.".format(max_param))

    prob_iden = 1 - p / max_param
    prob_pauli = p / num_terms
    probs = [prob_iden] + (num_terms - 1) * [prob_pauli]

    paulis = [Pauli("".join(tup)) for tup in it.product(['I', 'X', 'Y', 'Z'], repeat=n)]
    #print(paulis)
    gates_ind = np.random.choice(num_terms, p=probs, size=1)
    #print('gi',gates_ind,p)
    gates_ind = gates_ind[0]
    #gates = paulis[gates_ind]
    #print(gates_ind)
    gates = paulis[gates_ind]
    #print(gates)
    #print(gates,qreg[:])
    #jj=0
    #for gate in gates:
    #    circ.append(gate, [qreg[jj]])
    #    jj=jj+1
    
    circ.append(gates, qreg[:])

    if wrap:
        circ = circ.to_gate()
        circ.label = label

    if inverse:
        circ = circ.inverse()
        circ.label = label+'†'

    return circ

data=sin_gen(5, 10000)
qdata=[]


def quantum_data(p):

    #p = 0.8
    n = 8
    print('generating data')
    data=sin_gen(5, 10000)
    print('generated data')

    outputs=[] #add example to array
    labels=[] #add corresponding label to array
    f = open('Quantum_data/qdata_10000_'+str(p)+'.txt', 'a')
    for i in range(0,len(data[0])):
        #print(i)
        if i%100 == 0:
            print(i)
        wave=data[0][i]
        wave = wave/np.sqrt(np.sum(np.abs(wave)**2))
        #print(wave)
        label=data[1][i]
        q_reg = QuantumRegister(n, 'q_reg')
        circ = QuantumCircuit(q_reg)
        circ.initialize(wave, q_reg)
        backend = Aer.get_backend('statevector_simulator')
        circ = depolarisation_channel(circ, q_reg, p)
        job = execute(circ, backend)
        result = job.result()
        out_state = np.array(result.get_statevector(circ, decimals=5))
        labels.append(label)
        outputs.append(out_state)
        
    f_out=[outputs,labels]
    #np.savetxt('outfile.txt', array.view(float))
    f.write(str(f_out))
    f.close()

    return f_out

        #fig = plt.figure(figsize=figsize)
        #ax = fig.add_subplot(111)
        #ax.plot(wave.real, color='black', label='Real')
        #ax.plot(wave.imag, color='grey', label='Imaginary')

        #ax.tick_params(axis='both', labelsize=ticksize)
        #ax.set_xlabel('State', fontsize=fontsize)
        #ax.set_ylabel('Amplitude', fontsize=fontsize)

        #leg = ax.legend(fontsize=3*fontsize//4)
        #plt.savefig('Quantum_data1/new run '+str(i)+'input waveform '+str(p)+'.png')

        #fig = plt.figure(figsize=figsize)
        #ax = fig.add_subplot(111)
        #ax.plot(out_state.real, color='black', label='Real')
        #ax.plot(out_state.imag, color='grey', label='Imaginary')

        #ax.tick_params(axis='both', labelsize=ticksize)
        #ax.set_xlabel('State', fontsize=fontsize)
        #ax.set_ylabel('Amplitude', fontsize=fontsize)

        #leg = ax.legend(fontsize=3*fontsize//4)

        #plt.show()
        #plt.savefig('Quantum_data1/new run '+str(i)+'output waveform '+str(p)+'.png')

        #circ.draw('mpl')
        #circuit_drawer(circ, output='mpl', plot_barriers=False)
        #circ.draw("mpl", cregbundle=False, initial_state=True, reverse_bits=True)
        #circ.decompose(reps=1).draw('mpl')
        #plt.savefig('Quantum_data1/'+str(i)+'circuit other method E'+str(p)+'.png')
        #break
    
    
    #outputs.append(out_state)
    




    


