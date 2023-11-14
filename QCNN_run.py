import Benchmarking
#import sin_generator
import sin_data_generator
import Quantum_Data
import numpy as np
import datetime
import os
"""
Here are possible combinations of benchmarking user could try.
Unitaries: ['U_TTN', 'U_5', 'U_6', 'U_9', 'U_13', 'U_14', 'U_15', 'U_SO4', 'U_SU4', 'U_SU4_no_pooling', 'U_SU4_1D', 'U_9_1D']
U_num_params: [2, 10, 10, 2, 6, 6, 4, 6, 15, 15, 15, 2]

circuit: 'QCNN' 
cost_fn: 'cross_entropy'
Note: when using 'mse' as cost_fn binary="True" is recommended, when using 'cross_entropy' as cost_fn must be binary="False".
"""
#Declaring constants
EPOCHS = 161
#This is quite high
#LEARNING_RATE = 0.01
#BATCH_SIZE = 25
print("HERE")

if __name__ == "__main__":
    #Unitaries = ['U_13', 'U_14', 'U_15', 'U_SO4']
    print("HEre")
    Unitaries=['U_6']#, 'U_9', 'U_13', 'U_14', 'U_15', 'U_SO4', 'U_SU4', 'U_SU4_no_pooling', 'U_SU4_1D', 'U_9_1D']
    #U_num_params = [6, 6, 4, 6]
    #Unitaries=['U_14']
    U_num_params = [10]#, 10, 2, 6, 6, 4, 6, 15, 15, 15, 2]
    #dataset = sin_generator.sin_gen(10,10000)

    binary = False
    #cost_fn = 'cross_entropy'
    #freqs = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    #freqs = [1,3,5,7,9,11]
    #filename='Reg Results'
    #Runs the Benchmarking function
    #
    filename="Result"+str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')

    #fname =r'qdata_1009.txt'
    #fname =r'qdata_10000_snr_50.75.txt'
    #fname =r'qdata_10000_0.1.txt'
    #fname =r'Quantum_data\Qdata30.32021.txt'
    fname =r'Quantum_Data\NoPauil\Qdata20.31011.txt'
    testName ='G30.31011'
    with open(fname) as f:
        lines = f.readlines()
        filedata=''.join(lines)
        #data handeling
        allstates=filedata.split(')], ')
        print(len(allstates))
        #print(allstates[2])
        #data handeling of the labels which are at the end of the file
        #try:
        labels=allstates[1]
        labels=labels[:-1]
        labels=labels.strip(']')
        labels=labels.strip('[')
        labels=labels.split(',')
        #print("labels",labels)
        labels=[eval(i) for i in labels]
        '''except:
            print("Data labels type 2")
            labels=allstates[2]
            #labels=labels[:-1]
            labels=labels.strip(']')
            labels=labels.strip('[')
            #labels=labels.strip(' ')
            labels=labels.split(',')'''
            
            
        #print(labels)
            
        #are these labels literally like true/false
        #print(labels)
        #print(filedata[len(labels):])
        #filedata= filedata[:len(labels)]
        #filedata = filedata.replace('[array(','')

        #get sin data?
        filedata = filedata.split('),')
        #print(filedata)
        print(len(filedata))
        counter1=0
        filedata_new=[]
        for data in filedata:
            #print(data)
            data = data.replace('[array(','')
            data = data.replace('rray([','')
            data=data[2:-1]
            data=data.replace('\n       ','')
            data=data.replace('  ','')
            data=data.replace(' ','')
            data=data.split(',')
            #print("data length",len(data))

            if counter1 == len(filedata)-1:  
                data=';'.join(data)
                data=data.split(')')
                data=data[0]
                data=data[:-1]
                data=data.split(';')
                #print('new data length',len(data))
            #length of the file
            if len(data) == 256:
                complexData=[]
                for i in data:
                    h=np.complex128(i)
                    complexData.append(h)
                counter1=counter1+1
            complexData=np.array(complexData,dtype=complex)
            filedata_new.append(complexData)

    print(len(filedata_new))
    print(type(filedata_new[0]),labels[0])
    print('counter1:',counter1)
    dataset=[filedata_new,labels]

    #snr? Frequences
    #was 0.1
    freqs=[0.3]
    print('data read')
    print(freqs)
    print("test",testName)
    for p in freqs:
        #dataset = Quantum_Data.quantum_data(p)
        #dataset = Quantum_Data.sin_gen(p, 10000)
        print("HERE")
        #Step hyper parameters set up here
        Benchmarking.Benchmarking(dataset, Unitaries, U_num_params, filename,testName ,circuit='QCNN', steps = EPOCHS, snr=p, binary=binary)


        #train pnoise network with gnoise
