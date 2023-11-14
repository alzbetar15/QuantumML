import Benchmarking_pn_gn
import sin_generator
import Quantum_Data
import numpy as np

"""
Here are possible combinations of benchmarking user could try.
Unitaries: ['U_TTN', 'U_5', 'U_6', 'U_9', 'U_13', 'U_14', 'U_15', 'U_SO4', 'U_SU4', 'U_SU4_no_pooling', 'U_SU4_1D', 'U_9_1D']
U_num_params: [2, 10, 10, 2, 6, 6, 4, 6, 15, 15, 15, 2]

circuit: 'QCNN' 
cost_fn: 'cross_entropy'
Note: when using 'mse' as cost_fn binary="True" is recommended, when using 'cross_entropy' as cost_fn must be binary="False".
"""

#Unitaries = ['U_13', 'U_14', 'U_15', 'U_SO4']

Unitaries=['U_6']#, 'U_9', 'U_13', 'U_14', 'U_15', 'U_SO4', 'U_SU4', 'U_SU4_no_pooling', 'U_SU4_1D', 'U_9_1D']
#U_num_params = [6, 6, 4, 6]

U_num_params = [10]#, 10, 2, 6, 6, 4, 6, 15, 15, 15, 2]
#dataset = sin_generator.sin_gen(10,10000)

binary = False
#cost_fn = 'cross_entropy'
#freqs = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
#freqs = [1,3,5,7,9,11]
#filename='Reg Results'
#Runs the Benchmarking function

filename='/Quantum_Results/Quantum Results U_6 with pgnoise 0.9 convergence test new'

fname =r'C:\Users\ivoll\OneDrive\Documents\qml_sin-3 copy\Quantum_data\qdata_10000_0.9.txt'
with open(fname) as f:
    lines = f.readlines()
    all=''.join(lines)
    allstates=all.split(')], ')
    print(len(allstates))

    labels=allstates[1]
    #print(labels)
    labels=labels[:-1]
    labels=labels.strip(']')
    labels=labels.strip('[')
    labels=labels.split(',')
    labels=[eval(i) for i in labels]
    #print(labels)
    #print(all[len(labels):])
    #all= all[:len(labels)]
    #all = all.replace('[array(','')
    all = all.split('),')
    #print(all)
    print(len(all))
    jj=0
    all_new=[]
    for item in all:
    #    print(item)
        item = item.replace('[array(','')
        item = item.replace('rray([','')
        item=item[2:-1]
        #print(item)
        item=item.replace('\n       ','')
        item=item.replace('  ','')
        item=item.replace(' ','')
        item=item.split(',')
        #print(item)
        #print(len(item))
        #print(jj,len(item))

        if jj == 9999:
            
            item=';'.join(item)
            #print(item)
            item=item.split(')')
            item=item[0]
            item=item[:-1]
            #print(item)
            item=item.split(';')
            print('testsdvhfsuyds',len(item))
        if len(item) == 256:
            gg=[]
            for i in item:
                #try:
                h=np.complex128(i)
                gg.append(h)
                #gg= np.append(gg,h)
                #except:
                #print(i)
                #print(i)
            
        #    break
            jj=jj+1
        #break
        gg=np.array(gg,dtype=complex)
        #print(type(gg))
        all_new.append(gg)
print(len(all_new))
print(type(all_new[0]),labels[0])
#output=[all_new,labels]
#print(output)
print('jj:',jj)

    #for line in lines:
    #    print(line)
dataset=[all_new,labels]
#print(len(dataset))

freqs=[0.9]
print('data read')
for p in freqs:
    #dataset = Quantum_Data.quantum_data(p)
    #dataset = Quantum_Data.sin_gen(p, 10000)
    Benchmarking_pn_gn.Benchmarking(dataset, Unitaries, U_num_params, filename, circuit='QCNN', steps = 200, snr=p, binary=binary)


    #train pnoise network with gnoise
