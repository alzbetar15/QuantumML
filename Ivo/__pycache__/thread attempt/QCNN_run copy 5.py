import Benchmarking
import sin_generator
import Quantum_Data

"""
Here are possible combinations of benchmarking user could try.
Unitaries: ['U_TTN', 'U_5', 'U_6', 'U_9', 'U_13', 'U_14', 'U_15', 'U_SO4', 'U_SU4', 'U_SU4_no_pooling', 'U_SU4_1D', 'U_9_1D']
U_num_params: [2, 10, 10, 2, 6, 6, 4, 6, 15, 15, 15, 2]

circuit: 'QCNN' 
cost_fn: 'cross_entropy'
Note: when using 'mse' as cost_fn binary="True" is recommended, when using 'cross_entropy' as cost_fn must be binary="False".
"""

#Unitaries = ['U_13', 'U_14', 'U_15', 'U_SO4']

Unitaries=['U_14']# 'U_15', 'U_SO4', 'U_SU4', 'U_SU4_no_pooling', 'U_SU4_1D', 'U_9_1D']
#U_num_params = [6, 6, 4, 6]

U_num_params = [6]#, 6, 6, 4, 6, 15, 15, 15, 2]
#dataset = sin_generator.sin_gen(10,10000)

binary = False
#cost_fn = 'cross_entropy'

snrs=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
#freqs = [1,3,5,7,9,11]
filename='Reg Results U_13'
filename='/Final_Results/Reg Results U_14'
#Runs the Benchmarking function
for p in snrs:
    
    #dataset = Quantum_Data.quantum_data(p)
    dataset = sin_generator.sin_gen3(p, 10000)
    Benchmarking.Benchmarking(dataset, Unitaries, U_num_params, filename, circuit='QCNN', steps = 2000, snr=p, binary=binary)



