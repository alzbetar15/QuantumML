import Benchmarking
import sin_generator


"""
Here are possible combinations of benchmarking user could try.
Unitaries: ['U_TTN', 'U_5', 'U_6', 'U_9', 'U_13', 'U_14', 'U_15', 'U_SO4', 'U_SU4', 'U_SU4_no_pooling', 'U_SU4_1D', 'U_9_1D']
U_num_params: [2, 10, 10, 2, 6, 6, 4, 6, 15, 15, 15, 2]

circuit: 'QCNN' 
cost_fn: 'cross_entropy'
Note: when using 'mse' as cost_fn binary="True" is recommended, when using 'cross_entropy' as cost_fn must be binary="False".
"""

#Unitaries = ['U_13', 'U_14', 'U_15', 'U_SO4']

Unitaries=['U_SU4']#, 'U_9', 'U_13', 'U_14', 'U_15', 'U_SO4', 'U_SU4', 'U_SU4_no_pooling', 'U_SU4_1D', 'U_9_1D']
#U_num_params = [6, 6, 4, 6]

U_num_params = [15]#, 10, 2, 6, 6, 4, 6, 15, 15, 15, 2]
#dataset = sin_generator.sin_gen(10,10000)

binary = False
#cost_fn = 'cross_entropy'
#freqs = [0.024]#,0.026,0.028,0.03]
filename='xtra last test'
#Runs the Benchmarking function
snrs=[100]
for snr in snrs:
    dataset = sin_generator.sin_gen5(snr,0.024,5,20)
    Benchmarking.Benchmarking(dataset, Unitaries, U_num_params, filename, circuit='QCNN', steps = 50, snr=snr,f1=2, f2=20, binary=binary)



