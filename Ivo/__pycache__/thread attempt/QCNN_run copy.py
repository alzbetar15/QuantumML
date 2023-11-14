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

Unitaries = ['U_6']

#Unitaries=['U_TTN', 'U_5', 'U_6', 'U_9']#, 'U_13', 'U_14', 'U_15', 'U_SO4', 'U_SU4', 'U_SU4_no_pooling', 'U_SU4_1D', 'U_9_1D']
U_num_params = [10]

#[2, 10, 10, 2, 6, 6, 4, 6, 15, 15, 15, 2]

for i in range (0, 20):
    dataset = sin_generator.sin_gen(i,10000,0.5)
    binary = False
    #cost_fn = 'cross_entropy'
    
    Benchmarking.Benchmarking(dataset, Unitaries, U_num_params, circuit='QCNN', steps = 200, snr=i, binary=binary)