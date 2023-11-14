import Benchmarking
import sin_generator


Unitaries=['U_6']#, 'U_9', 'U_13', 'U_14', 'U_15', 'U_SO4', 'U_SU4', 'U_SU4_no_pooling', 'U_SU4_1D', 'U_9_1D']
#U_num_params = [6, 6, 4, 6]

U_num_params = [10]#, 10, 2, 6, 6, 4, 6, 15, 15, 15, 2]
#dataset = sin_generator.sin_gen(10,10000)

binary = False
#cost_fn = 'cross_entropy'
freqs = [0.024,0.026,0.028,0.03]
freqs = [0.026,0.028,0.03]

filename='Results_modnoisecomplx.txt'
#Runs the Benchmarking function
for  fr in freqs:
    dataset = sin_generator.sin_gen3(1,10000,fr)
    Benchmarking.Benchmarking(dataset, Unitaries, U_num_params, filename, circuit='QCNN', steps = 200, snr=1, binary=binary)