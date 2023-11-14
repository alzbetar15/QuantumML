import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2664)
import random
from sklearn.preprocessing import normalize

'''
singen generates 50/50 split of signal and noise examples
'''
def sin_gen(snr, length): #input snr as power ratio and length of dataset
    data_length = 256
    x = np.linspace(0,2*np.pi,data_length) #create "time series" over period 2pi with data_length points 
    outputs = [] # initialize empty example and label arrays
    labels = []

    for i in range(0,length): #10000 total examples (training and test)

        f=np.random.randint(10,20)          #random freq and phase to give sin wave
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
            output = signal + noise

            label = 1 
        else:       #half of examples only noise
            output = noise 
            label = 0 #label corresponding to just noise is 0
    
        outputs.append(output) #add example to array
        labels.append(label) #add corresponding label to array
    
    dataset = [outputs, labels] #entire dataset contains all examples and their corresponding labels
    return(dataset)


'''
singen2 creates datatset where signal examples contain two examples of different frequency stitches together
'''
def sin_gen2(snr, length, noisefreq, f1, f2): 

    data_length = 256
    x = np.linspace(0,2*np.pi,data_length) #create "time series" over period 2pi with data_length points 
    outputs = [] # initialize empty example and label arrays
    labels = []

    for i in range(0,length): #10000 total examples (training and test)
        
        signal1 = np.sin(f1*x)
        signal2 = np.sin(f2*x)
        sig_avg_power1 = np.mean(signal1**2)
        sig_avg_power2 = np.mean(signal2**2)
        #noise_power = sig_avg_power / snr
        if snr == 0:
            noise1 = [1e-20] * data_length
            noise2 = [1e-20] * data_length
        else:
            noise_power1 = sig_avg_power1 / snr
            noise_power2 = sig_avg_power2 / snr
            noise1 = np.random.normal(0,np.sqrt(noise_power1),data_length)   #normal dist. mean 0, st.dev. given by noise power
            noise2 = np.random.normal(0,np.sqrt(noise_power2),data_length)
        if i < (length/2): #half of examples to be signal with noise
            output1 = signal1 + noise1
            output2 = signal2 + noise2
            output = np.concatenate((output1,output2))
            label = 1   
        else:       #half of examples only noise
            output1 = np.sin((noisefreq*x)) + noise1
            output2 = np.sin((noisefreq*x)) + noise2
            output = np.concatenate((output1,output2))
            label = 0 #label corresponding to just noise is 0
    
        outputs.append(output) #add example to array
        labels.append(label) #add corresponding label to array
    
    dataset = [outputs, labels] #entire dataset contains all examples and their corresponding labels
    return(dataset)

'''
singen3 generates 50/50 split of signal and noise (noise examples now a modulated sinwave with low freq)
'''
def sin_gen3(snr, length, noisefreq): #input snr as power ratio
    data_length = 256
    x = np.linspace(0,2*np.pi,data_length) #create "time series" over period 2pi with data_length points 
    outputs = [] # initialize empty example and label arrays
    labels = []

    for i in range(0,length): #10000 total examples (training and test)

        f=np.random.randint(10,20)          #random freq and phase to give sin wave
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
            output = signal + noise
            output=[float(i)/max(output) for i in output] #normalise datas
            label = 1 
        else:       #half of examples only noise
            output = np.sin((noisefreq*x)) + noise
            #output = [0.00000001] * data_length
            output=[float(i)/max(output) for i in output] #normalise noise
            label = 0 #label corresponding to just noise is 0
    
        outputs.append(output) #add example to array
        labels.append(label) #add corresponding label to array
    
    dataset = [outputs, labels] #entire dataset contains all examples and their corresponding labels
    return(dataset)


'''
singen4 is similar to singen2 but uses the modulated sin wave of low freq as noise examples
'''
def sin_gen4(snr, length, noisefreq, f1, f2): #input snr as power ratio
    data_length = 256
    x = np.linspace(0,2*np.pi,data_length) #create "time series" over period 2pi with data_length points 
    outputs = [] # initialize empty example and label arrays
    labels = []

    for i in range(0,length): #10000 total examples (training and test)
        
        signal1 = np.sin(f1*x)
        signal2 = np.sin(f2*x)
        sig_avg_power1 = np.mean(signal1**2)
        sig_avg_power2 = np.mean(signal2**2)
        #noise_power = sig_avg_power / snr
        if snr == 0:
            noise1 = [1e-20] * data_length
            noise2 = [1e-20] * data_length
        else:
            noise_power1 = sig_avg_power1 / snr
            noise_power2 = sig_avg_power2 / snr
            noise1 = np.random.normal(0,np.sqrt(noise_power1),data_length)   #normal dist. mean 0, st.dev. given by noise power
            noise2 = np.random.normal(0,np.sqrt(noise_power2),data_length)
        if i < (length/2): #half of examples to be signal with noise
            output1 = signal1 + noise1
            output2 = signal2 + noise2
            output = np.concatenate((output1,output2))
            #output=[float(i)/max(output) for i in output]
            label = 1   
        else:       #half of examples only noise
            output1 = np.sin((noisefreq*x)) + noise1
            output2 = np.sin((noisefreq*x)) + noise2
            output = np.concatenate((output1,output2))
            #output = [0.00000001] * data_length
            #output=[float(i)/max(output) for i in output] #normalise noise
            label = 0 #label corresponding to just noise is 0
    
        outputs.append(output) #add example to array
        labels.append(label) #add corresponding label to array
    
    dataset = [outputs, labels] #entire dataset contains all examples and their corresponding labels
    return(dataset)


'''
singen5 used for extra qubit testing. Generates training set with two examples stitched together and test set of one signal per example
'''
def sin_gen5(snr, noisefreq, f1, f2): #input snr as power ratio
    data_length = 256
    x = np.linspace(0,2*np.pi,data_length) #create "time series" over period 2pi with data_length points 
    x_train = [] # initialize empty example and label arrays
    y_train = []
    x_test = []
    y_test = []

    for i in range(0,8000): #80% training
        
        signal1 = np.sin(f1*x)
        signal2 = np.sin(f2*x)
        sig_avg_power1 = np.mean(signal1**2)
        sig_avg_power2 = np.mean(signal2**2)
        #noise_power = sig_avg_power / snr
        if snr == 0:
            noise1 = [1e-20] * data_length
            noise2 = [1e-20] * data_length
        else:
            noise_power1 = sig_avg_power1 / snr
            noise_power2 = sig_avg_power2 / snr
            noise1 = np.random.normal(0,np.sqrt(noise_power1),data_length)   #normal dist. mean 0, st.dev. given by noise power
            noise2 = np.random.normal(0,np.sqrt(noise_power2),data_length)
        if i < (4000): #half of examples to be signal with noise
            output1 = signal1 + noise1
            output2 = signal2 + noise2
            output = np.concatenate((output1,output2))
            #output=[float(i)/max(output) for i in output]
            label = 1   
        else:       #half of examples only noise
            output1 = np.sin(noisefreq*x) + noise1
            output2 = np.sin(noisefreq*x) + noise2
            output = np.concatenate((output1,output2))
            #output = [0.00000001] * data_length
            #output=[float(i)/max(output) for i in output] #normalise noise
            label = 0 #label corresponding to just noise is 0
    
        x_train.append(output) #add example to array
        y_train.append(label) #add corresponding label to array
    
    for i in range(0,2000): #20% test 
        
        signal1 = np.sin(f1*x)
        signal2 = x*0
        sig_avg_power1 = np.mean(signal1**2)
                #noise_power = sig_avg_power / snr
        if snr == 0:
            noise1 = [1e-20] * data_length
        else:
            noise_power1 = sig_avg_power1 / snr
            noise1 = np.random.normal(0,np.sqrt(noise_power1),data_length)   #normal dist. mean 0, st.dev. given by noise power
        if i < (1000): #half of examples to be signal with noise
            output1 = signal1 + noise1
            output2 = signal2 
            output = np.concatenate((output1,output2))
            #output=[float(i)/max(output) for i in output]
            label = 1   
        else:       #half of examples only noise
            output1 = np.sin(noisefreq*x) + noise1
            output2 = x*0
            output = np.concatenate((output1,output2))
            #output = [0.00000001] * data_length
            #output=[float(i)/max(output) for i in output] #normalise noise
            label = 0 #label corresponding to just noise is 0
    
        x_test.append(output) #add example to array
        y_test.append(label) #add corresponding label to array


    combined = list(zip(x_train, y_train))
    random.shuffle(combined)
    x_train, y_train = zip(*combined)

    combined2 = list(zip(x_test, y_test))
    random.shuffle(combined2)
    x_test, y_test = zip(*combined2)


    return(x_train,x_test,y_train,y_test)