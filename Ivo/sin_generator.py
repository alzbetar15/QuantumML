import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2664)
import random
from sklearn.preprocessing import normalize

def sin_gen(snr, length): #input snr as power ratio
    data_length = 256
    x = np.linspace(0,2*np.pi,data_length) #create "time series" over period 2pi with data_length points 
    outputs = [] # initialize empty example and label arrays
    labels = []

    for i in range(0,length): #10000 total examples (training and test)

        #f=np.random.randint(1,10)          #random freq and phase to give sin wave
        f=2
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
            output=[float(i)/max(output) for i in output] #normalise data
            label = 1 
        else:       #half of examples only noise
            output = noise 
            #output = [0.00000001] * data_length
            output=[float(i)*0.1/max(output) for i in output] #normalise noise
            label = 0 #label corresponding to just noise is 0
    
        outputs.append(output) #add example to array
        labels.append(label) #add corresponding label to array
    
    dataset = [outputs, labels] #entire dataset contains all examples and their corresponding labels
    return(dataset)

def sin_gen2(snr, length): #input snr as power ratio
    data_length = 256
    x = np.linspace(0,2*np.pi,data_length) #create "time series" over period 2pi with data_length points 
    outputs = [] # initialize empty example and label arrays
    labels = []

    for i in range(0,length): #10000 total examples (training and test)

        #f=2#np.random.randint(1,10)          #random freq and phase to give sin wave
        f=np.random.randint(10,20)
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
            #output=[float(i)/max(output) for i in output] #normalise data
            label = 1 
        else:       #half of examples only noise
            #output = np.sin(10*x) + noise
            output = np.sin((f*x)) + noise
            #output = [0.00000001] * data_length
            #output=[float(i)/max(output) for i in output] #normalise noise
            label = 0 #label corresponding to just noise is 0
    
        outputs.append(output) #add example to array
        labels.append(label) #add corresponding label to array
    
    dataset = [outputs, labels] #entire dataset contains all examples and their corresponding labels
    return(dataset)


def sin_gen3(snr, length): #input snr as power ratio
    data_length = 256
    x = np.linspace(0,2*np.pi,data_length) #create "time series" over period 2pi with data_length points 
    outputs = [] # initialize empty example and label arrays
    labels = []
    #noisefreq = np.random.randint(1,5)
    for i in range(0,length): #10000 total examples (training and test)
        f=2
        #f= np.random.randint(10,20)          #random freq and phase to give sin wave
        #noisefreq = np.random.randint(1,5)
        #noisefreq = 0.024
        noisefreq = 0.5
        phase = np.random.randint(0,256)
        signal = np.sin((f*x)+phase)
        sig_avg_power = np.mean(signal**2)
        #noise_power = sig_avg_power / snr
        if snr == 0:
            noise = [1e-20] * data_length
        else:
            noise_power = sig_avg_power / snr
            noise = np.random.normal(0,np.sqrt(noise_power),data_length)  #normal dist. mean 0, st.dev. given by noise power
            #noise = np.random.multivariate_normal(np.zeros(2), 0.5*np.eye(2), size=data_length).view(np.complex)  
        if i < (length/2): #half of examples to be signal with noise
            output = signal + noise
            #output=[float(i)/max(output) for i in output] #normalise datas
            label = 1 
        else:       #half of examples only noise
            #output = np.sin((noisefreq*x+phase)) + noise
            output = np.sin((noisefreq*x)) + noise
            #output = [0.00000001] * data_length
            #output=[float(i)/max(output) for i in output] #normalise noise
            label = 0 #label corresponding to just noise is 0
    
        outputs.append(output) #add example to array
        labels.append(label) #add corresponding label to array
    
    dataset = [outputs, labels] #entire dataset contains all examples and their corresponding labels
    print('f',f,'noisefreq',noisefreq)
    return(dataset)

def sin_gen3nn(snr, length): #input snr as power ratio
    data_length = 256
    x = np.linspace(0,2*np.pi,data_length) #create "time series" over period 2pi with data_length points 
    outputs = [] # initialize empty example and label arrays
    labels = []
    #noisefreq = np.random.randint(1,5)
    for i in range(0,length): #10000 total examples (training and test)
        f=2
        #f= np.random.randint(10,20)          #random freq and phase to give sin wave
        #noisefreq = np.random.randint(1,5)
        #noisefreq = 0.024
        noisefreq = 0.5
        phase = np.random.randint(0,256)
        signal = np.sin((f*x)+phase)
        sig_avg_power = np.mean(signal**2)
        #noise_power = sig_avg_power / snr
        if snr == 0:
            noise = [1e-20] * data_length
        else:
            noise_power = sig_avg_power / snr
            noise = np.random.normal(0,np.sqrt(noise_power),data_length)  #normal dist. mean 0, st.dev. given by noise power
            #noise = np.random.multivariate_normal(np.zeros(2), 0.5*np.eye(2), size=data_length).view(np.complex)  
        if i < (length/2): #half of examples to be signal with noise
            output = signal
            #output=[float(i)/max(output) for i in output] #normalise datas
            label = 1 
        else:       #half of examples only noise
            #output = np.sin((noisefreq*x+phase)) + noise
            output = np.sin((noisefreq*x))
            #output = [0.00000001] * data_length
            #output=[float(i)/max(output) for i in output] #normalise noise
            label = 0 #label corresponding to just noise is 0
    
        outputs.append(output) #add example to array
        labels.append(label) #add corresponding label to array
    
    dataset = [outputs, labels] #entire dataset contains all examples and their corresponding labels
    print('f',f,'noisefreq',noisefreq)
    return(dataset)


def sin_gen3m(snr, length): #input snr as power ratio
    data_length = 256
    x = np.linspace(0,2*np.pi,data_length) #create "time series" over period 2pi with data_length points 
    outputs = [] # initialize empty example and label arrays
    labels = []
    #noisefreq = np.random.randint(1,5)
    for i in range(0,length): #10000 total examples (training and test)
        #f=2
        f= np.random.randint(1,100)          #random freq and phase to give sin wave
        noisefreq = np.random.randint(1,20)
        #noisefreq = 0.5
        phase = np.random.randint(0,256)
        signal = np.sin((f*x)+phase)
        sig_avg_power = np.mean(signal**2)
        #noise_power = sig_avg_power / snr
        if snr == 0:
            noise = [1e-20] * data_length
        else:
            noise_power = sig_avg_power / snr
            noise = np.random.normal(0,np.sqrt(noise_power),data_length)  #normal dist. mean 0, st.dev. given by noise power
            #noise = np.random.multivariate_normal(np.zeros(2), 0.5*np.eye(2), size=data_length).view(np.complex)  
        if i < (length/2): #half of examples to be signal with noise
            output = signal + noise
            #output=[float(i)/max(output) for i in output] #normalise datas
            label = 1 
        else:       #half of examples only noise
            output = np.sin((noisefreq*x+phase)) + noise
            #output = [0.00000001] * data_length
            #output=[float(i)/max(output) for i in output] #normalise noise
            label = 0 #label corresponding to just noise is 0
    
        outputs.append(output) #add example to array
        labels.append(label) #add corresponding label to array
    
    dataset = [outputs, labels] #entire dataset contains all examples and their corresponding labels
    return(dataset)

def sin_genn(snr, length): #input snr as power ratio
    data_length = 256
    x = np.linspace(0,2*np.pi,data_length) #create "time series" over period 2pi with data_length points 
    outputs = [] # initialize empty example and label arrays
    labels = []
    #noisefreq = np.random.randint(1,5)
    for i in range(0,length): #10000 total examples (training and test)
        f=2
        #f= np.random.randint(10,20)          #random freq and phase to give sin wave
        noisefreq = np.random.randint(1,5)
        noisefreq = 0.5
        phase = np.random.randint(0,256)
        signal = np.sin((f*x)+phase)
        sig_avg_power = np.mean(signal**2)
        #noise_power = sig_avg_power / snr
        if snr == 0:
            noise = [1e-20] * data_length
        else:
            noise_power = sig_avg_power / snr
            noise = np.random.normal(0,np.sqrt(noise_power),data_length)  #normal dist. mean 0, st.dev. given by noise power
            #noise = np.random.multivariate_normal(np.zeros(2), 0.5*np.eye(2), size=data_length).view(np.complex)  
        if i < (length/2): #half of examples to be signal with noise
            output = signal + noise
            #output=[float(i)/max(output) for i in output] #normalise datas
            label = 1 
        else:       #half of examples only noise
            output = noise
            #output = [0.00000001] * data_length
            #output=[float(i)/max(output) for i in output] #normalise noise
            label = 0 #label corresponding to just noise is 0
    
        outputs.append(output) #add example to array
        labels.append(label) #add corresponding label to array
    
    dataset = [outputs, labels] #entire dataset contains all examples and their corresponding labels
    return(dataset)


def sin_gen4(snr, length): #input snr as power ratio
    data_length = 256
    x = np.linspace(0,2*np.pi,data_length) #create "time series" over period 2pi with data_length points 
    outputs = [] # initialize empty example and label arrays
    labels = []
    #noisefreq = np.random.randint(1,5)
    for i in range(0,length): #10000 total examples (training and test)
        #f=2 
        f= np.random.randint(10,20)          #random freq and phase to give sin wave
        noisefreq = np.random.randint(1,5)
        #noisefreq = 0.5
        phase = np.random.randint(0,256)
        signal = np.sin((f*x)+phase)
        sig_avg_power = np.mean(signal**2)
        #noise_power = sig_avg_power / snr
        if snr == 0:
            noise = [1e-20] * data_length
        else:
            noise_power = sig_avg_power / snr
            noise = np.random.normal(0,np.sqrt(noise_power),data_length)  #normal dist. mean 0, st.dev. given by noise power
            #noise = np.random.multivariate_normal(np.zeros(2), 0.5*np.eye(2), size=data_length).view(np.complex)  
        if i < (length/2): #half of examples to be signal with noise
            output = signal + noise
            #output=[float(i)/max(output) for i in output] #normalise datas
            label = 1 
        else:       #half of examples only noise
            output = np.sin((noisefreq*x+phase)) + noise
            #output = [0.00000001] * data_length
            #output=[float(i)/max(output) for i in output] #normalise noise
            label = 0 #label corresponding to just noise is 0
    
        output = output/np.sqrt(np.sum(np.abs(output)**2))
        outputs.append(output) #add example to array
        labels.append(label) #add corresponding label to array
    
    dataset = [outputs, labels] #entire dataset contains all examples and their corresponding labels
    return(dataset)

#reinitailize data, 


def sin_gen3o(snr, length, noisefreq): #input snr as power ratio
    data_length = 256
    x = np.linspace(0,2*np.pi,data_length) #create "time series" over period 2pi with data_length points 
    outputs = [] # initialize empty example and label arrays
    labels = []

    for i in range(0,length): #10000 total examples (training and test)

        #f=np.random.randint(10,20)          #random freq and phase to give sin wave
        f=2
        #f=10
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
            #output=[float(i)/max(output) for i in output] #normalise datas
            label = 1 
        else:       #half of examples only noise
            output = np.sin((noisefreq*x)+ phase) + noise
            #output = [0.00000001] * data_length
            #output=[float(i)/max(output) for i in output] #normalise noise
            label = 0 #label corresponding to just noise is 0
    
        outputs.append(output) #add example to array
        labels.append(label) #add corresponding label to array
    
    dataset = [outputs, labels] #entire dataset contains all examples and their corresponding labels
    return(dataset)

def sin_gen3o10(snr, length, noisefreq): #input snr as power ratio
    data_length = 256
    x = np.linspace(0,2*np.pi,data_length) #create "time series" over period 2pi with data_length points 
    outputs = [] # initialize empty example and label arrays
    labels = []

    for i in range(0,length): #10000 total examples (training and test)

        #f=np.random.randint(10,20)          #random freq and phase to give sin wave
        #f=2
        f=10
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
            #output=[float(i)/max(output) for i in output] #normalise datas
            label = 1 
        else:       #half of examples only noise
            output = np.sin((noisefreq*x)+ phase) + noise
            #output = [0.00000001] * data_length
            #output=[float(i)/max(output) for i in output] #normalise noise
            label = 0 #label corresponding to just noise is 0
    
        outputs.append(output) #add example to array
        labels.append(label) #add corresponding label to array
    
    dataset = [outputs, labels] #entire dataset contains all examples and their corresponding labels
    return(dataset)