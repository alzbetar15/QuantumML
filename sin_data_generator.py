import numpy as np
import matplotlib.pyplot as plt
import statistics as stat

def sin_gen(snr,freq,length):
    data_length = 256
    #Our inital values for sin?
    #why stop at 2*np.pi
    
    #plt.plot(x)
    #plt.show()
    #arrays for the output
    outputs = []
    labels = []
    for i in range(0,length):
        #randomly decides if the data will be signal or noise
        label=np.random.randint(0,2)
        #print(label)
        #0 represents nosie
        if label == 0:
            x = np.linspace(0,0,data_length)
            labels.append(label)
            sigma = snr
            noise =  np.random.normal(0, sigma, 256)
            output = x+noise


        #1 represents signal
        elif label == 1:
            x = np.linspace(0,2*np.pi,data_length)
            labels.append(label)
            #frequency of the sin wave
            #frequency = np.random.randint(40,120)
            #print(freq)
            frequency = np.random.randint(freq[0],freq[1])
            phase = np.random.randint(0,256)
            signal = np.sin((frequency*x)+phase)
            #sigma = stat.variance(signal)

            #previous way of calculating the variance
            #sig_avg_power = np.mean(signal**2)
            #noise_power = sig_avg_power / snr
            #print("there var",np.sqrt(noise_power))
            #sigma = np.sqrt(noise_power)

            sigma =snr
            #print(sigma)
            #sigma is the variance
            #sigma = 10
            noise =  np.random.normal(0, sigma, 256)
            output = signal + noise
        else:
            print("ERROR")
        output = output/np.sqrt(np.sum(np.abs(output)**2))
        outputs.append(output)
        #to see the actual graphs, remove for 
        #print(label)
        #plt.plot(output)
        #plt.show()
    #print(labels)
    dataset = [outputs, labels]
    return(dataset)
if __name__ == "__main__":
    sin_gen(0.5,[10,30],10)