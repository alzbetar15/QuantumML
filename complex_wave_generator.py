import numpy as np
import matplotlib.pyplot as plt
import statistics as stat

def complex_wave_gen(snr: float,freq: list,length: int) -> list:
    data_length = 256

    #empty arrays for the output
    outputs = []
    labels = []

    for i in range(0,length):
        #randomly decides if the data will be signal or noise
        label=np.random.randint(0,2)
        #label = (number + 1j*number)
        #print(label)
    
        #0 represents noise
        if label == 0:
            x = np.linspace(0,0,data_length)
            labels.append(label)
            sigma = snr
            noise =  np.random.normal(0, sigma, data_length)
            noise_complex = np.random.normal(0, sigma, data_length)
            output = x + noise + 1j*noise_complex


        #1 represents signal
        elif label == 1:
            #randomly generates a complex signal of a given frequency
            x = np.linspace(0,2*np.pi,data_length)
            labels.append(label)
            frequency = np.random.randint(freq[0],freq[1])
            phase = np.random.randint(0,256)
            signal = np.exp(1j*((frequency*x)+phase))
            #sigma = stat.variance(signal)

            #previous way of calculating the variance
            #sig_avg_power = np.mean(signal**2)
            #noise_power = sig_avg_power / snr
            #print("there var",np.sqrt(noise_power))
            #sigma = np.sqrt(noise_power)

            #sigma is the variance
            sigma = snr
            noise =  np.random.normal(0, sigma, 256)
            noise_complex = np.random.normal(0, sigma, data_length)
            output = signal + noise + 1j*noise_complex
            
        else:
            print("ERROR")

        output = output/np.sqrt(np.sum(np.abs(output)**2))
        outputs.append(output)
        #to see the actual graphs, remove for 
        #print(label)
        #plt.plot(output)
        #plt.show()
    
    dataset = [outputs, labels]
    return(dataset)


if __name__ == "__main__":
    complex_wave_gen(0.5,[10,30],10)