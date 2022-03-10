import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (9, 7)

sampFreq, sound = wavfile.read('feel_cut.wav')

sound = sound / 2.0**15

length_in_s = sound.shape[0] / sampFreq
print(length_in_s)

"""
plt.subplot(2,1,1)
plt.plot(sound[:,0], 'r')
plt.xlabel("left channel, sample #")
plt.subplot(2,1,2)
plt.plot(sound[:,1], 'b')
plt.xlabel("right channel, sample #")
plt.tight_layout()
plt.show()
"""

signal = sound[:,0]
time = np.arange(sound.shape[0]) / sound.shape[0] * length_in_s
"""
xoff = 200000
plt.plot(time[xoff+6000:xoff+7000], signal[xoff+6000:xoff+7000])
plt.xlabel("time, s")
plt.ylabel("Signal, relative units")
plt.show()
"""

fft_spectrum = np.fft.rfft(signal)
freq = np.fft.rfftfreq(signal.size, d=1./sampFreq)

overtones = [0.25,0.5,1.0,2.0,4.0]

length = len(fft_spectrum)

new_arr = fft_spectrum.copy()

twelfth = 2**(1/12)

for i,f in enumerate(freq):
    #for x in overtones:
    #    if f > 710*x and f < 770*x:# (1)
    #        fft_spectrum[i] = fft_spectrum[i]*0.01
    if f<20 or f>15000:
        continue
    ind=i/twelfth
    # approximating values from neighbors
    new_arr[i] = fft_spectrum[int(ind)]*(int(ind+1)-ind)
    new_arr[i] += fft_spectrum[int(ind+1)]*(ind-int(ind))
    
    #if f > 10000:# (2)
    #    break
    #    fft_spectrum[i] = 0.0


fft_spectrum_abs = np.abs(new_arr)


noiseless_signal = np.fft.irfft(new_arr)

wavfile.write("noiseless.wav", sampFreq, noiseless_signal)
print("saved")

plt.plot(freq[:100000], fft_spectrum_abs[:100000])
plt.xlabel("frequency, Hz")
plt.ylabel("Amplitude, units")
plt.show()


"""
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft
import numpy as np
rate, data = wav.read('400 Hz Test Tone.wav')
fft_out = fft(data)
#%matplotlib inline
plt.plot(data, np.abs(fft_out))
plt.show()

import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile # get the api
fs, data = wavfile.read('400 Hz Test Tone.wav') # load the data
a = data.T[0] # this is a two channel soundtrack, I get the first track
b=[(ele/2**8.)*2-1 for ele in a] # this is 8-bit track, b is now normalized on [-1,1)
c = fft(b) # calculate fourier transform (complex numbers list)
d = len(c)/2  # you only need half of the fft list (real signal symmetry)
plt.plot(abs(c[:(d-1)]),'r') 
plt.show()

"""
