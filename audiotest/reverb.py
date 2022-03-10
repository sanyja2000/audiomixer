import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (9, 7)

sampFreq, sound = wavfile.read('feel_cut.wav')

#sound = sound / 2.0**15

length_in_s = sound.shape[0] / sampFreq
print(length_in_s)
signal = sound[:,0]
time = np.arange(sound.shape[0]) / sound.shape[0] * length_in_s
"""
xoff = 200000
plt.plot(time[xoff+6000:xoff+7000], signal[xoff+6000:xoff+7000])
plt.xlabel("time, s")
plt.ylabel("Signal, relative units")
plt.show()
"""

#fft_spectrum = np.fft.rfft(signal)
#freq = np.fft.rfftfreq(signal.size, d=1./sampFreq)


#new_arr = fft_spectrum.copy()


#fft_spectrum_abs = np.abs(new_arr)


#noiseless_signal = np.fft.irfft(new_arr)


plt.plot(time, signal)
plt.xlabel("time, s")
plt.ylabel("Signal, relative units")
plt.show()


reverbTime = 1.0
mixPercent = 50
reverbSamp = int(reverbTime*sampFreq)
output = signal.copy()
l = len(signal)-reverbSamp
print(reverbSamp)
i=0


while i<l:
    #for i in range(len(signal)-reverbSamp):
    # https://medium.com/the-seekers-project/coding-a-basic-reverb-algorithm-part-2-an-introduction-to-audio-programming-4db79dd4e325
    output[i] = ((100 - mixPercent) * signal[i]) + (mixPercent * outputComb[i]); 
    i+=1


"""
#wobble
for i in range(len(signal)):
    signal[i] *= np.sin(i/10000.0)
"""



plt.plot(time, signal)
plt.xlabel("time, s")
plt.ylabel("Signal, relative units")
plt.show()

#wavfile.write("noiseless.wav", sampFreq, noiseless_signal)
wavfile.write("out.wav", sampFreq, signal)
print("saved")

"""
plt.plot(freq[:100000], fft_spectrum_abs[:100000])
plt.xlabel("frequency, Hz")
plt.ylabel("Amplitude, units")
plt.show()
"""
