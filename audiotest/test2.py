import pyaudio
import pyfftw
import wave
import numpy as np

def linp(a,b,t):
    return a*(1-t)+b*t


pa = pyaudio.PyAudio()

wf = wave.open('feel_cut.wav', 'rb')
form = pa.get_format_from_width(wf.getsampwidth())
print(form)
#paInt16
stream = pa.open(format=form,
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True)
a = pyfftw.empty_aligned(2048, dtype='float32')
b = pyfftw.empty_aligned(1025, dtype='complex64')
c = pyfftw.empty_aligned(1025, dtype='complex64')


fft_object = pyfftw.FFTW(a, b)
ifft_object = pyfftw.FFTW(b, a, direction='FFTW_BACKWARD')

data = wf.readframes(1024)

arr = []
for x in range(2048):
    arr.append(0.2)
    #arr.append(np.sin(1000*x)/2+0.5)

arr = np.array(arr)
print(arr)

t = 0

twe = 2**(1/12)

while len(data) > 0:
    numpydata = np.frombuffer(data, dtype=np.int16)
    a[:] = np.array(numpydata, dtype=np.float32)
    fft_object()

    """
    for i in range(1025):
        ind = int(i/twe)
        t = i/twe-ind
        if ind<1025:
            c[i] = linp(b[ind],b[ind+1],t)
        else:
            c[i] = 0.0

    """

    b[:300] = b[:300]*0.1
    
    ifft_object()
    numpydata = np.array(a, dtype=np.int16)
    
    outdata = numpydata.astype(np.int16).tobytes()
    
    stream.write(outdata)
    data = wf.readframes(1024)
    t+=0.1
stream.stop_stream()
stream.close()
