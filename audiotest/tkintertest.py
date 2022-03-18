from tkinter import *
from tkinter import ttk

from threading import Thread
import pyaudio
import numpy as np

pa = pyaudio.PyAudio()

sampleRate=44100

stream = pa.open(format=pyaudio.paInt16,
            channels=2,
            rate=sampleRate,
            output=True)

masterVol = 3000

desFreqSine = 440
desVolSine = 0
desFreqTri = 440
desVolTri = 0
desFreqSq = 440
desVolSq = 0
desVolDc = 0

def sinToSq(n):
    return np.floor(n*0.999)*2+1



sinarr = np.ones(2048)
#mult = desFreqSine/(sampleRate/3.1415)
val = 0
direc = 1
t=0

exited = False

def playWave():
    global sinarr, t, stream, mult, desFreq, val, direc
    while not exited:
        t+=len(sinarr)

        
        multS = desFreqSine/(sampleRate/3.1415)
        multSq = desFreqSq/(sampleRate/3.1415)
        multT = desFreqTri/sampleRate*2
        for x in range(len(sinarr)):
            val+=direc*multT
            if val>=1 or val<=-1:
                direc*=-1
            sinval = np.sin((t+x)*multS)
            sqval = np.sign(np.sin((t+x)*multSq))
            sinarr[x] = sqval*desVolSq + sinval*desVolSine + val*desVolTri + desVolDc
        output = (sinarr*masterVol).astype(np.int16).tobytes()

        stream.write(output)


th = Thread(target=playWave)
th.start()



def changeSinFreq(n):
    global desFreqSine
    desFreqSine = int(sampleRate/8*(float(n)+0.001))
    lblsinfreq.config(text = "Sine Frequency: "+str(desFreqSine)+"Hz")

def changeSinVol(n):
    global desVolSine
    desVolSine = float(n)
    lblsinvol.config(text = "Sine volume: "+str(desVolSine)+"%")


def changeTriFreq(n):
    global desFreqTri
    desFreqTri = int(sampleRate/8*(float(n)+0.001))
    lbltrifreq.config(text = "Triwave Frequency: "+str(desFreqTri)+"Hz")

def changeTriVol(n):
    global desVolTri
    desVolTri = float(n)
    lbltrivol.config(text = "Triwave volume: "+str(desVolTri)+"%")

def changeSqFreq(n):
    global desFreqSq
    desFreqSq = int(sampleRate/8*(float(n)+0.001))
    lblsqfreq.config(text = "Squarewave Frequency: "+str(desFreqSq)+"Hz")

def changeSqVol(n):
    global desVolSq
    desVolSq = float(n)
    lblsqvol.config(text = "Squarewave volume: "+str(desVolSq)+"%")

def changeDcVol(n):
    global desVolDc
    desVolDc = float(n)
    lbldcvol.config(text = "DC offset: "+str(desVolDc)+"%")


root = Tk()
frm = ttk.Frame(root, padding=10)
frm.grid()
lblsinfreq = ttk.Label(frm, text="Sine Frequency: 440Hz")
lblsinfreq.grid(column=0, row=0)
ttk.Scale(frm, command=changeSinFreq,length=300).grid(column=0, row=1)

lblsinvol = ttk.Label(frm, text="Sine volume: 0%")
lblsinvol.grid(column=0, row=2)
ttk.Scale(frm, command=changeSinVol,length=300).grid(column=0, row=3)


lbltrifreq = ttk.Label(frm, text="Triwave Frequency: 440Hz")
lbltrifreq.grid(column=0, row=5)
ttk.Scale(frm, command=changeTriFreq,length=300).grid(column=0, row=6)

lbltrivol = ttk.Label(frm, text="Triwave volume: 0%")
lbltrivol.grid(column=0, row=7)
ttk.Scale(frm, command=changeTriVol,length=300).grid(column=0, row=8)


lblsqfreq = ttk.Label(frm, text="Squarewave Frequency: 440Hz")
lblsqfreq.grid(column=0, row=10)
ttk.Scale(frm, command=changeSqFreq,length=300).grid(column=0, row=11)

lblsqvol = ttk.Label(frm, text="Squarewave volume: 0%")
lblsqvol.grid(column=0, row=12)
ttk.Scale(frm, command=changeSqVol,length=300).grid(column=0, row=13)

lbldcvol = ttk.Label(frm, text="DC offset: 0%")
lbldcvol.grid(column=0, row=15)
ttk.Scale(frm, command=changeDcVol,length=300).grid(column=0, row=16)


ttk.Button(frm, text="Quit", command=root.destroy).grid(column=2, row=0)
root.mainloop()
exited= True




"""
while len(data) > 0:
    t+=len(sinarr)
    for x in range(len(sinarr)):
        sinarr[x] = np.sin((t+x)*mult)*3000
    output = sinarr.astype(np.int16).tobytes()#c.astype(np.int16).tobytes()


    stream.write(output)
"""
