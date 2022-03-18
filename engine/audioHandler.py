import pyaudio, time, numpy as np
import wave
from threading import Thread
from functools import partial
import math
import struct

class AudioHandler:
    def __init__(self):
        self.p = pyaudio.PyAudio()

        self.masterVolume = 1
        self.channelVolume = [1,0,0,0,0]
        self.isStopped = [0,0,0,0,0]

        self.dataPlaying = np.zeros(2048)
        self.dataReady = False
        self.exit = False

        self.maxVolume = 0

        self.currentlyPlaying = {}
        #self.stream.start_stream()
    def isStillPlaying(self,filename):
        if filename in self.currentlyPlaying:
            if self.currentlyPlaying[filename]:
                return True
        return False

    def speakerStart(self):
        wf = wave.open("res/audio/feel_cut.wav", 'rb')
        stream = self.p.open(format=self.p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)#,stream_callback=self.fileCallback)
        sformat = self.p.get_format_from_width(wf.getsampwidth())

        stream.start_stream()
        th = Thread(target=self.speakerOutPlayBlock, args=(wf,stream))
        th.start()

    def speakerOutPlayBlock(self,wf,stream):
        #data = wf.readframes(1024)
        while not self.exit:
            if self.dataReady:
                stream.write(self.dataPlaying)
                #data = wf.readframes(1024)
                self.dataReady = False
            else:
                time.sleep(0.001)
        stream.stop_stream()
        stream.close()
        

    def playSound(self,filename,volumeIndex=0):
        self.isStopped[volumeIndex] = 0
        wf = wave.open(filename, 'rb')
        stream = self.p.open(format=self.p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)#,stream_callback=self.fileCallback)
        sformat = self.p.get_format_from_width(wf.getsampwidth())

        stream.start_stream()
        th = Thread(target=self.playBlockSound, args=(wf,stream,filename,volumeIndex))
        th.start()
    
    def playBlockSound(self,wf,stream,filename,volumeIndex):
        i = 0
        data = wf.readframes(1024)
        self.currentlyPlaying[filename] = True
        # play stream (3)
        while len(data) > 0:
            if self.isStopped[volumeIndex]:
                break
            buf = np.frombuffer(data,dtype=np.int16)*self.channelVolume[volumeIndex]*self.masterVolume
            # Pause menu animation?
            # print([abs(i) for i in buf[:len(buf)//2]])
            # self.maxVolume += sum([abs(i) for i in buf[:len(buf)//2]])
            outdata = buf.astype(np.int16).tostring()
            stream.write(outdata)
            data = wf.readframes(1024)
            #i=(i+0.01)%1
        self.currentlyPlaying[filename] = False 
        stream.stop_stream()
        stream.close()
    def stopAll(self):
        self.isStopped = [1,1,1,1,1]
        self.channelVolume = [1,0,0,0,0]
    def update(self):
        self.maxVolume = 0

