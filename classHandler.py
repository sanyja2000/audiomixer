import math
from multiprocessing import parent_process

from engine.objectHandler import Object3D
from engine.renderer import IndexBuffer, Shader, Texture, VertexArray, VertexBuffer, VertexBufferLayout
import numpy as np
import time
from PIL import Image
import pyrr
import random
import wave
from OpenGL.GLUT import *
import glm
import tkinter as tk
import tkinter.filedialog
import os.path
from OpenGL.GL import *
import pyfftw
from threading import Thread
"""
This file contains the classes for different types of objects in the map files.

"""

SAMPLESIZE = 3072 # Must be even, choose higher for better performance


def easeInOutSine(x):
    return -(math.cos(math.pi * x) - 1) / 2


def lerp(f, t, n):
    return f*(1-n)+t*n

def lerpVec3(f,t,n):
    out = []
    for x in range(3):
        out.append(f[x]*(1-n)+t[x]*n)
    return np.array(out)


def dist(a,b):
    return math.sqrt((b[0]-a[0])**2+(b[1]-a[1])**2+(b[2]-a[2])**2)

def lerpConst(f,t,n):
    # Lerp with constaints
    val = f*(1-n)+t*n
    if f>t:
        f,t=t,f
    if val<f:
        val = f
    if val>t:
        val = t

    return val

def constraint(n,f,t):
    if n<f:
        return f
    if n>t:
        return t
    return n


def lerparr(arr,dlen):
    # this function changes the arrays length to dlen
    # and interpolates the values between
    xp = np.arange(len(arr))
    step = (len(arr)-1)/(dlen-1)
    outarr = np.interp(np.arange(dlen)*step,xp,arr)
    return outarr

class Decoration:
    def __init__(self,ph,props):
        """Basic wrapper for decoration objects. These don't interact with anything."""
        self.name = props["name"]
        self.model = ph.loadFile(props["file"],props["texture"])
        self.model.SetScale(props["scale"])
        self.model.SetPosition(np.array(props["pos"]))
        self.model.SetRotation(np.array(props["rot"]))
        self.model.defaultPosition = np.array(props["pos"])
        self.shaderName = "default"
        if("transparent" in props):
            self.shaderName = "default_transparent"
    def draw(self,shaderhandler,renderer,viewMat):
        self.model.DrawWithShader(shaderhandler.getShader(self.shaderName),renderer,viewMat,options={"selected":int(self.isSelected)})
    def update(self,fpsCounter,audioHandler):
        pass
        #self.model.SetRotation(np.array([self.model.rot[0],self.rot,self.model.rot[2]]))

class NodeElement:
    def __init__(self,ph,props):
        """Inputs: freq,amp. Output: wave signal"""
        self.name = props["name"]
        self.model = ph.loadFile(props["file"],props["texture"])
        self.model.SetScale(props["scale"])
        self.model.SetPosition(np.array(props["pos"]))
        self.model.SetRotation(np.array(props["rot"]))
        self.model.defaultPosition = np.array(props["pos"])
        self.inputs = {}
        self.outputs = {}
        self.isSelected = False
        self.changedProperty = True
        self.properties = {}
        self.propertyTypes = {}
        self.connpoints = []
        self.shaderName = "default"
    """
    def draw(self,shaderhandler,renderer,viewMat):
        self.model.DrawWithShader(shaderhandler.getShader(self.shaderName),renderer,viewMat)
    """
    def draw(self,shaderhandler,renderer,viewMat):
        self.model.DrawWithShader(shaderhandler.getShader(self.shaderName),renderer,viewMat,options={"selected":int(self.isSelected)})
        for cp in self.connpoints:
            cp.draw(shaderhandler,renderer,viewMat)
    def checkIntersect(self,x,y):
        for cp in self.connpoints:
            if cp.checkIntersect(x,y):
                return cp
        if self.model.pos[0]-self.model.scale*1 <x<self.model.pos[0]+self.model.scale*1:
            if self.model.pos[1]-self.model.scale*1 <y<self.model.pos[1]+self.model.scale*1:
                return self
        return None
    def update(self,fpsCounter,audioHandler):
        pass
    def updateProperty(self):
        pass

class ConnectionPoint:
    def __init__(self,ph,parent,name,side,pos,posoffset):
        """Connection points for inputs, outputs."""
        self.side = side
        self.name = name
        self.parent = parent
        self.model = ph.loadFile("res/torus.obj","res/cptexture.png")
        self.model.SetScale(0.03)
        self.model.SetPosition(glm.vec3(pos)+glm.vec3(posoffset))
        self.model.SetRotation(np.array([1.57,0,0]))
        self.positionOffset = posoffset
        self.shaderName = "default"
        self.bezier = None
        self.data = None
    def draw(self,shaderhandler,renderer,viewMat):
        self.model.DrawWithShader(shaderhandler.getShader(self.shaderName),renderer,viewMat,options={"selected":0})
        #CPH.drawablePoints.append(self)
    def checkIntersect(self,x,y):
        if((self.model.pos[0]-x)**2+(self.model.pos[1]-y)**2<0.007):
            return True
        return False
    def updatePos(self,pos):
        self.model.SetPosition(pos+self.positionOffset)

class ConnectionPointHandler:
    def __init__(self):
        self.drawablePoints = []
    def DrawWithShader(self,shaderhandler,renderer,viewMat):
        if len(self.drawablePoints) == 0:
            return
        #print(self.drawablePoints[0].model.pos)
        shader = shaderhandler.getShader("instanced")
        #self.model.DrawWithShader(shaderhandler.getShader(self.shaderName),renderer,viewMat,options={"selected":0})
        shader.Bind()
        self.drawablePoints[0].model.texture.Bind()
        shader.SetUniform1i("u_Texture",0)
        veclist = []
        ori_p = np.matmul(np.append(self.drawablePoints[0].model.pos,1.0),np.matmul(viewMat,self.drawablePoints[0].model.modelMat))
        for dp in self.drawablePoints:
            # z x y
            newpos = np.matmul(np.append(dp.model.pos,1.0),np.matmul(viewMat,dp.model.modelMat))
            #veclist.append(np.array([0,newpos[1]*5,0]))
            veclist.append((newpos)[:3])
            #veclist.append(dp.model.pos-self.drawablePoints[0].model.pos)
        
        offsetList = np.array(veclist).flatten()
        #offsetList = np.array([0,0,0,1,1,1,2,2,2])
        #print(offsetList)
        mvp = np.transpose(np.matmul(viewMat,self.drawablePoints[0].model.modelMat))
        shader.SetUniformMat4f("u_MVP", mvp)
        renderer.DrawInstanced(self.drawablePoints[0].model.va,self.drawablePoints[0].model.ib,shader,len(self.drawablePoints),offsetList)
        self.drawablePoints = []

CPH = ConnectionPointHandler()

class SineGenerator(NodeElement):
    def __init__(self,ph,name, pos):
        """Inputs: amp. Output: sinewave signal"""
        
        NodeElement.__init__(self,ph,{"name":name, "pos":pos,"rot":[1.57,0,0],"scale":0.3,"file":"res/input.obj","texture":"res/sinewave.png"})
        self.inputs = ["amp","freq"]
        self.outputs = ["signal"]
        self.properties = {"offset":0}
        self.propertyTypes = {"offset":"float"}
        self.lastSent = 0
        self.processedFrames = 0
        self.deffreq = 440
        self.defamp = 1000
        self.amp = self.defamp
        self.freq = self.deffreq
        self.connpoints = []
        for ind in range(len(self.outputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.outputs[ind],"out",pos,glm.vec3([-0.55,-0.1*ind,0])))
        for ind in range(len(self.inputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.inputs[ind],"in",pos,glm.vec3([0.55,-0.1*ind,0])))  

    def draw(self,shaderhandler,renderer,viewMat):
        self.model.DrawWithShader(shaderhandler.getShader(self.shaderName),renderer,viewMat,options={"selected":int(self.isSelected)})
        for cp in self.connpoints:
            cp.draw(shaderhandler,renderer,viewMat)


    def update(self,fpsCounter,audioHandler):
        for cp in self.connpoints:
            cp.updatePos(self.model.pos)
    def audioUpdate(self,fpsCounter,audioHandler):
        for cp in self.connpoints:
            if cp.name == "freq":
                if cp.data is None:
                    self.freq = self.deffreq
                else:
                    self.freq = np.abs(cp.data[0])
            if cp.name == "amp":
                if cp.data is not None:
                    self.amp = cp.data
                else:
                    self.amp = self.defamp
            if cp.name == "signal" and cp.bezier != None:
                cur = fpsCounter.currentTime
                if not audioHandler.dataReady  and cur-self.lastSent>fpsCounter.deltaTime:
                    mult = self.freq/(44100/3.1415)
                    cp.data = np.sin((np.arange(SAMPLESIZE)+self.processedFrames)*mult)*self.amp+self.properties["offset"]
                    self.lastSent = cur
                    self.processedFrames += SAMPLESIZE
        



class SquareGenerator(NodeElement):
    def __init__(self,ph,name, pos):
        """Inputs: freq. Output: squarewave signal"""
        
        NodeElement.__init__(self,ph,{"name":name, "pos":pos,"rot":[1.57,0,0],"scale":0.3,"file":"res/input.obj","texture":"res/squarewave.png"})
        self.inputs = ["amp","freq"]
        self.outputs = ["signal"]
        self.properties = {"offset":0}
        self.propertyTypes = {"offset":"float"}
        self.processedFrames = 0
        self.deffreq = 440
        self.defamp = 1000
        self.amp = self.defamp
        self.freq = self.deffreq
        self.lastSent = 0
        self.connpoints = []
        for ind in range(len(self.inputs)):
            self.connpoints.append(ConnectionPoint(ph,self,list(self.inputs)[ind],"in",pos,glm.vec3([0.55,-0.1*ind,0])))  
        for ind in range(len(self.outputs)):
            self.connpoints.append(ConnectionPoint(ph,self,list(self.outputs)[ind],"out",pos,glm.vec3([-0.55,-0.1*ind,0])))



    def update(self,fpsCounter,audioHandler):
        for cp in self.connpoints:
            cp.updatePos(self.model.pos)
    def audioUpdate(self,fpsCounter,audioHandler):
        for cp in self.connpoints:
            if cp.name == "freq":
                if cp.bezier is None or cp.data is None:
                    self.freq = self.deffreq
                else:
                    self.freq = np.abs(cp.data[0])
            if cp.name == "amp":
                if cp.data is not None:
                    self.amp = cp.data
                else:
                    self.amp = self.defamp
            if cp.name == "signal" and cp.bezier != None:
                cur = fpsCounter.currentTime
                if not audioHandler.dataReady and cur-self.lastSent>fpsCounter.deltaTime:
                    mult = self.freq/(44100/3.1415)
                    cp.data = np.sign(np.sin((np.arange(SAMPLESIZE)+self.processedFrames)*mult))*self.amp+self.properties["offset"]
                    self.lastSent = cur
                    self.processedFrames += SAMPLESIZE
        


class FilePlayer(NodeElement):
    def __init__(self,ph,name, pos):
        """Inputs: None. Output: file signal"""
        
        NodeElement.__init__(self,ph,{"name":name, "pos":pos,"rot":[1.57,0,0],"scale":0.3,"file":"res/input.obj","texture":"res/inputfile.png"})
        self.inputs = ["speed"]
        self.outputs = ["signal"]
        self.lastSent = 0
        self.validFile = False
        self.openedFile = ""
        if name == "":
            self.properties = {"file":"<not selected>","cue":False,"cueLength":10,"speed":1,"paused":False,"reset":self.resetFile}
        else:
            self.properties = {"file":name,"cue":False,"cueLength":10,"speed":1,"paused":False,"reset":self.resetFile}
            self.changeFileTo(name)
        self.propertyTypes = {"file":"file","cue":"bool","cueLength":"integer","speed":"float","paused":"bool","reset":"button"}
        self.cueTime = 0
        self.cueEnabled = True
        self.cuePoint = 0
        self.connpoints = []
        for ind in range(len(self.inputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.inputs[ind],"in",pos,glm.vec3([0.55,-0.1*ind,0])))  
        for ind in range(len(self.outputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.outputs[ind],"out",pos,glm.vec3([-0.55,-0.1*ind,0])))

    def resetFile(self):
        if self.validFile:
            self.wf.setpos(0)

    def changeFileTo(self,filename):
        self.wf = wave.open(filename, 'rb')
        self.validFile = True
        self.openedFile = filename

    def updateProperty(self):
        if self.properties["file"]=="<not selected>":
            self.validFile = False
            self.connpoints[1].data = np.zeros(SAMPLESIZE,dtype=np.float32)
        elif self.properties["file"]!=self.openedFile:
            self.changeFileTo(self.properties["file"])
            
        if self.properties["cue"] != False:
            if not self.cueEnabled:
                if self.validFile:
                    self.cuePoint = self.wf.tell()
                else:
                    self.cuePoint = 0
                self.cueEnabled = True
        else:
            self.cueEnabled = False
        self.properties["cueLength"] = int(self.properties["cueLength"])
        if self.properties["cueLength"]<1:
            self.properties["cueLength"] = 1
        if self.cueTime>self.properties["cueLength"]:
            self.cueTime = 0
        if self.properties["speed"]<0.1:
            self.properties["speed"] = 0.1


    def update(self,fpsCounter,audioHandler):
        for cp in self.connpoints:
            cp.updatePos(self.model.pos)
        if self.changedProperty:
            self.updateProperty()
    
    def audioUpdate(self,fpsCounter,audioHandler):
        if not self.validFile or self.properties["paused"]:
            return
        if self.connpoints[0].bezier != None and self.connpoints[0].data is not None:
            d = np.abs(self.connpoints[0].data[0])
            if d>0.01:
                self.properties["speed"] = d
        outsig = self.connpoints[1]
        if outsig.bezier != None:
            cur = fpsCounter.currentTime 
            if not audioHandler.dataReady and cur-self.lastSent>fpsCounter.deltaTime:
                if self.cueEnabled:
                    self.cueTime = (self.cueTime + 1) % self.properties["cueLength"]
                    if self.cueTime == 0:
                        self.wf.setpos(self.cuePoint)
                outsig.data = self.wf.readframes(int(SAMPLESIZE/2*self.properties["speed"]))
                #outsig.data = lerparr(self.wf.readframes(int(SAMPLESIZE/2*self.properties["speed"])),SAMPLESIZE)
                self.lastSent = cur
                if len(outsig.data)/2<int(SAMPLESIZE/2*self.properties["speed"])*2:
                    self.wf.setpos(0)
                    #print(len(outsig.data)/2, int(SAMPLESIZE*self.properties["speed"]))
                    outsig.data = self.wf.readframes(int(SAMPLESIZE/2*self.properties["speed"]))
                    #outsig.data = self.wf.readframes(SAMPLESIZE//2)
                outsig.data = lerparr(np.frombuffer(outsig.data, dtype=np.int16),SAMPLESIZE)


class NoiseNode(NodeElement):
    def __init__(self,ph,name, pos):
        """Inputs: amp. Output: sinewave signal"""
        
        NodeElement.__init__(self,ph,{"name":name, "pos":pos,"rot":[1.57,0,0],"scale":0.3,"file":"res/input.obj","texture":"res/noisenode.png"})
        self.inputs = []
        self.outputs = ["signal"]
        self.properties = {"offset":0}
        self.propertyTypes = {"offset":"float"}
        self.lastSent = 0
        self.processedFrames = 0
        self.connpoints = []
        for ind in range(len(self.outputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.outputs[ind],"out",pos,glm.vec3([-0.55,-0.1*ind,0])))
        for ind in range(len(self.inputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.inputs[ind],"in",pos,glm.vec3([0.55,-0.1*ind,0])))  

    def draw(self,shaderhandler,renderer,viewMat):
        self.model.DrawWithShader(shaderhandler.getShader(self.shaderName),renderer,viewMat,options={"selected":int(self.isSelected)})
        for cp in self.connpoints:
            cp.draw(shaderhandler,renderer,viewMat)


    def update(self,fpsCounter,audioHandler):
        for cp in self.connpoints:
            cp.updatePos(self.model.pos)
    def audioUpdate(self,fpsCounter,audioHandler):
        self.connpoints[0].data = np.random.normal(0, 2**15, SAMPLESIZE)
        """
        for cp in self.connpoints:
            if cp.name == "freq":
                if cp.data is None:
                    self.freq = self.deffreq
                else:
                    self.freq = np.abs(cp.data[0])
            if cp.name == "amp":
                if cp.data is not None:
                    self.amp = cp.data
                else:
                    self.amp = self.defamp
            if cp.name == "signal" and cp.bezier != None:
                cur = fpsCounter.currentTime
                if not audioHandler.dataReady  and cur-self.lastSent>fpsCounter.deltaTime:
                    mult = self.freq/(44100/3.1415)
                    cp.data = np.sin((np.arange(SAMPLESIZE)+self.processedFrames)*mult)*self.amp+self.properties["offset"]
                    self.lastSent = cur
                    self.processedFrames += SAMPLESIZE
        """
        



class Keyboard(NodeElement):
    def __init__(self,ph,name, pos):
        """Inputs: None. Output: constant*np.ones(n)"""
        
        NodeElement.__init__(self,ph,{"name":name, "pos":pos,"rot":[1.57,0,0],"scale":0.3,"file":"res/inputsmall.obj","texture":"res/keyboardnode.png"})
        self.inputs = []
        self.outputs = ["freq","gate"]
        self.lastSent = 0
        self.connpoints = []
        self.properties = {"freq":261.63,"mult":1}
        self.propertyTypes = {"freq":"float","mult":"float"}
        self.gate = 0
        self.keys = "ysxdcvgbhnjm,"
        self.freqs = [261.63,277.18,293.66,311.13,329.63,349.23,369.99,392,415.3,440,466.16,493.88,523.25]
        self.outputKey = np.ones(SAMPLESIZE,dtype=np.float32)*self.properties["freq"]/440
        #self.outputMult = np.ones(SAMPLESIZE,dtype=np.float32)*self.properties["mult"]
        self.outputGate = np.ones(SAMPLESIZE,dtype=np.float32)*self.gate
        
        for ind in range(len(self.inputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.inputs[ind],"in",pos,glm.vec3([0.35,-0.1*ind,0])))  
        for ind in range(len(self.outputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.outputs[ind],"out",pos,glm.vec3([-0.34,-0.1*ind,0])))
        

    def updateProperty(self):
        self.outputKey = np.ones(SAMPLESIZE,dtype=np.float32)*self.properties["freq"]/440
        #self.outputMult = np.ones(SAMPLESIZE,dtype=np.float32)*self.properties["mult"]
        self.outputGate = np.ones(SAMPLESIZE,dtype=np.float32)*self.gate
        self.changedProperty = False
    
    def update(self,fpsCounter,audioHandler):
        if self.changedProperty:
            self.updateProperty()
        for cp in self.connpoints:
            cp.updatePos(self.model.pos)
    def audioUpdate(self,fpsCounter,audioHandler):
        freqout = self.connpoints[0]
        if freqout.bezier != None:
            freqout.data = self.outputKey#np.frombuffer(self.outputData, dtype=np.float32)
        gateout = self.connpoints[1]
        if gateout.bezier != None:
            gateout.data = self.outputGate
    def updateKeys(self,inputHandler):
        
        keydown = 0
        for k in self.keys:
            if inputHandler.isKeyHeldDown(str.encode(k)) or inputHandler.isKeyDown(str.encode(k)):
                self.properties["freq"]=self.freqs[self.keys.index(k)]
                keydown = 1
                break
        if self.gate != keydown:
            self.gate = keydown
            self.updateProperty()
        

class TextDisplay:
    def __init__(self,ph,props):
        """Text displayer"""
        self.model = ph.empty()
        self.model.SetScale(props["scale"])
        self.model.SetPosition(np.array(props["pos"]))
        self.model.SetRotation(np.array(props["rot"]))
        self.model.defaultPosition = np.array(props["pos"])
        self.positionOffset = glm.vec3(props["posoffset"])
        self.text = "asd"

        self.shaderName = "font"
    def draw(self,shaderhandler,renderer,viewMat):
        pass
    def updatePos(self,pos):
        self.model.SetPosition(pos+self.positionOffset)


class ConstantNode(NodeElement):
    def __init__(self,ph,name, pos):
        """Inputs: None. Output: constant*np.ones(n)"""
        
        NodeElement.__init__(self,ph,{"name":name, "pos":pos,"rot":[1.57,0,0],"scale":0.3,"file":"res/inputsmall.obj","texture":"res/constantnode.png"})
        self.inputs = []
        self.outputs = ["signal"]
        self.lastSent = 0
        self.connpoints = []
        self.properties = {"value":50}
        self.propertyTypes = {"value":"float"}

        self.textDisplay = TextDisplay(ph, {"pos":pos,"rot":[0,3.1415,0],"scale":0.2,"posoffset":[0.25,-0.1,-0.1]})

        self.outputData = np.ones(SAMPLESIZE,dtype=np.float32)*self.properties["value"]
        for ind in range(len(self.inputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.inputs[ind],"in",pos,glm.vec3([0.35,-0.1*ind,0])))  
        for ind in range(len(self.outputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.outputs[ind],"out",pos,glm.vec3([-0.34,-0.1*ind-0.2,0])))
        

    def updateProperty(self):
        self.outputData = np.ones(SAMPLESIZE,dtype=np.int16)*self.properties["value"]
        self.changedProperty = False
    
    def update(self,fpsCounter,audioHandler):
        if self.changedProperty:
            self.updateProperty()
        for cp in self.connpoints:
            cp.updatePos(self.model.pos)
        self.textDisplay.updatePos(self.model.pos)
    def audioUpdate(self,fpsCounter,audioHandler):
        outsig = self.connpoints[0]
        if outsig.bezier != None:
            outsig.data = self.outputData#np.frombuffer(self.outputData, dtype=np.float32)

    def drawText(self,fontHandler,renderer,viewMat):
        fontHandler.drawText3D(str(self.outputData[0]),self.textDisplay.model.modelMat,0.05,viewMat,renderer)
        

class MixerNode(NodeElement):
    def __init__(self,ph,name, pos):
        """Inputs: A,B. Output: A+B"""
        
        NodeElement.__init__(self,ph,{"name":name, "pos":pos,"rot":[1.57,0,0],"scale":0.3,"file":"res/inputsmall.obj","texture":"res/mixernode.png"})
        self.inputs = ["A","B","mix"]
        self.outputs = ["signal"]

        self.connpoints = []
        self.outputData = np.ones(SAMPLESIZE,dtype=np.int16)*880
        for ind in range(len(self.inputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.inputs[ind],"in",pos,glm.vec3([0.35,-0.14*ind+0.13,0])))  
        for ind in range(len(self.outputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.outputs[ind],"out",pos,glm.vec3([-0.35,-0.1*ind,0])))
        
        self.aconn = self.connpoints[0]
        self.bconn = self.connpoints[1]
        self.mixconn = self.connpoints[2]


    def update(self,fpsCounter,audioHandler):
        for cp in self.connpoints:
            cp.updatePos(self.model.pos)
    
    def audioUpdate(self,fpsCounter,audioHandler):
        outsig = self.connpoints[3]
        outsig.data = np.zeros(SAMPLESIZE, dtype=np.float32)
        multiplier = 0.5
        if self.mixconn.data is not None:
            multiplier = constraint(self.mixconn.data[0],0,1)
            self.mixconn.data = None
        if self.aconn.data is not None:
            np.add(outsig.data, self.aconn.data*(1-multiplier), out=outsig.data, casting="unsafe")
            self.aconn.data = None
            #outsig.data = np.maximum(self.aconn.data, outsig.data)
        if self.bconn.data is not None:
            np.add(outsig.data, self.bconn.data*multiplier, out=outsig.data, casting="unsafe")
            self.bconn.data = None
            #outsig.data = np.maximum(self.bconn.data, outsig.data)



class FilterNode(NodeElement):
    def __init__(self,ph,name, pos):
        """Inputs: A,B. Output: filtered A"""
        
        NodeElement.__init__(self,ph,{"name":name, "pos":pos,"rot":[1.57,0,0],"scale":0.3,"file":"res/inputsmall.obj","texture":"res/filternode.png"})
        self.inputs = ["A","B"]
        self.outputs = ["signal"]
        self.lastSent = 0
        self.connpoints = []

        self.properties = {"frequency":4000,"enabled":True}
        self.propertyTypes = {"frequency":"float","enabled":"bool"}

        self.datain = pyfftw.empty_aligned(SAMPLESIZE, dtype='float32')
        self.fftdata = pyfftw.empty_aligned(SAMPLESIZE//2+1, dtype='complex64')
        self.dataout = pyfftw.empty_aligned(SAMPLESIZE, dtype='float32')

        self.samplerate = 48000

        self.fft_function_forw = pyfftw.FFTW(self.datain, self.fftdata)
        self.fft_function_back = pyfftw.FFTW(self.fftdata, self.dataout, direction='FFTW_BACKWARD')
        
        self.window = np.hanning(SAMPLESIZE)

        self.outputData = np.ones(SAMPLESIZE,dtype=np.int16)*880
        for ind in range(len(self.inputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.inputs[ind],"in",pos,glm.vec3([0.35,-0.14*ind+0.13,0])))  
        for ind in range(len(self.outputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.outputs[ind],"out",pos,glm.vec3([-0.35,-0.1*ind,0])))
        
        
        freq = int(self.properties["frequency"]/self.samplerate*SAMPLESIZE/2)
        self.currentfreq = freq
        self.filterarr = np.concatenate((np.ones(freq),np.zeros(SAMPLESIZE//2+1-freq)))
        print(self.filterarr)

    def updateProperty(self):
        #self.outputData = np.ones(SAMPLESIZE,dtype=np.int16)*self.properties["value"]
        if self.properties["enabled"]:
            freq = min(int(self.properties["frequency"]/self.samplerate*SAMPLESIZE/2),SAMPLESIZE//2+1)
            self.filterarr = np.concatenate((np.ones(freq),np.zeros(SAMPLESIZE//2+1-freq)))
            self.currentfreq = freq
        else:           
            self.filterarr = np.ones(SAMPLESIZE//2+1)
        self.changedProperty = False

    def update(self,fpsCounter,audioHandler):
        for cp in self.connpoints:
            cp.updatePos(self.model.pos)
        if self.changedProperty:
            self.updateProperty()
    
    def audioUpdate(self,fpsCounter,audioHandler):
        outsig = self.connpoints[2]
        outsig.data = np.zeros(SAMPLESIZE, dtype=np.float32)
        
        if self.connpoints[1].data is not None:
            self.properties["frequency"] = self.connpoints[1].data[0]
            if self.properties["frequency"] != self.currentfreq:
                self.updateProperty()

        if self.connpoints[0].data is not None:
            self.datain[:] = self.connpoints[0].data#*self.window
            self.fft_function_forw()
            self.fftdata[:] = self.fftdata*self.filterarr
            self.fft_function_back()
            outsig.data = self.dataout#/self.window
            self.connpoints[0].data = None



class SplitterNode(NodeElement):
    def __init__(self,ph,name, pos):
        """Inputs: A. Output: A,A,A"""
        
        NodeElement.__init__(self,ph,{"name":name, "pos":pos,"rot":[1.57,0,0],"scale":0.3,"file":"res/inputsmall.obj","texture":"res/splitternode.png"})
        self.inputs = ["signal"]
        self.outputs = ["A","B","C"]
        self.lastSent = 0
        self.connpoints = []
        self.outputData = np.ones(SAMPLESIZE,dtype=np.int16)*880
        for ind in range(len(self.inputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.inputs[ind],"in",pos,glm.vec3([0.35,-0.14*ind+0.13,0])))  
        for ind in range(len(self.outputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.outputs[ind],"out",pos,glm.vec3([-0.35,-0.14*ind+0.13,0])))
        


    def update(self,fpsCounter,audioHandler):
        for cp in self.connpoints:
            cp.updatePos(self.model.pos)
    
    def audioUpdate(self,fpsCounter,audioHandler):
        self.connpoints[1].data = np.zeros(SAMPLESIZE, dtype=np.float32)
        self.connpoints[2].data = np.zeros(SAMPLESIZE, dtype=np.float32)
        self.connpoints[3].data = np.zeros(SAMPLESIZE, dtype=np.float32)
        if self.connpoints[0].data is not None:
            self.connpoints[1].data = np.copy(self.connpoints[0].data)
            self.connpoints[2].data = np.copy(self.connpoints[0].data)
            self.connpoints[3].data = np.copy(self.connpoints[0].data)

class EnvelopeNode(NodeElement):
    def __init__(self,ph,name, pos):
        """Inputs: clock. Output: envelope signal"""
        
        NodeElement.__init__(self,ph,{"name":name, "pos":pos,"rot":[1.57,0,0],"scale":0.3,"file":"res/inputsmall.obj","texture":"res/envelopenode.png"})
        self.inputs = ["clock"]
        self.outputs = ["envelope"]
        self.connpoints = []
        self.lastInputValue = 0
        self.state = "Z"
        self.states = ["A","D","S","R","Z"]
        self.properties = {"A":200,"D":200,"S":200,"R":200,"SLevel":0.5,"currentS":"Z"}
        self.propertyTypes = {"A":"integer","D":"integer","S":"integer","SLevel":"float","R":"integer","currentS":"text"}
        self.envvalue = 0
        self.statetimer = 0
        self.outputData = np.ones(SAMPLESIZE,dtype=np.int16)*880
        for ind in range(len(self.inputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.inputs[ind],"in",pos,glm.vec3([0.35,-0.14*ind,0])))  
        for ind in range(len(self.outputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.outputs[ind],"out",pos,glm.vec3([-0.35,-0.14*ind,0])))
        


    def update(self,fpsCounter,audioHandler):
        for cp in self.connpoints:
            cp.updatePos(self.model.pos)
    
    def audioUpdate(self,fpsCounter,audioHandler):
        self.connpoints[1].data = np.zeros(SAMPLESIZE, dtype=np.float32)
        if self.connpoints[0].data is not None:
            for idx, x in np.ndenumerate(self.connpoints[0].data):
                if idx[0] == 0:
                    if x > self.lastInputValue:
                        self.state = "A"
                    #elif x < self.lastInputValue:
                    #    self.state = "R"
                if x > self.connpoints[0].data[idx[0]-1]:
                    self.state = "A"
                #elif x < self.connpoints[0].data[idx[0]-1]:
                    #self.state = "R"
                if self.state == "A":
                    self.envvalue += 1/self.properties["A"]
                    if self.envvalue >= 1:
                        self.envvalue = 1.0
                        self.state = "D"
                elif self.state == "D":
                    self.envvalue -= self.properties["SLevel"]/self.properties["D"]
                    if self.envvalue <= self.properties["SLevel"]:
                        self.state = "S"
                        self.statetimer = self.properties["S"]
                elif self.state == "S":
                    self.envvalue = self.properties["SLevel"]
                    self.statetimer -= 1
                    if self.statetimer <=0:
                        self.state = "R"
                elif self.state == "R":
                    self.envvalue -= self.properties["S"]/self.properties["R"]
                    if self.envvalue <= 0.0:
                        self.envvalue = 0.0
                        self.state = "Z"
                self.connpoints[1].data[idx] = self.envvalue
            self.properties["currentS"] = self.state

            #self.connpoints[1].data = np.copy(self.connpoints[0].data)
            self.lastInputValue = self.connpoints[0].data[-1]

class DelayNode(NodeElement):
    def __init__(self,ph,name, pos):
        """Inputs: signal, wet. Output: signal+lastsignal*wet"""
        
        NodeElement.__init__(self,ph,{"name":name, "pos":pos,"rot":[1.57,0,0],"scale":0.3,"file":"res/inputsmall.obj","texture":"res/delaynode.png"})
        self.inputs = ["in","wet"]
        self.outputs = ["signal"]
        self.lastSent = 0
        self.connpoints = []
        self.properties = {"frames":5}
        self.propertyTypes = {"frames":"integer"}
        for ind in range(len(self.inputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.inputs[ind],"in",pos,glm.vec3([0.35,-0.14*ind+0.13,0])))  
        for ind in range(len(self.outputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.outputs[ind],"out",pos,glm.vec3([-0.35,-0.1*ind,0])))
        self.defwet = 0.5
        self.inconn = self.connpoints[0]
        self.wetconn = self.connpoints[1]
        self.frameDelay = 5
        self.delayData = []
        for x in range(self.properties["frames"] ):
            self.delayData.append(np.zeros(SAMPLESIZE,dtype=np.float32))
        self.currentDelayFrame = 0



    def update(self,fpsCounter,audioHandler):
        for cp in self.connpoints:
            cp.updatePos(self.model.pos)
        if self.changedProperty:
            self.updateProperty()
    
    def audioUpdate(self,fpsCounter,audioHandler):
        outsig = self.connpoints[2]
        outsig.data = None

        
        if self.wetconn.data is not None:
            multiplier = constraint(self.wetconn.data[0],0,1)
            # TODO: file in, speaker out, wet sinewave
            # multiplier cant be negative
            self.wetconn.data = None
        else:
            multiplier = self.defwet
        if self.inconn.data is not None:
            outsig.data = np.array(self.inconn.data*(1-multiplier),dtype=np.float32)
            outsig.data+= self.delayData[self.currentDelayFrame]*multiplier
            self.delayData[self.currentDelayFrame] = np.copy(outsig.data)#np.copy(self.inconn.data)
            self.currentDelayFrame = (self.currentDelayFrame + 1) % self.frameDelay
            self.inconn.data = None
    
    def updateProperty(self):
        if self.properties["frames"] < 1:
            self.properties["frames"] = 1
            self.frameDelay = int(self.properties["frames"])
            self.changedProperty = False
            return
        if self.properties["frames"] > 300:
            self.properties["frames"] = 300
            self.frameDelay = int(self.properties["frames"])
            self.changedProperty = False
            return
        if self.properties["frames"] == self.frameDelay:
            self.changedProperty = False
            return
        if self.properties["frames"] > self.frameDelay:
            for x in range(int(self.properties["frames"])- self.frameDelay):
                self.delayData.append(np.zeros(SAMPLESIZE,dtype=np.float32))
        if self.properties["frames"] < self.frameDelay:
            self.delayData = self.delayData[:int(self.properties["frames"])]
            self.currentDelayFrame = self.currentDelayFrame % int(self.properties["frames"])
        self.frameDelay = int(self.properties["frames"])
        self.changedProperty = False
        

class LinearAnim(NodeElement):
    """Inputs: None. Output: constant*np.ones(n)"""
    def __init__(self,ph,name, pos):   
        NodeElement.__init__(self,ph,{"name":name, "pos":pos,"rot":[1.57,0,0],"scale":0.3,"file":"res/inputsmall.obj","texture":"res/linearanimnode.png"})
        self.inputs = []
        self.outputs = ["signal"]
        self.lastSent = 0
        self.animTime = 0
        self.connpoints = []
        
        self.properties = {"from":100,"to":10,"time":1,"enabled":False,"repeat":False}
        
        self.propertyTypes = {"from":"float","to":"float","time":"float","enabled":"bool","repeat":"bool"}
        self.enabled = False

        self.value = self.properties["from"]
        self.textDisplay = TextDisplay(ph, {"pos":pos,"rot":[0,3.1415,0],"scale":0.2,"posoffset":[0.25,-0.1,-0.1]})

        self.oneData = np.ones(SAMPLESIZE,dtype=np.float32)
        for ind in range(len(self.inputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.inputs[ind],"in",pos,glm.vec3([0.35,-0.1*ind,0])))  
        for ind in range(len(self.outputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.outputs[ind],"out",pos,glm.vec3([-0.34,-0.1*ind-0.2,0])))
        

    def updateProperty(self):
        if self.properties["enabled"]:
            if self.enabled:
                if self.animTime>=self.properties["time"]:
                    self.animTime = 0
                if self.value < self.properties["from"] or self.value > self.properties["to"]:
                    self.value = self.properties["from"]
            else:
                self.animTime = 0
                self.enabled = True
        else:
            self.enabled = False
        #self.outputData = np.ones(SAMPLESIZE,dtype=np.int16)*self.properties["value"]
        self.changedProperty = False
    
    def update(self,fpsCounter,audioHandler):
        if self.changedProperty:
            self.updateProperty()
        for cp in self.connpoints:
            cp.updatePos(self.model.pos)
        self.textDisplay.updatePos(self.model.pos)
        if self.changedProperty:
            self.updateProperty()
        if self.enabled:
            self.animTime += fpsCounter.deltaTime
            self.value = lerpConst(self.properties["from"],self.properties["to"],self.animTime/self.properties["time"])
            if self.animTime>=self.properties["time"]:
                if not self.properties["repeat"]:
                    self.enabled = False
                    self.properties["enabled"] = False
                else:
                    self.animTime = 0

           
    def audioUpdate(self,fpsCounter,audioHandler):
        outsig = self.connpoints[0]
        
        
        if outsig.bezier != None:
            outsig.data = self.oneData*self.value#np.frombuffer(self.oneData*self.value, dtype=np.int16)

    def drawText(self,fontHandler,renderer,viewMat):
        if 0<self.value<100:
            fontHandler.drawText3D(str(round(self.value,1)),self.textDisplay.model.modelMat,0.05,viewMat,renderer)
        else:
            fontHandler.drawText3D(str(int(self.value)),self.textDisplay.model.modelMat,0.05,viewMat,renderer)

class SequencerNode(NodeElement):
    """Inputs: clock. Output: constant*np.ones(n)"""
    def __init__(self,ph,name, pos):
        NodeElement.__init__(self,ph,{"name":name, "pos":pos,"rot":[1.57,0,0],"scale":0.3,"file":"res/inputsmall.obj","texture":"res/sequencernode.png"})
        self.inputs = ["clock"]
        self.outputs = ["signal"]
        self.lastSent = 0
        self.connpoints = []
        self.properties = {"steps":4,"freq1":440,"freq2":440,"freq3":440,"freq4":440}
        self.propertyTypes = {"steps":"integer","freq1":"float","freq2":"float","freq3":"float","freq4":"float"}
        self.stepNames = ["freq1","freq2","freq3","freq4"]
        self.currentData = np.ones(SAMPLESIZE,dtype=np.float32)*self.properties["freq1"]
        for ind in range(len(self.inputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.inputs[ind],"in",pos,glm.vec3([0.35,-0.14*ind,0])))  
        for ind in range(len(self.outputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.outputs[ind],"out",pos,glm.vec3([-0.35,-0.1*ind,0])))
        self.clock = self.connpoints[0]
        self.outsig = self.connpoints[1]
        self.lastval = 0
        self.currentStep = 0

    def update(self,fpsCounter,audioHandler):
        for cp in self.connpoints:
            cp.updatePos(self.model.pos)
        if self.changedProperty:
            self.updateProperty()
    
    def audioUpdate(self,fpsCounter,audioHandler):
        self.outsig.data = None
        
        if self.clock.data is not None:
            maxval = np.amax(self.clock.data)
            if maxval != self.lastval:
                self.currentStep=(self.currentStep+1)%self.properties["steps"]
                self.currentData = np.ones(SAMPLESIZE,dtype=np.float32)*self.properties[self.stepNames[self.currentStep]]
                self.lastval = maxval

        self.outsig.data = self.currentData
    
    def updateProperty(self):
        if self.properties["steps"] < 1:
            self.properties["steps"] = 1
            self.currentStep = 0
            self.changedProperty = False
            return
        if self.properties["steps"] > 4:
            self.properties["steps"] = 4
            self.currentStep = 0
            self.changedProperty = False
            return
        self.properties["steps"] = int(self.properties["steps"])
        self.currentData = np.ones(SAMPLESIZE,dtype=np.float32)*self.properties["freq1"]
        self.changedProperty = False
        


class SpeakerOut(NodeElement):
    def __init__(self,ph,name, pos):
        """Inputs: output signal"""
        
        NodeElement.__init__(self,ph,{"name":name, "pos":pos,"rot":[1.57,0,0],"scale":0.3,"file":"res/input.obj","texture":"res/speaker.png"})
        self.inputs = ["signal"]
        self.connpoints = []
        self.lastSentTime = 0
        self.properties = {"volume":100}
        self.propertyTypes = {"volume":"float"}

        self.outputSample = np.array([],dtype=np.int16)
        for ind in range(len(self.inputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.inputs[ind],"in",pos,glm.vec3([0.55,-0.1*ind,0])))


    def update(self,fpsCounter,audioHandler):
        for cp in self.connpoints:
            cp.updatePos(self.model.pos)
        
        if self.changedProperty:
            self.updateProperty()
        
        if not audioHandler.dataReady:

            if not self.connpoints[0].data is None:
                audioHandler.dataPlaying = (self.connpoints[0].data*(self.properties["volume"]/100)).astype(np.int16).tobytes()
                audioHandler.dataReady = True
                if self.connpoints[0].bezier is not None:
                    self.connpoints[0].bezier.fromConnPoint.data = None

    def updateProperty(self):
        if self.properties["volume"]>100:
            self.properties["volume"] = 100
        if self.properties["volume"]<0:
            self.properties["volume"] = 0
        self.changedProperty = False



        

        
        

class BezierCurve:
    def __init__(self,ph,fcp,tcp):
        """Bezier curve for connecting the display elements."""
        self.name = "bez1"
        self.model = ph.loadFile("res/curve.obj","res/crystal.png")
        self.model.SetScale(0.02)
        self.model.SetPosition(np.array([0,0,0]))
        self.model.SetRotation(np.array([1.57,0,1.57]))
        self.model.defaultPosition = self.model.pos.copy()
        self.fromPos = np.array([0,0,0])
        self.toPos = np.array([100,50,0])
        self.fromConnPoint = fcp
        self.toConnPoint = tcp

    def draw(self,shaderhandler,renderer,viewMat):
        shader = shaderhandler.getShader("bezier")
        shader.Bind()
        self.model.texture.Bind()
        shader.SetUniform1i("u_Texture",0)
        #mvp = np.transpose(np.matmul(viewMat,self.model.modelMat))     
        shader.SetUniform3f("u_from",self.fromPos[0],-self.fromPos[1],self.fromPos[2])
        shader.SetUniform3f("u_to",self.toPos[0],-self.toPos[1],self.toPos[2])   
        #shader.SetUniformMat4f("u_MVP", mvp)
        shader.SetUniformMat4f("u_VP", np.transpose(viewMat))
        shader.SetUniformMat4f("u_Model", np.transpose(self.model.modelMat))
        
        renderer.Draw(self.model.va,self.model.ib,shader)
        #
        #self.model.DrawWithShader(shaderhandler.getShader("bezier"),renderer,viewMat)
    def update(self,fpsCounter,audioHandler):
        if self.fromConnPoint != None:
            self.fromPos = glm.vec3(self.fromConnPoint.model.pos)*50
        if self.toConnPoint != None:
            self.toPos = glm.vec3(self.toConnPoint.model.pos)*50
        if self.toConnPoint != None and self.fromConnPoint != None:
            self.toConnPoint.data = self.fromConnPoint.data
    


class Camera:
    def __init__(self):
        """Basic camera wrapper, for position
        """
        self.zoomLevel = -2
        self.pos = glm.vec3(0,0,self.zoomLevel)
        
        self.savedPos = glm.vec3(0,0,self.zoomLevel)

        self.rot = glm.vec3(0,0,0)
        self.camModel = None

        self.update(0,0)
    def draw(self,shaderhandler,renderer,viewMat):
        pass
    def changeZoom(self,dir):
        self.zoomLevel += dir*0.1
        if self.zoomLevel>-1.1:
            self.zoomLevel = -1.1
        if self.zoomLevel<-8:
            self.zoomLevel = -8
        self.pos = glm.vec3(self.pos.x,self.pos.y,self.zoomLevel)
        self.update(0,0)
    def update(self,fpsCounter,audioHandler):
        """
        rotz = pyrr.matrix44.create_from_z_rotation(self.rot[2])
        rotx = pyrr.matrix44.create_from_x_rotation(self.rot[0])
        rot = np.matmul(np.matmul(pyrr.matrix44.create_from_y_rotation(self.rot[1]),rotz),rotx)
        self.camModel = np.matmul(rot,np.transpose(pyrr.matrix44.create_from_translation(self.pos)))
        """
        
        self.camModel=glm.lookAt(self.pos, self.pos + glm.vec3(0,0,1) , glm.vec3(0,1,0))
        
class PropertyMenu:
    def __init__(self):
        """
            Property menu opening from the right
        """
        self.isOpen = True
        self.backgroundTexture = Texture("res/1px.png")
        self.points = np.array([0.5, -1, 0.0, 0.0,
                                1, -1, 1.0, 0.0,
                                1, 1, 1.0, 1.0,
                                0.5, 1, 0.0, 1.0],dtype='float32')
        self.indices = np.array([0,1,2, 2,3,0])
        self.va = VertexArray()
        self.vb = VertexBuffer(self.points)
        self.layout = VertexBufferLayout()
        self.layout.PushF(2)
        self.layout.PushF(2)
        self.va.AddBuffer(self.vb, self.layout)
        self.ib = IndexBuffer(self.indices, 6)
        self.shader = None
        self.lastChangeTime = 0
        self.blinkOn = True
        self.selectedProperty = -1
        self.string = "80"
        self.propertyList = []
        self.dialogOpen = False

        self.activeNodeName = "aaa"

    def openFileDialog(self):
        root = tk.Tk()
        root.withdraw()

        path = ""

        file_path = tk.filedialog.askopenfilename(title="Open a music file",filetypes=[('Wave files','*.wav')])
        if file_path == "":
            path = "<not selected>"
        else:
            path = os.path.relpath(file_path)
        return path
       

    def draw(self,shaderhandler,renderer,fontHandler):
        """
        Draw current properties for selected node
        """
        if self.shader is None:
            self.shader = shaderhandler.getShader("propertyMenu")
        self.backgroundTexture.Bind()
        self.shader.Bind()

        self.shader.SetUniform1i("u_Texture",0)
        self.shader.SetUniform1i("u_time",0)
        self.shader.SetUniform1f("xcoord",0)
        renderer.Draw(self.va,self.ib,self.shader)

        fontHandler.drawText(self.activeNodeName.upper(),0.55,0.4,0.05,renderer)
        yoffset = 0.2
        for prop in self.propertyList:
            value = self.propertyList[prop]
            isSelected = False
            if self.selectedProperty > -1:
                isSelected = list(self.propertyList)[self.selectedProperty] == prop
            if isSelected:
                value = self.string
            if self.propertyTypes[prop] == "bool":
                if value:
                    value = "☒"
                else:
                    value = "☐"
            if self.propertyTypes[prop] == "button":
                fontHandler.drawText(prop,0.65,yoffset,0.03,renderer)
                yoffset -= 0.05
                continue
            if self.blinkOn or not isSelected:
                fontHandler.drawText(prop+": "+str(value),0.55,yoffset,0.03,renderer)
            else:
                fontHandler.drawText(prop+": ",0.55,yoffset,0.03,renderer)
            yoffset -= 0.05

    def update(self,fpsCounter,audioHandler,inputHandler,activeNode):
        self.propertyList = activeNode.properties
        self.propertyTypes = activeNode.propertyTypes
            
        if self.selectedProperty > -1:
            if self.propertyTypes[list(self.propertyList)[self.selectedProperty]] == "file":
                # someone sad mktkinter library is maybe an option
                activeNode.properties[list(activeNode.properties)[self.selectedProperty]] = self.openFileDialog()
                activeNode.changedProperty = True
                self.selectedProperty = -1
                self.blinkOn = True
                return
            
            if self.propertyTypes[list(self.propertyList)[self.selectedProperty]] == "bool":
                activeNode.properties[list(activeNode.properties)[self.selectedProperty]] = not activeNode.properties[list(activeNode.properties)[self.selectedProperty]]
                activeNode.changedProperty = True
                self.selectedProperty = -1
                return
            
            if self.propertyTypes[list(self.propertyList)[self.selectedProperty]] == "button":
                
                activeNode.properties[list(activeNode.properties)[self.selectedProperty]]()
                self.selectedProperty = -1
                return

            if fpsCounter.currentTime-self.lastChangeTime>0.2:
                self.lastChangeTime = fpsCounter.currentTime
                self.blinkOn = not self.blinkOn
            for n in "0123456789.":
                if inputHandler.isKeyDown(str.encode(n)):
                    self.string += n
            if inputHandler.isKeyDown(b'\x08'):
                # b'\x08' is backspace
                self.string = self.string[:-1]
            if inputHandler.isKeyDown(b'\r'):
                # b'\r' is return
                try:
                    proptype = activeNode.propertyTypes[list(activeNode.properties)[self.selectedProperty]]
                    if proptype == "integer":
                        activeNode.properties[list(activeNode.properties)[self.selectedProperty]] = int(self.string)
                    if proptype == "float":
                        activeNode.properties[list(activeNode.properties)[self.selectedProperty]] = float(self.string)
                    #if proptype == "bool":
                    #    activeNode.properties[list(activeNode.properties)[self.selectedProperty]] = int(self.string)
                    activeNode.changedProperty = True
                    self.propertyList = activeNode.properties
                except Exception as err:
                    print("not valid number")
                    print(err)
                    
                self.selectedProperty = -1
                self.blinkOn = True
        if activeNode is not None:
            self.activeNodeName = (type(activeNode).__name__)
            self.propertyList = activeNode.properties
    def checkPropertyClick(self,mouseX,mouseY):
        if self.selectedProperty == -1:
            index = int(np.ceil((0.2-mouseY)/0.05))
            if(index>-1 and index<len(self.propertyList)):
                self.selectedProperty = index
                self.string = str(self.propertyList[list(self.propertyList)[self.selectedProperty]])
                

class AddMenu:
    def __init__(self):
        """
            Add menu opening from the left
        """
        self.isOpen = True
        self.backgroundTexture = Texture("res/addmenu.png")
        self.points = np.array([-1, -1, 0.0, 0.0,
                                -0.9, -1, 1.0, 0.0,
                                -0.9, 1, 1.0, 1.0,
                                -1, 1, 0.0, 1.0],dtype='float32')
        self.indices = np.array([0,1,2, 2,3,0])
        self.va = VertexArray()
        self.vb = VertexBuffer(self.points)
        self.layout = VertexBufferLayout()
        self.layout.PushF(2)
        self.layout.PushF(2)
        self.va.AddBuffer(self.vb, self.layout)
        self.ib = IndexBuffer(self.indices, 6)
        self.shader = None
        self.lastChangeTime = 0
        self.blinkOn = True
        self.selectedItem = -1
        self.string = "80"
        self.elementCount = 12
        self.displayElements = 11
        self.scrollOffset = 0
        self.selectableClasses = [FilePlayer,SineGenerator,SquareGenerator,ConstantNode,MixerNode,DelayNode,SplitterNode,LinearAnim,SequencerNode,FilterNode,NoiseNode,EnvelopeNode]

        self.activeNodeName = "aaa"

    def draw(self,shaderhandler,renderer,fontHandler):
        """
        Draw current properties for selected node
        """
        if self.shader is None:
            self.shader = shaderhandler.getShader("addMenu")
        self.backgroundTexture.Bind()
        self.shader.Bind()

        self.shader.SetUniform1i("u_Texture",0)
        self.shader.SetUniform1f("u_time",0)
        self.shader.SetUniform1f("xcoord",0)
        self.shader.SetUniform1f("selectedIndex",-1)#self.selectedItem/11+1/22)
        self.shader.SetUniform1f("ymax",self.elementCount/self.displayElements)
        self.shader.SetUniform1f("scrollOffset",self.scrollOffset)
        
        
        renderer.Draw(self.va,self.ib,self.shader)

        #fontHandler.drawText(self.activeNodeName.upper(),0.55,0.4,0.05,renderer)

    def scroll(self, direction):
        self.scrollOffset += direction/self.elementCount
        if self.scrollOffset<0:
            self.scrollOffset = 0
        if self.scrollOffset>(self.elementCount-self.displayElements)/self.elementCount:
            self.scrollOffset = (self.elementCount-self.displayElements)/self.elementCount

    def update(self,fpsCounter,audioHandler,inputHandler,activeNode):
        return
        
    def checkAddClick(self,mouseX,mouseY):
        ind = math.floor((2-(mouseY+1))*11/2+self.scrollOffset*self.elementCount)
        if(ind < self.elementCount):
            self.selectedItem = ind
            return self.selectableClasses[self.selectedItem]
            # change element

        else:
            self.selectedItem = -1
        return None
        

    

        

class BackgroundPlane:
    """
    Deskmat textured background
    """
    def __init__(self):
        #self.backgroundTexture = Texture("res/1px.png")
        self.points = np.array([-1, -1, 0.0, 0.0,
                                1, -1, 1.0, 0.0,
                                1, 1, 1.0, 1.0,
                                -1, 1, 0.0, 1.0],dtype='float32')
        self.indices = np.array([0,1,2, 2,3,0])
        self.va = VertexArray()
        self.vb = VertexBuffer(self.points)
        self.layout = VertexBufferLayout()
        self.layout.PushF(2)
        self.layout.PushF(2)
        self.va.AddBuffer(self.vb, self.layout)
        self.ib = IndexBuffer(self.indices, 6)
        
        #self.im = Image.open("res/constantnode.png")
        
        #self.Width, self.Height = self.im.size
        
        self.average = np.zeros(SAMPLESIZE//2+1,dtype=np.int16)

        # SAMPLESIZE = 3072*2 = 512*3*4
        self.twidth = SAMPLESIZE//16 # 192
        self.theight = 1
        self.buffer = (np.ones(SAMPLESIZE//8,dtype=np.int16)).tobytes()#np.zeros(SAMPLESIZE//2,dtype=np.int16).tobytes()

        self.datafrom = pyfftw.empty_aligned(SAMPLESIZE, dtype='float32') # 3072
        self.datato = pyfftw.empty_aligned(SAMPLESIZE//2+1, dtype='complex64') # 1537

        self.fft_object = pyfftw.FFTW(self.datafrom, self.datato)
        self.window = np.hanning(SAMPLESIZE)
        self.logarr = np.logspace(0,np.log10(SAMPLESIZE//8),SAMPLESIZE//8,endpoint=False).astype(np.int16)

        pyfftw.interfaces.cache.enable()

        self.RendererId = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D,self.RendererId)
        #self.buffer = self.im.tobytes("raw", "RGBA", 0, -1)
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,GL_NEAREST)
        
        #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE)
        #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)


        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.twidth, self.theight, 0, GL_RGBA, GL_UNSIGNED_BYTE, self.buffer)
        glBindTexture(GL_TEXTURE_2D, 0)


    def draw(self,shaderhandler,renderer,camera,audiohandler):
        """
        Draw current properties for selected node
        """
        if audiohandler.dataReady:
            #print(len(audiohandler.dataPlaying))
            self.datafrom[:] = np.frombuffer(audiohandler.dataPlaying, dtype=np.int16).astype(np.float32)*self.window
            self.fft_object()
            values = (np.absolute(self.datato)/1000).astype(np.int16)#(np.log2(np.absolute(self.datato))*30).astype(np.int16)
            # max around 32 000 000?
            #cmax = np.max(np.absolute(self.datato))
            #self.average = (self.average*0.9).astype(np.int16)
            #self.average += values//2
            self.buffer = values[:SAMPLESIZE//8].tobytes()#self.average.tobytes()
            #print(len(self.buffer))
            glBindTexture(GL_TEXTURE_2D,self.RendererId)
            #self.buffer = audiohandler.dataPlaying
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.twidth, self.theight, 0, GL_RGBA, GL_UNSIGNED_BYTE, self.buffer)
            #glBindTexture(GL_TEXTURE_2D, 0)
        
        self.shader = shaderhandler.getShader("background")
        #self.backgroundTexture.Bind()
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.RendererId)
        self.shader.Bind()

        self.shader.SetUniform1i("u_Texture",0)
        self.shader.SetUniform1f("u_time",time.perf_counter())
        #self.shader.SetUniform1f("xcoord",0)
        #self.shader.SetUniform3f("xyzoom",camera.pos.x,camera.pos.y,camera.zoomLevel)
        
        renderer.Draw(self.va,self.ib,self.shader)
