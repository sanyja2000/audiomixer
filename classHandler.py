import math

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
#import pyfftw
"""
This file contains the classes for different types of objects in the map files.
Classes are required to have a draw function and optionally an update and moveWithKeys function.


"""

SAMPLESIZE = 6144 # Must be even, choose higher for better performance


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
    def checkIntersect(self,x,y):
        if((self.model.pos[0]-x)**2+(self.model.pos[1]-y)**2<0.007):
            return True
        return False
    def updatePos(self,pos):
        self.model.SetPosition(pos+self.positionOffset)

class SineGenerator(NodeElement):
    def __init__(self,ph,name, pos):
        """Inputs: amp. Output: sinewave signal"""
        
        NodeElement.__init__(self,ph,{"name":name, "pos":pos,"rot":[1.57,0,0],"scale":0.3,"file":"res/input.obj","texture":"res/sinewave.png"})
        self.inputs = ["freq","amp"]
        self.outputs = ["signal"]
        self.properties = {"offset":0}
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
        self.inputs = {"amp":0,"freq":4}
        self.outputs = {"signal":0}
        self.properties = {"offset":0}
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
        self.properties = {"file":name.split("/")[1][:7]+"...","cue":0,"cueLength":10,"speed":1}
        self.cueTime = 0
        self.cueEnabled = True
        self.cuePoint = 0
        self.wf = wave.open(name, 'rb')
        self.connpoints = []
        for ind in range(len(self.inputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.inputs[ind],"in",pos,glm.vec3([0.55,-0.1*ind,0])))  
        for ind in range(len(self.outputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.outputs[ind],"out",pos,glm.vec3([-0.55,-0.1*ind,0])))

    def updateProperty(self):
        if self.properties["cue"] != 0:
            if not self.cueEnabled:
                self.cuePoint = self.wf.tell()
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

class Keyboard(NodeElement):
    def __init__(self,ph,name, pos):
        """Inputs: None. Output: constant*np.ones(n)"""
        
        NodeElement.__init__(self,ph,{"name":name, "pos":pos,"rot":[1.57,0,0],"scale":0.3,"file":"res/inputsmall.obj","texture":"res/keyboardnode.png"})
        self.inputs = []
        self.outputs = ["freq","gate"]
        self.lastSent = 0
        self.connpoints = []
        self.properties = {"freq":261.63,"mult":1}
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
        self.lastSent = 0
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
        
                


class DelayNode(NodeElement):
    def __init__(self,ph,name, pos):
        """Inputs: signal, wet. Output: signal+lastsignal*wet"""
        
        NodeElement.__init__(self,ph,{"name":name, "pos":pos,"rot":[1.57,0,0],"scale":0.3,"file":"res/inputsmall.obj","texture":"res/delaynode.png"})
        self.inputs = ["in","wet"]
        self.outputs = ["signal"]
        self.lastSent = 0
        self.connpoints = []
        self.properties = {"frames":5}
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
        NodeElement.__init__(self,ph,{"name":name, "pos":pos,"rot":[1.57,0,0],"scale":0.3,"file":"res/inputsmall.obj","texture":"res/constantnode.png"})
        self.inputs = []
        self.outputs = ["signal"]
        self.lastSent = 0
        self.animTime = 0
        self.connpoints = []
        
        self.properties = {"from":100,"to":10,"time":1,"enabled":0,"repeat":0}
        self.enabled = False

        self.value = self.properties["from"]
        self.textDisplay = TextDisplay(ph, {"pos":pos,"rot":[0,3.1415,0],"scale":0.2,"posoffset":[0.25,-0.1,-0.1]})

        self.oneData = np.ones(SAMPLESIZE,dtype=np.float32)
        for ind in range(len(self.inputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.inputs[ind],"in",pos,glm.vec3([0.35,-0.1*ind,0])))  
        for ind in range(len(self.outputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.outputs[ind],"out",pos,glm.vec3([-0.34,-0.1*ind-0.2,0])))
        

    def updateProperty(self):
        if self.properties["enabled"] != 0 and not self.enabled:
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
                if self.properties["repeat"]==0:
                    self.enabled = False
                    self.properties["enabled"] = 0
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
        


class SpeakerOut(NodeElement):
    def __init__(self,ph,name, pos):
        """Inputs: output signal"""
        
        NodeElement.__init__(self,ph,{"name":name, "pos":pos,"rot":[1.57,0,0],"scale":0.3,"file":"res/input.obj","texture":"res/speaker.png"})
        self.inputs = ["signal"]
        self.connpoints = []
        self.lastSentTime = 0
        self.properties = {"volume":100}

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
        """Basic wrapper for decoration objects. These don't interact with anything."""
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

        self.activeNodeName = "aaa"

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

        fontHandler.drawText(self.activeNodeName.upper(),0.55,0.4,0.05,renderer,spacing=0.7)
        yoffset = 0.2
        for prop in self.propertyList:
            value = self.propertyList[prop]
            isSelected = False
            if self.selectedProperty > -1:
                isSelected = list(self.propertyList)[self.selectedProperty] == prop
            if isSelected:
                value = self.string
            if self.blinkOn or not isSelected:
                fontHandler.drawText(prop+": "+str(value),0.55,yoffset,0.05,renderer)
            else:
                fontHandler.drawText(prop+": ",0.55,yoffset,0.05,renderer)
            yoffset -= 0.1

    def update(self,fpsCounter,audioHandler,inputHandler,activeNode):
        self.propertyList = activeNode.properties
        if self.selectedProperty > -1:
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
                    activeNode.properties[list(activeNode.properties)[self.selectedProperty]] = float(self.string)
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
            index = int(np.ceil((0.2-mouseY)/0.1))
            if(index>-1 and index<len(self.propertyList)):
                self.selectedProperty = index
                self.string = str(self.propertyList[list(self.propertyList)[self.selectedProperty]])

    

        

class BackgroundPlane:
    """
    Deskmat textured background
    """
    def __init__(self):
        self.backgroundTexture = Texture("res/1px.png")
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
        self.shader = None

    def draw(self,shaderhandler,renderer,camera):
        """
        Draw current properties for selected node
        """
        if self.shader is None:
            self.shader = shaderhandler.getShader("background")
        self.backgroundTexture.Bind()
        self.shader.Bind()

        self.shader.SetUniform1i("u_Texture",0)
        self.shader.SetUniform1i("u_time",0)
        #self.shader.SetUniform1f("xcoord",0)
        self.shader.SetUniform3f("xyzoom",camera.pos.x,camera.pos.y,camera.zoomLevel)
        
        renderer.Draw(self.va,self.ib,self.shader)
