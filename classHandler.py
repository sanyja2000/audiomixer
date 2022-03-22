from email.mime import audio
import math
from turtle import position
from typing import Text
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
"""
This file contains the classes for different types of objects in the map files.
Classes are required to have a draw function and optionally an update and moveWithKeys function.


"""

SAMPLESIZE = 2048 # Must be even, choose higher for better performance


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


class Map:
    def __init__(self,ph,props):
        self.objFile = props["file"]
        self.name = props["name"]
        self.cardNum = props["cardNum"]
        self.model = ph.loadFile(props["file"],props["texture"],textureRepeat=True)
        self.model.SetScale(10)
        self.model.SetPosition(np.array(props["pos"]))
        self.model.SetRotation(np.array(props["rot"]))
        # vec4 points, (x, y, z, radius) for sphere which is cleared
        # maximum of 5 points
        self.maxPoints = np.ones((5,4))
        self.clearedPoints = np.array([[-5,-10.0,-5,2],[5,-10.0,-5,2]])
    def draw(self,shaderhandler,renderer,viewMat):
        points = []
        for x in range(len(self.maxPoints)):
            if x<len(self.clearedPoints):
                points.append(self.clearedPoints[x])
            else:
                points.append(self.maxPoints[x])
        parameters = {"u_Time":time.perf_counter(),"4fv,clearedPoints":np.array(points),"numPoints":len(self.clearedPoints)}
        self.model.DrawWithShader(shaderhandler.getShader("map"),renderer,viewMat,options=parameters)





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
        self.model.DrawWithShader(shaderhandler.getShader(self.shaderName),renderer,viewMat)
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

        self.shaderName = "default"
    def draw(self,shaderhandler,renderer,viewMat):
        self.model.DrawWithShader(shaderhandler.getShader(self.shaderName),renderer,viewMat)
    def update(self,fpsCounter,audioHandler):
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
        self.model.DrawWithShader(shaderhandler.getShader(self.shaderName),renderer,viewMat)
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
        self.inputs = ["amp","freq"]
        self.outputs = ["signal"]
        self.lastSent = 0
        self.processedFrames = 0
        self.deffreq = 440
        self.defamp = 1
        self.freq = self.deffreq
        self.connpoints = []
        for ind in range(len(self.outputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.outputs[ind],"out",pos,glm.vec3([-0.55,-0.1*ind,0])))
        for ind in range(len(self.inputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.inputs[ind],"in",pos,glm.vec3([0.55,-0.1*ind,0])))  

    def checkIntersect(self,x,y):
        for cp in self.connpoints:
            if cp.checkIntersect(x,y):
                return cp
        if self.model.pos[0]-self.model.scale*1 <x<self.model.pos[0]+self.model.scale*1:
            if self.model.pos[1]-self.model.scale*1 <y<self.model.pos[1]+self.model.scale*1:
                return self
        return None

    def draw(self,shaderhandler,renderer,viewMat):
        self.model.DrawWithShader(shaderhandler.getShader(self.shaderName),renderer,viewMat)
        for cp in self.connpoints:
            cp.draw(shaderhandler,renderer,viewMat)


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
            if cp.name == "signal" and cp.bezier != None:
                cur = fpsCounter.currentTime
                if not audioHandler.dataReady  and cur-self.lastSent>fpsCounter.deltaTime:
                    mult = self.freq/(44100/3.1415)
                    cp.data = np.sin((np.arange(SAMPLESIZE)+self.processedFrames)*mult)*1000
                    self.lastSent = cur
                    self.processedFrames += SAMPLESIZE
        


class SquareGenerator(NodeElement):
    def __init__(self,ph,name, pos):
        """Inputs: freq. Output: squarewave signal"""
        
        NodeElement.__init__(self,ph,{"name":name, "pos":pos,"rot":[1.57,0,0],"scale":0.3,"file":"res/input.obj","texture":"res/squarewave.png"})
        self.inputs = {"amp":0,"freq":4}
        self.outputs = {"signal":0}
        self.processedFrames = 0
        self.deffreq = 440
        self.defamp = 1
        self.freq = self.deffreq
        self.lastSent = 0
        self.connpoints = []
        for ind in range(len(self.inputs)):
            self.connpoints.append(ConnectionPoint(ph,self,list(self.inputs)[ind],"in",pos,glm.vec3([0.55,-0.1*ind,0])))  
        for ind in range(len(self.outputs)):
            self.connpoints.append(ConnectionPoint(ph,self,list(self.outputs)[ind],"out",pos,glm.vec3([-0.55,-0.1*ind,0])))
        
    def checkIntersect(self,x,y):
        for cp in self.connpoints:
            if cp.checkIntersect(x,y):
                return cp
        if self.model.pos[0]-self.model.scale*1 <x<self.model.pos[0]+self.model.scale*1:
            if self.model.pos[1]-self.model.scale*1 <y<self.model.pos[1]+self.model.scale*1:
                return self
        return None

    def draw(self,shaderhandler,renderer,viewMat):
        self.model.DrawWithShader(shaderhandler.getShader(self.shaderName),renderer,viewMat)
        for cp in self.connpoints:
            cp.draw(shaderhandler,renderer,viewMat)


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
            if cp.name == "signal" and cp.bezier != None:
                cur = fpsCounter.currentTime
                if not audioHandler.dataReady and cur-self.lastSent>fpsCounter.deltaTime:
                    mult = self.freq/(44100/3.1415)
                    cp.data = np.sign(np.sin((np.arange(SAMPLESIZE)+self.processedFrames)*mult))*1000
                    self.lastSent = cur
                    self.processedFrames += SAMPLESIZE
        


class FilePlayer(NodeElement):
    def __init__(self,ph,name, pos):
        """Inputs: None. Output: file signal"""
        
        NodeElement.__init__(self,ph,{"name":name, "pos":pos,"rot":[1.57,0,0],"scale":0.3,"file":"res/input.obj","texture":"res/inputfile.png"})
        self.inputs = []
        self.outputs = ["signal"]
        self.lastSent = 0
        self.wf = wave.open(name, 'rb')
        self.connpoints = []
        for ind in range(len(self.inputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.inputs[ind],"in",pos,glm.vec3([0.55,-0.1*ind,0])))  
        for ind in range(len(self.outputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.outputs[ind],"out",pos,glm.vec3([-0.55,-0.1*ind,0])))
        
    def checkIntersect(self,x,y):
        for cp in self.connpoints:
            if cp.checkIntersect(x,y):
                return cp
        if self.model.pos[0]-self.model.scale*1 <x<self.model.pos[0]+self.model.scale*1:
            if self.model.pos[1]-self.model.scale*1 <y<self.model.pos[1]+self.model.scale*1:
                return self
        return None

    def draw(self,shaderhandler,renderer,viewMat):
        self.model.DrawWithShader(shaderhandler.getShader(self.shaderName),renderer,viewMat)
        for cp in self.connpoints:
            cp.draw(shaderhandler,renderer,viewMat)


    def update(self,fpsCounter,audioHandler):
        for cp in self.connpoints:
            cp.updatePos(self.model.pos)
    
    def audioUpdate(self,fpsCounter,audioHandler):
        outsig = self.connpoints[0]
        if outsig.bezier != None:
            cur = fpsCounter.currentTime 
            if not audioHandler.dataReady and cur-self.lastSent>fpsCounter.deltaTime:
                outsig.data = self.wf.readframes(SAMPLESIZE//2)
                self.lastSent = cur
                if len(outsig.data)<2048:
                    self.wf.setpos(0)
                    outsig.data = self.wf.readframes(SAMPLESIZE//2)
                outsig.data = np.frombuffer(outsig.data, dtype=np.int16)

class ConstantNode(NodeElement):
    def __init__(self,ph,name, pos):
        """Inputs: None. Output: constant*np.ones(n)"""
        
        NodeElement.__init__(self,ph,{"name":name, "pos":pos,"rot":[1.57,0,0],"scale":0.3,"file":"res/input.obj","texture":"res/constantnode.png"})
        self.inputs = []
        self.outputs = ["signal"]
        self.lastSent = 0
        self.connpoints = []
        self.outputData = np.ones(SAMPLESIZE,dtype=np.int16)*880
        for ind in range(len(self.inputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.inputs[ind],"in",pos,glm.vec3([0.55,-0.1*ind,0])))  
        for ind in range(len(self.outputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.outputs[ind],"out",pos,glm.vec3([-0.55,-0.1*ind,0])))
        
    def checkIntersect(self,x,y):
        for cp in self.connpoints:
            if cp.checkIntersect(x,y):
                return cp
        if self.model.pos[0]-self.model.scale*1 <x<self.model.pos[0]+self.model.scale*1:
            if self.model.pos[1]-self.model.scale*1 <y<self.model.pos[1]+self.model.scale*1:
                return self
        return None

    def draw(self,shaderhandler,renderer,viewMat):
        self.model.DrawWithShader(shaderhandler.getShader(self.shaderName),renderer,viewMat)
        for cp in self.connpoints:
            cp.draw(shaderhandler,renderer,viewMat)


    def update(self,fpsCounter,audioHandler):
        for cp in self.connpoints:
            cp.updatePos(self.model.pos)
    def audioUpdate(self,fpsCounter,audioHandler):
        outsig = self.connpoints[0]
        if outsig.bezier != None:
            outsig.data = np.frombuffer(self.outputData, dtype=np.int16)
        

class AddNode(NodeElement):
    def __init__(self,ph,name, pos):
        """Inputs: A,B. Output: A+B"""
        
        NodeElement.__init__(self,ph,{"name":name, "pos":pos,"rot":[1.57,0,0],"scale":0.3,"file":"res/input.obj","texture":"res/constantnode.png"})
        self.inputs = ["A","B"]
        self.outputs = ["signal"]
        self.lastSent = 0
        self.connpoints = []
        self.outputData = np.ones(SAMPLESIZE,dtype=np.int16)*880
        for ind in range(len(self.inputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.inputs[ind],"in",pos,glm.vec3([0.55,-0.1*ind,0])))  
        for ind in range(len(self.outputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.outputs[ind],"out",pos,glm.vec3([-0.55,-0.1*ind,0])))
        
        self.aconn = self.connpoints[0]
        self.bconn = self.connpoints[1]
        
    def checkIntersect(self,x,y):
        for cp in self.connpoints:
            if cp.checkIntersect(x,y):
                return cp
        if self.model.pos[0]-self.model.scale*1 <x<self.model.pos[0]+self.model.scale*1:
            if self.model.pos[1]-self.model.scale*1 <y<self.model.pos[1]+self.model.scale*1:
                return self
        return None

    def draw(self,shaderhandler,renderer,viewMat):
        self.model.DrawWithShader(shaderhandler.getShader(self.shaderName),renderer,viewMat)
        for cp in self.connpoints:
            cp.draw(shaderhandler,renderer,viewMat)


    def update(self,fpsCounter,audioHandler):
        for cp in self.connpoints:
            cp.updatePos(self.model.pos)
    
    def audioUpdate(self,fpsCounter,audioHandler):
        outsig = self.connpoints[2]
        outsig.data = np.ones(SAMPLESIZE, dtype=np.int16)
        if self.aconn.data is not None:
            np.add(outsig.data, self.aconn.data//2, out=outsig.data, casting="unsafe")
            #outsig.data = np.maximum(self.aconn.data, outsig.data)
        if self.bconn.data is not None:
            np.add(outsig.data, self.bconn.data//2, out=outsig.data, casting="unsafe")
            #outsig.data = np.maximum(self.bconn.data, outsig.data)
        """
        if outsig.bezier != None:
            cur = fpsCounter.currentTime
            
            
            if not audioHandler.dataReady and cur-self.lastSent>fpsCounter.deltaTime:
                outsig.data = np.zeros(SAMPLESIZE, dtype=np.int16)
                if self.aconn.data is not None:
                    np.add(outsig.data, self.aconn.data, out=outsig.data, casting="unsafe")
                    if self.aconn.bezier is not None:
                        self.aconn.bezier.fromConnPoint.data = None
                    #outsig.data += self.aconn.data//2
                if self.bconn.data is not None:
                    np.add(outsig.data, self.bconn.data, out=outsig.data, casting="unsafe")
                    if self.bconn.bezier is not None:
                        self.bconn.bezier.fromConnPoint.data = None
                    #outsig.data += self.bconn.data//2
                self.lastSent = cur
        """
                



class SpeakerOut(NodeElement):
    def __init__(self,ph,name, pos):
        """Inputs: freq,amp. Output: wave signal"""
        
        NodeElement.__init__(self,ph,{"name":name, "pos":pos,"rot":[1.57,0,0],"scale":0.3,"file":"res/input.obj","texture":"res/speaker.png"})
        self.inputs = ["signal"]
        self.connpoints = []
        self.lastSentTime = 0

        self.outputSample = np.array([],dtype=np.int16)
        for ind in range(len(self.inputs)):
            self.connpoints.append(ConnectionPoint(ph,self,self.inputs[ind],"in",pos,glm.vec3([0.55,-0.1*ind,0])))
        
    def checkIntersect(self,x,y):
        for cp in self.connpoints:
            if cp.checkIntersect(x,y):
                return cp
        if self.model.pos[0]-self.model.scale*1 <x<self.model.pos[0]+self.model.scale*1:
            if self.model.pos[1]-self.model.scale*1 <y<self.model.pos[1]+self.model.scale*1:
                return self
        return None

    def draw(self,shaderhandler,renderer,viewMat):
        self.model.DrawWithShader(shaderhandler.getShader(self.shaderName),renderer,viewMat)
        for cp in self.connpoints:
            cp.draw(shaderhandler,renderer,viewMat)


    def update(self,fpsCounter,audioHandler):
        for cp in self.connpoints:
            cp.updatePos(self.model.pos)
        
        if not audioHandler.dataReady:

            if not self.connpoints[0].data is None:
                audioHandler.dataPlaying = self.connpoints[0].data.astype(np.int16).tobytes()
                audioHandler.dataReady = True
                if self.connpoints[0].bezier is not None:
                    self.connpoints[0].bezier.fromConnPoint.data = None



        

        
        

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
        """Basic camera wrapper
            syntax: {"name":"camera1","type":"camera","movement":"fixed","pos":[0,1,0],"rot":[0,0,0]}
        """
        #self.pos = np.array([0,0,-1],dtype="float64")
        self.pos = glm.vec3(0,0,-2)
        self.savedPos = glm.vec3(0,0,-2)
        #self.rot = np.array([0,0,0])
        self.rot = glm.vec3(0,0,0)
        self.camModel = None
        #self.defaultPosition = np.array(props["pos"])
        self.update(0,0)
    def draw(self,shaderhandler,renderer,viewMat):
        pass
    def update(self,fpsCounter,audioHandler):
        """
        rotz = pyrr.matrix44.create_from_z_rotation(self.rot[2])
        rotx = pyrr.matrix44.create_from_x_rotation(self.rot[0])
        rot = np.matmul(np.matmul(pyrr.matrix44.create_from_y_rotation(self.rot[1]),rotz),rotx)
        self.camModel = np.matmul(rot,np.transpose(pyrr.matrix44.create_from_translation(self.pos)))
        """
        self.camModel=glm.lookAt(self.pos, self.pos + glm.vec3(0,0,1) , glm.vec3(0,1,0))
        
