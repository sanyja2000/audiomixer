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
from OpenGL.GLUT import *
import glm
"""
This file contains the classes for different types of objects in the map files.
Classes are required to have a draw function and optionally an update and moveWithKeys function.


"""



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
    def update(self,deltaTime,audioHandler):
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
    def update(self,deltaTime,audioHandler):
        pass

class ConnectionPoint:
    def __init__(self,ph,parent,name,side,pos,posoffset):
        """Connection points for inputs, outputs."""
        self.side = side
        self.name = name
        self.parent = parent
        self.model = ph.loadFile("res/torus.obj","res/crystal.png")
        self.model.SetScale(0.03)
        self.model.SetPosition(glm.vec3(pos)+glm.vec3(posoffset))
        self.model.SetRotation(np.array([1.57,0,0]))
        self.positionOffset = posoffset
        self.shaderName = "default"
    def draw(self,shaderhandler,renderer,viewMat):
        self.model.DrawWithShader(shaderhandler.getShader(self.shaderName),renderer,viewMat)
    def checkIntersect(self,x,y):
        if((self.model.pos[0]-x)**2+(self.model.pos[1]-y)**2<0.007):
            return True
        return False
    def updatePos(self,pos):
        self.model.SetPosition(pos+self.positionOffset)

class WaveGenerator(NodeElement):
    def __init__(self,ph,name, type, pos):
        """Inputs: freq,amp. Output: wave signal"""
        
        NodeElement.__init__(self,ph,{"name":name, "pos":pos,"rot":[1.57,0,0],"scale":0.3,"file":"res/input.obj","texture":"res/input_uvd.png"})
        self.inputs = {"amp":0,"freq":4}
        self.outputs = {"signal":0}
        self.connpoints = []
        for ind in range(len(self.outputs)):
            self.connpoints.append(ConnectionPoint(ph,self,list(self.outputs)[ind],"out",pos,glm.vec3([-0.55,-0.1*ind,0])))
        for ind in range(len(self.inputs)):
            self.connpoints.append(ConnectionPoint(ph,self,list(self.inputs)[ind],"in",pos,glm.vec3([0.55,-0.1*ind,0])))  

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


    def update(self,deltaTime,audioHandler):
        for cp in self.connpoints:
            cp.updatePos(self.model.pos)
        


class SpeakerOut(NodeElement):
    def __init__(self,ph,name, pos):
        """Inputs: freq,amp. Output: wave signal"""
        
        NodeElement.__init__(self,ph,{"name":name, "pos":pos,"rot":[1.57,0,0],"scale":0.3,"file":"res/input.obj","texture":"res/speaker.png"})
        self.inputs = {"signal":0}
        self.connpoints = []
        self.time = 0
        self.outputSample = np.array([],dtype=np.int16)
        for ind in range(len(self.inputs)):
            self.connpoints.append(ConnectionPoint(ph,self,list(self.inputs)[ind],"in",pos,glm.vec3([0.55,-0.1*ind,0])))
        
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


    def update(self,deltaTime,audioHandler):
        self.time += deltaTime
        for cp in self.connpoints:
            cp.updatePos(self.model.pos)
        self.outputSample=np.append(self.outputSample, np.sin(self.time))
        if len(self.outputSample)>=500:
            print(self.outputSample)
            audioHandler.dataPlaying = self.outputSample
            audioHandler.dataReady = False
            self.outputSample = np.array([],dtype=np.int16)


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
        for key in []:
            options = ""
            val = key.split(",")
            if len(val)>1 and val[0] == "3fv":
                shader.SetUniform3fv(val[1],options[key])
            elif len(val)>1 and val[0] == "4fv":
                shader.SetUniform4fv(val[1],options[key])
            else:
                shader.SetUniform1f(key,options[key])

        #mvp = np.transpose(np.matmul(viewMat,self.model.modelMat))     
        shader.SetUniform3f("u_from",self.fromPos[0],-self.fromPos[1],self.fromPos[2])
        shader.SetUniform3f("u_to",self.toPos[0],-self.toPos[1],self.toPos[2])   
        #shader.SetUniformMat4f("u_MVP", mvp)
        shader.SetUniformMat4f("u_VP", np.transpose(viewMat))
        shader.SetUniformMat4f("u_Model", np.transpose(self.model.modelMat))
        
        renderer.Draw(self.model.va,self.model.ib,shader)
        #
        #self.model.DrawWithShader(shaderhandler.getShader("bezier"),renderer,viewMat)
    def update(self,deltaTime,audioHandler):
        if self.fromConnPoint != None:
            self.fromPos = glm.vec3(self.fromConnPoint.model.pos)*50
        if self.toConnPoint != None:
            self.toPos = glm.vec3(self.toConnPoint.model.pos)*50
    


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
    def update(self,deltaTime,audioHandler):
        """
        rotz = pyrr.matrix44.create_from_z_rotation(self.rot[2])
        rotx = pyrr.matrix44.create_from_x_rotation(self.rot[0])
        rot = np.matmul(np.matmul(pyrr.matrix44.create_from_y_rotation(self.rot[1]),rotz),rotx)
        self.camModel = np.matmul(rot,np.transpose(pyrr.matrix44.create_from_translation(self.pos)))
        """
        self.camModel=glm.lookAt(self.pos, self.pos + glm.vec3(0,0,1) , glm.vec3(0,1,0))
        
