import numpy as np
from .objloader import processObjFile
from .renderer import *
import pyrr

DRAWCALLCOUNT = 0

class Object3D:
    def __init__(self,filename,texture="res/Gun.png",textureRepeat=False):
        self.points, self.indices = np.array([]),np.array([])
        self.va = None
        self.ib = None
        self.texture = None
        if filename != None:
            self.filePath = filename
            self.points, self.indices = processObjFile(self.filePath)
            self.va = VertexArray()
            self.vb = VertexBuffer(self.points)
            self.layout = VertexBufferLayout()
            self.layout.PushF(3)
            self.layout.PushF(2)
            self.layout.PushF(3)
            self.va.AddBuffer(self.vb, self.layout)
            self.ib = IndexBuffer(self.indices, len(self.indices))
            self.texture = Texture(texture,repeat=textureRepeat)
        
        self.pos = np.array([0,0,0])
        self.rot = np.array([0,0,0])
        self.scale = 1
        self.GenerateModelMatrix()
    def DrawWithShader(self,shader,renderer,viewMat,options={}):
        shader.Bind()
        self.texture.Bind()
        shader.SetUniform1i("u_Texture",0)
        for key in options:
            val = key.split(",")
            if len(val)>1 and val[0] == "3fv":
                shader.SetUniform3fv(val[1],options[key])
            elif len(val)>1 and val[0] == "4fv":
                shader.SetUniform4fv(val[1],options[key])
            else:
                shader.SetUniform1f(key,options[key])
        mvp = np.transpose(np.matmul(viewMat,self.modelMat))        
        shader.SetUniformMat4f("u_MVP", mvp)
        renderer.Draw(self.va,self.ib,shader)
        global DRAWCALLCOUNT
        DRAWCALLCOUNT += 1
    def GenerateModelMatrix(self):
        rxyz = np.matmul(np.matmul(pyrr.matrix44.create_from_x_rotation(self.rot[0]),pyrr.matrix44.create_from_y_rotation(self.rot[1])),pyrr.matrix44.create_from_z_rotation(self.rot[2]))
        scl = pyrr.matrix44.create_from_scale(np.array([self.scale,self.scale,self.scale]))
        self.modelMat = np.transpose(np.matmul(np.matmul(scl,rxyz),pyrr.matrix44.create_from_translation(np.array(self.pos))))
    def SetPosition(self,vec):
        if np.min(self.pos == vec):
            return
        self.pos = vec
        self.GenerateModelMatrix()
    def SetRotation(self,vec):
        if np.min(self.rot == vec):
            return
        self.rot = vec
        self.GenerateModelMatrix()
    def SetScale(self,scl):
        if self.scale == scl:
            return
        self.scale=scl
        self.GenerateModelMatrix()
    def Clone(self):
        o = Object3D(None)
        o.va = self.va
        o.ib = self.ib
        o.points = self.points
        o.texture = self.texture
        return o
    def DistanceTo(self,vec):
        d = (vec[0]-self.pos[0])**2+(vec[1]-self.pos[1])**2+(vec[2]-self.pos[2])**2
        return d**0.5

class prefabHandler:
    def __init__(self):
        self.loadedFiles = []
        self.loadedObjects = []
        self.emptyObj = Object3D(None,texture="res/1px.png")
    def loadFile(self,_filename,_texture,textureRepeat=False,unique=False):
        if _filename in self.loadedFiles and not unique:
            index = self.loadedFiles.index(_filename)
            o = self.loadedObjects[index].Clone()
            if(hasattr(self.loadedObjects[index],"texture") and _texture != self.loadedObjects[index].texture.FilePath):
                o.texture = Texture(_texture)
            return o
        o = Object3D(_filename,texture=_texture,textureRepeat=textureRepeat)
        self.loadedFiles.append(_filename)
        self.loadedObjects.append(o)
        return o
    def empty(self):
        return self.emptyObj.Clone()