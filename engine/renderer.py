from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.GLUT import *
from OpenGL.GLU import *
from ctypes import c_void_p, pointer, sizeof, c_float, c_uint
import numpy as np
from PIL import Image

def GetSizeOfType(t):
    if t == GL_FLOAT:
        return 4
    if t == GL_UNSIGNED_INT:
        return 4
    return 1


class VertexBuffer:
    def __init__(self, data, draw=GL_STATIC_DRAW, size=0):
        self.RendererId = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.RendererId)
        if data is None:
            glBufferData(GL_ARRAY_BUFFER, size, None, draw)
        else:
            glBufferData(GL_ARRAY_BUFFER, data, draw)
    def Bind(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.RendererId)
    def UnBind(self):
        glBindBuffer(GL_ARRAY_BUFFER, 0)


class IndexBuffer:
    def __init__(self, data, count):
        self.RendererId = glGenBuffers(1)
        self.Count = count
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.RendererId)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, data, GL_STATIC_DRAW)
    def Bind(self):
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.RendererId)
    def UnBind(self):
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

class VertexArray:
    def __init__(self):
        self.RendererId = glGenVertexArrays(1)
    def AddBuffer(self,vb, layout):
        self.Bind()
        vb.Bind()
        elements = layout.Elements
        offset = 0
        for i in range(len(elements)):
            element = elements[i]
            glEnableVertexAttribArray(i)
            glVertexAttribPointer(i, element[1], element[0], element[2], layout.Stride, c_void_p(offset))
            offset += element[1]*GetSizeOfType(element[0])
            
    def Bind(self):
        glBindVertexArray(self.RendererId)
    def UnBind(self):
        glBindVertexArray(0)
    
class VertexBufferLayout:
    def __init__(self):
        self.Elements = []
        self.Stride = 0
    def PushF(self, count):
        # type, size, normalized
        self.Elements.append((GL_FLOAT, count, GL_FALSE))
        self.Stride += count * sizeof(c_float)
    def PushI(self, count):
        # type, size, normalized
        self.Elements.append((GL_UNSIGNED_INT, count, GL_FALSE))
        self.Stride += count * GetSizeOfType(GL_UNSIGNED_INT)
            
class Shader:
    def __init__(self, vertexsrc, fragmentsrc):
        self.VertexSrc = vertexsrc
        self.FragmentSrc = fragmentsrc
        self.VertexShader = shaders.compileShader(self.VertexSrc,GL_VERTEX_SHADER)
        self.FragmentShader = shaders.compileShader(self.FragmentSrc,GL_FRAGMENT_SHADER)
        self.RendererId = shaders.compileProgram(self.VertexShader,self.FragmentShader)
        self.LocationCache = {}
        
    def SetUniform4f(self,name,v0,v1,v2,v3):
        glUniform4f(self.GetUniformLocation(name),v0,v1,v2,v3)
        
    def SetUniform3f(self,name,v0,v1,v2):
        glUniform3f(self.GetUniformLocation(name),v0,v1,v2)

    def SetUniform1f(self,name,v0):
        glUniform1f(self.GetUniformLocation(name),v0)

    def SetUniform1i(self,name,v0):
        glUniform1i(self.GetUniformLocation(name),v0)

    def SetUniform1iv(self,name,arr):
        glUniform1iv(self.GetUniformLocation(name),len(arr),arr)

    def SetUniformMat4f(self,name, mat):
        glUniformMatrix4fv(self.GetUniformLocation(name), 1, GL_FALSE, mat)
    
    def SetUniform3fv(self,name,arr):
        glUniform3fv(self.GetUniformLocation(name),len(arr),arr)
    
    def SetUniform4fv(self,name,arr):
        glUniform4fv(self.GetUniformLocation(name),len(arr),arr)


    def Bind(self):
        shaders.glUseProgram(self.RendererId)

    def UnBind(self):
        shaders.glUseProgram(0)

    def GetUniformLocation(self,name):
        if name in self.LocationCache:
            return self.LocationCache[name]
        else:
            location = glGetUniformLocation(self.RendererId, name)
            if location == -1:
                print("Warning: uniform location "+name+" doesn't exist")
                if name == "lightPos":
                    raise ArgumentError
            self.LocationCache[name] = location
            return location    

class Texture:
    def __init__(self,filename,repeat=False):
        self.FilePath = filename
        self.BPP = 0
        self.LocalBuffer = ""

        self.im = Image.open(self.FilePath)
        
        self.Width, self.Height = self.im.size

        #out = im.transpose(Image.FLIP_TOP_BOTTOM)
        
        self.RendererId = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D,self.RendererId)
        self.LocalBuffer = self.im.tobytes("raw", "RGBA", 0, -1)
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,GL_LINEAR)
        if repeat:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        else:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.Width, self.Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, self.LocalBuffer)
        glBindTexture(GL_TEXTURE_2D, 0)
    def OverrideTexture(self,image,repeat=False):
        self.im = image
        
        self.Width, self.Height = self.im.size

        #out = im.transpose(Image.FLIP_TOP_BOTTOM)
        
        self.RendererId = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D,self.RendererId)
        self.LocalBuffer = self.im.tobytes("raw", "RGBA", 0, -1)
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,GL_LINEAR)
        if repeat:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        else:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.Width, self.Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, self.LocalBuffer)
        glBindTexture(GL_TEXTURE_2D, 0)
    def Bind(self,slot=0):
        glActiveTexture(GL_TEXTURE0+slot)
        glBindTexture(GL_TEXTURE_2D, self.RendererId)
    def UnBind(self):
        glBindTexture(GL_TEXTURE_2D, 0)


class Renderer:
    def __init__(self):
        print(glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS),"texture slots are available.")
        pass
    def Clear(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    def Draw(self, vertarr, indbuf, shader, drawUntil=None):
        shader.Bind()
        vertarr.Bind()
        indbuf.Bind()
        if drawUntil == None:
            drawUntil = indbuf.Count
        glDrawElements(GL_TRIANGLES, drawUntil, GL_UNSIGNED_INT, None)
    def DrawInstanced(self, vertarr, indbuf, shader, count, offsets):
        shader.Bind()
        vertarr.Bind()
        indbuf.Bind()
        drawUntil = indbuf.Count
        instanceVBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO)
        glBufferData(GL_ARRAY_BUFFER, offsets, GL_STATIC_DRAW)
        glEnableVertexAttribArray(3)
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(c_float), c_void_p(0))
        #glVertexAttribPointer(i, element[1], element[0], element[2], layout.Stride, c_void_p(offset))
        #glBindBuffer(GL_ARRAY_BUFFER, 0)
        glVertexAttribDivisor(3, 1)
        #glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDrawElementsInstanced(GL_TRIANGLES, drawUntil, GL_UNSIGNED_INT, None, count)


# draw->buffer
# flush
#


class FPSCounter:
    def __init__(self):
        self.fpsCounter = []
        self.deltaTime = 0
        self.FPS = 0
        self.currentTime = 0
        self.averageFrameTime = 0
        self.pastFrameTimes = []
    def drawFrame(self, now):
        for x in range(len(self.fpsCounter)-1,0,-1):
            if self.fpsCounter[x]<now-1:
                self.fpsCounter.remove(self.fpsCounter[x])
        if len(self.fpsCounter)>0:
            self.deltaTime = now-self.fpsCounter[-1]
            if len(self.pastFrameTimes) > 100:
                self.pastFrameTimes.pop(0)
            self.pastFrameTimes.append(self.deltaTime)
            self.averageFrameTime = sum(self.pastFrameTimes)/len(self.pastFrameTimes)
        self.fpsCounter.append(now)
        self.FPS = len(self.fpsCounter)
        self.currentTime = now

class ShaderHandler:
    def __init__(self):
        self.shaders = {}
    def loadShader(self,name,vfile,ffile):
        with open(vfile,"r") as vert:
            vshader = "".join(vert.readlines())
        with open(ffile,"r") as frag:
            fshader = "".join(frag.readlines())
        self.shaders[name] = Shader(vshader, fshader)
    def getShader(self,name):
        if name in self.shaders:
            return self.shaders[name]
        print('The shader: '+name+' is not loaded.')
        return None