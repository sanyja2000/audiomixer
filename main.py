from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from ctypes import c_void_p, pointer, sizeof, c_float
import numpy as np
import sys, math
import time
from engine.renderer import VertexBuffer, IndexBuffer, VertexArray, VertexBufferLayout, Shader, Renderer, Texture, FPSCounter, ShaderHandler
from inputHandler import InputHandler
from engine.audioHandler import AudioHandler
from engine.objloader import processObjFile
from engine.objectHandler import Object3D
import pyrr
import random
from threading import Thread
from classHandler import *
from engine.fontHandler import FontHandler
from table import Table

def constrain(n,f,t):
    if n<f:
        return f
    if n>t:
        return t
    return n

def GLClearError():
    while glGetError() != GL_NO_ERROR:
        pass
        
def GLCheckError():
    while True:
        err = glGetError()
        if err == 0:
            break
        print("[OpenGL Error] ",err)

class Game:
    def __init__(self):
        glutInit()
        glutInitDisplayMode(GLUT_RGBA)

        
        OPENGL_VERSION = 3

        if OPENGL_VERSION == 3:
            glutInitContextVersion (3, 3)
        else:
            glutInitContextVersion (2, 1)
        glutInitContextProfile (GLUT_COMPATIBILITY_PROFILE)
        self.windowSize = [1280,720]
        glutInitWindowSize(self.windowSize[0], self.windowSize[1])
        glutInitWindowPosition(0, 0)
        self.window = glutCreateWindow("AudioMixer")
        glutReshapeFunc(self.windowResize)
        
        glutDisplayFunc(self.showScreen)
        glutIdleFunc(self.showScreen)
        self.inputHandler = InputHandler()
        self.inputHandler.changeWindowSize(self.windowSize)
        glutKeyboardFunc(self.inputHandler.keyDownHandler)
        glutKeyboardUpFunc(self.inputHandler.keyUpHandler)

        glutMouseFunc(self.mouseClicked)

        glutPassiveMotionFunc(self.inputHandler.passiveMouseEventHandler)
        glutMotionFunc(self.inputHandler.passiveMouseEventHandler)

        
        GLClearError()

        print(glGetString(GL_SHADING_LANGUAGE_VERSION))

        glClearColor(0.52,0.80,0.92,1.0)

        self.shaderHandler = ShaderHandler()
        
        if OPENGL_VERSION == 3:
            self.shaderHandler.loadShader("default","shaders/3.3/vertex_new.shader","shaders/3.3/fragment_new.shader")
            self.shaderHandler.loadShader("default_transparent","shaders/3.3/vertex_new.shader","shaders/3.3/fragment_def_transparent.shader")
            self.shaderHandler.loadShader("font","shaders/3.3/vertex_font.shader","shaders/3.3/fragment_font.shader")
            self.shaderHandler.loadShader("bezier","shaders/3.3/bezier.vert","shaders/3.3/bezier.frag")
        else:
            # TODO: Add pauseMenu shaders
            self.shaderHandler.loadShader("default","shaders/2.1/vertex_new.shader","shaders/2.1/fragment_new.shader")
            self.shaderHandler.loadShader("default_transparent","shaders/2.1/vertex_new.shader","shaders/2.1/fragment_def_transparent.shader")
            self.shaderHandler.loadShader("font","shaders/2.1/vertex_font.shader","shaders/2.1/fragment_font.shader")
            

        self.fontHandler = FontHandler(self.shaderHandler.getShader("font"))

        self.audioHandler = AudioHandler()
        

        #self.proj = pyrr.matrix44.create_perspective_projection(45.0, self.windowSize[0]/self.windowSize[1], 1.0, 10.0)
        self.proj = glm.perspective(45.0, self.windowSize[0]/self.windowSize[1], 1.0, 10.0)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glEnable(GL_DEPTH_TEST)


        self.renderer = Renderer()


        self.FPSCounter = FPSCounter()



        self.table = Table("maps/empty.json")

        self.tmpwg = WaveGenerator(self.table.prefabHandler, {"name":"32","pos":[0.5,0,0],"rot":[1.57,0,0],"scale":0.3,"file":"res/input.obj","texture":"res/input_uvd.png"})


        self.tmpcurve = BezierCurve(self.table.prefabHandler, {})

        self.camera = Camera()

        glutMainLoop()
    def errorMsg(self, *args):
        print(args)
        return 0
    def windowResize(self, *args):
        self.windowSize = [args[0], args[1]]
        #self.proj = pyrr.matrix44.create_perspective_projection(45.0, self.windowSize[0]/self.windowSize[1], 1.0, 10.0)
        self.proj = glm.perspective(45.0, self.windowSize[0]/self.windowSize[1], 1.0, 10.0)
        glViewport(0,0,self.windowSize[0],self.windowSize[1])
        self.inputHandler.changeWindowSize(self.windowSize)
    def mouseClicked(self,*args):
        if args[1] == 1:
            return
        output = self.inputHandler.screenToWorld(self.proj,self.camera,self.inputHandler.mouseX,self.inputHandler.mouseY)
        
        #self.table.objects.append(Decoration(self.table.prefabHandler, {"name":"o"+str(len(self.table.objects)),"pos":[output[0],-output[1],0],"rot":[1.57,0,0],"scale":0.3,"file":"res/input.obj","texture":"res/input_uvd.png"}))
        


    def showScreen(self):
        

        now = time.perf_counter()
        glutSetWindowTitle("AudioMixer - FPS: "+str(self.FPSCounter.FPS))
    
        self.audioHandler.update()




        self.renderer.Clear()

        if self.inputHandler.isKeyHeldDown(b'w'):
            self.camera.pos += glm.vec3(0,1.0,0)*self.FPSCounter.deltaTime
        elif self.inputHandler.isKeyHeldDown(b's'):
            self.camera.pos += glm.vec3(0,-1.0,0)*self.FPSCounter.deltaTime
        if self.inputHandler.isKeyHeldDown(b'a'):
            self.camera.pos += glm.vec3(1,0.0,0)*self.FPSCounter.deltaTime
        elif self.inputHandler.isKeyHeldDown(b'd'):
            self.camera.pos += glm.vec3(-1,0.0,0)*self.FPSCounter.deltaTime
        self.camera.update(0,0)

        # camera
        viewMat = np.matmul(self.proj,self.camera.camModel)


        popupText = ""
        

        for i in []:#self.table.objects:
            i.draw(self.shaderHandler,self.renderer,viewMat)
            if hasattr(i, "update"):
                i.update(self.FPSCounter.deltaTime,self.audioHandler)

        self.tmpwg.draw(self.shaderHandler,self.renderer,viewMat)
        self.tmpwg.update(self.FPSCounter.deltaTime,self.audioHandler)

        output = self.inputHandler.screenToWorld(self.proj,self.camera,self.inputHandler.mouseX,self.inputHandler.mouseY)
        self.tmpcurve.toPos = glm.vec3(output.x, output.y, 0)*51
        
        self.tmpcurve.draw(self.shaderHandler,self.renderer,viewMat)
        
        self.fontHandler.drawText(popupText,-1*len(popupText)/50,-0.6,0.05,self.renderer)

        # Draw in game menu
        glutSwapBuffers()
        
        self.inputHandler.updateKeysDown()
        
        self.FPSCounter.drawFrame(now)

g = Game()

