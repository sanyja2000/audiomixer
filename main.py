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

        #glClearColor(0.52,0.80,0.92,1.0)
        glClearColor(0.039,0.5,0.4,1.0)

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

        self.drawingCurve = False


        self.table = Table("maps/empty.json")

        #self.tmpwg = SineGenerator(self.table.prefabHandler, "32",[0.5,0,0])
        self.tmpwg = SquareGenerator(self.table.prefabHandler, "32",[1,0.7,0])
        
        self.table.objects.append(self.tmpwg)

        self.table.objects.append(AddNode(self.table.prefabHandler, "2314",[-1,-0.7,0]))


        self.table.objects.append(FilePlayer(self.table.prefabHandler, "audiotest/Cartoon_On&On.wav",[2,-0.7,0]))
        self.table.objects.append(FilePlayer(self.table.prefabHandler, "audiotest/LostSky_Fearless.wav",[1,-0.7,0]))

        self.table.objects.append(ConstantNode(self.table.prefabHandler, "33322",[-1,1.4,0]))

        self.speakerOut = SpeakerOut(self.table.prefabHandler, "3342",[-1,0.7,0])


        self.audioHandler.speakerStart()
        #self.audioHandler.playSound("res/audio/feel_cut.wav")
        self.mouseCurve = BezierCurve(self.table.prefabHandler, None, None)

        self.camera = Camera()

        self.grabbedNode = None

        self.sortedNodes = []

        self.rearrangeNodes()

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
    def addNodeRecursive(self, node):
        self.sortedNodes.append(node)
        for cp in node.connpoints:
            if cp.side == "in":
                if cp.bezier is not None:
                    self.addNodeRecursive(cp.bezier.fromConnPoint.parent)
    def rearrangeNodes(self):
        self.sortedNodes = []
        if self.speakerOut.connpoints[0].bezier is not None:
            curnode = self.speakerOut.connpoints[0].bezier.fromConnPoint.parent
            self.addNodeRecursive(curnode)
        self.sortedNodes = self.sortedNodes[::-1]
        """
        for i in self.table.objects:
            if i not in self.sortedNodes:
                self.sortedNodes.append(i)
        """
    def mouseClicked(self,*args):
        output = self.inputHandler.screenToWorld(self.proj,self.camera,self.inputHandler.mouseX,self.inputHandler.mouseY)
        

        if args[1] == 1:
            if args[0] == 0:
                # Left click up
                self.inputHandler.mouseLeftDown = False
                self.grabbedNode = None
                if self.drawingCurve:
                    for i in self.table.objects+[self.speakerOut]:
                        if hasattr(i, "checkIntersect"):
                            intersectObj = i.checkIntersect(output.x,output.y)
                            if intersectObj != None:
                                if(type(intersectObj)==ConnectionPoint):
                                    if intersectObj.side != self.mouseCurve.fromConnPoint.side:
                                        if intersectObj.parent == self.mouseCurve.fromConnPoint.parent:
                                            break
                                        alreadyConnected = []
                                        for bc in self.table.objects+[self.speakerOut]:
                                            if type(bc)==BezierCurve:
                                                if intersectObj == bc.fromConnPoint or intersectObj == bc.toConnPoint or self.mouseCurve.fromConnPoint == bc.fromConnPoint or self.mouseCurve.fromConnPoint == bc.toConnPoint:
                                                    alreadyConnected.append(bc)
                                        for bc in alreadyConnected:
                                            bc.fromConnPoint.bezier = None
                                            bc.toConnPoint.bezier = None
                                            self.table.objects.remove(bc)
                                        fromPoint = self.mouseCurve.fromConnPoint
                                        toPoint = intersectObj
                                        if intersectObj.side == "out":
                                            fromPoint = intersectObj
                                            toPoint = self.mouseCurve.fromConnPoint
                                        newbc = BezierCurve(self.table.prefabHandler, fromPoint,toPoint)
                                        self.table.objects.append(newbc)
                                        intersectObj.bezier = newbc
                                        self.mouseCurve.fromConnPoint.bezier = newbc
                                        self.rearrangeNodes()
                                        break
                self.drawingCurve = False             
                                        
            if args[0] == 2:
                # Right click up
                self.inputHandler.mouseRightDown = False
        elif args[1] == 0:
            if args[0] == 0:
                # Left click down
                self.inputHandler.mouseLeftDown = True
                if self.drawingCurve:
                    self.drawingCurve = False
                    return

                for i in self.table.objects:
                    if hasattr(i, "checkIntersect"):
                        intersectObj = i.checkIntersect(output.x,output.y)
                        if intersectObj != None:
                            if(type(intersectObj)==ConnectionPoint):
                                self.drawingCurve = True
                                self.mouseCurve.fromConnPoint = intersectObj
                            else:
                                self.grabbedNode = intersectObj
                            break
            if args[0] == 2:
                # Right click down
                self.inputHandler.mouseRightDown = True
                self.camera.savedPos = glm.vec3(self.inputHandler.mouseX,self.inputHandler.mouseY,0)/200-self.camera.pos
                pass


    def showScreen(self):
        

        now = time.perf_counter()
        glutSetWindowTitle("AudioMixer - FPS: "+str(self.FPSCounter.FPS))
    
        self.audioHandler.update()

        self.renderer.Clear()


        if self.inputHandler.mouseRightDown:
            output = self.inputHandler.screenToWorld(self.proj,self.camera,self.inputHandler.mouseX,self.inputHandler.mouseY)
            self.camera.pos = glm.vec3(self.inputHandler.mouseX,self.inputHandler.mouseY,0)/200 - self.camera.savedPos
            self.camera.update(None,None)
        
        
        # camera
        viewMat = np.matmul(self.proj,self.camera.camModel)


        popupText = ""
        

        for i in self.table.objects:
            i.draw(self.shaderHandler,self.renderer,viewMat)
            if hasattr(i, "update"):
                i.update(self.FPSCounter,self.audioHandler)


        for i in self.sortedNodes:
            if hasattr(i, "audioUpdate") and not self.audioHandler.dataReady:
                i.audioUpdate(self.FPSCounter,self.audioHandler)
            
        self.speakerOut.draw(self.shaderHandler,self.renderer,viewMat)
        self.speakerOut.update(self.FPSCounter,self.audioHandler)
        #self.tmpwg.draw(self.shaderHandler,self.renderer,viewMat)
        #self.tmpwg.update(self.FPSCounter.deltaTime,self.audioHandler)

        output = self.inputHandler.screenToWorld(self.proj,self.camera,self.inputHandler.mouseX,self.inputHandler.mouseY)

        if self.grabbedNode != None:
            self.grabbedNode.model.SetPosition(glm.vec3(output.x,output.y,0))

        if self.drawingCurve:
            self.mouseCurve.toPos = glm.vec3(output.x, output.y, 0)*50
            self.mouseCurve.update(self.FPSCounter,self.audioHandler)
            self.mouseCurve.draw(self.shaderHandler,self.renderer,viewMat)
        
        self.fontHandler.drawText("",-1*len(popupText)/50,-0.6,0.05,self.renderer)

        # Draw in game menu
        glutSwapBuffers()
        
        self.inputHandler.updateKeysDown()
        
        self.FPSCounter.drawFrame(now)

g = Game()

