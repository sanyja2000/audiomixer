from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from ctypes import c_void_p, pointer, sizeof, c_float
import numpy as np
import sys, math
import time
from engine.renderer import VertexBuffer, IndexBuffer, VertexArray, VertexBufferLayout, Shader, Renderer, Texture, FPSCounter, ShaderHandler
import classHandler
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
import engine.objectHandler

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
            self.shaderHandler.loadShader("instanced","shaders/3.3/instanced.vert","shaders/3.3/fragment_new.shader")
            self.shaderHandler.loadShader("default_transparent","shaders/3.3/vertex_new.shader","shaders/3.3/fragment_def_transparent.shader")
            self.shaderHandler.loadShader("font","shaders/3.3/vertex_font3d.shader","shaders/3.3/fragment_font.shader")
            self.shaderHandler.loadShader("bezier","shaders/3.3/bezier.vert","shaders/3.3/bezier.frag")
            self.shaderHandler.loadShader("propertyMenu","shaders/3.3/propertyMenu.vert","shaders/3.3/propertyMenu.frag")
            self.shaderHandler.loadShader("addMenu","shaders/3.3/addMenu.vert","shaders/3.3/addMenu.frag")
            self.shaderHandler.loadShader("background","shaders/3.3/background.vert","shaders/3.3/background.frag")
        else:
            self.shaderHandler.loadShader("default","shaders/2.1/vertex_new.shader","shaders/2.1/fragment_new.shader")
            self.shaderHandler.loadShader("default_transparent","shaders/2.1/vertex_new.shader","shaders/2.1/fragment_def_transparent.shader")
            self.shaderHandler.loadShader("font","shaders/2.1/vertex_font3d.shader","shaders/2.1/fragment_font.shader")
            self.shaderHandler.loadShader("bezier","shaders/2.1/bezier.vert","shaders/2.1/bezier.frag")
            self.shaderHandler.loadShader("propertyMenu","shaders/2.1/propertyMenu.vert","shaders/2.1/propertyMenu.frag")
            self.shaderHandler.loadShader("background","shaders/2.1/background.vert","shaders/2.1/background.frag")
          

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

        # Setup base Table class
        self.table = Table("maps/empty.json")
        
        
        #self.table.objects.append(FilePlayer(self.table.prefabHandler, "audiotest/Cartoon_On&On.wav",[1,0,0]))
        #self.table.objects.append(FilePlayer(self.table.prefabHandler, "audiotest/AllIWant.wav",[1,-0.7,0])) #"audiotest/LostSky_Fearless.wav"

        #self.table.objects.append(LinearAnim(self.table.prefabHandler, "33322",[-1,1.4,0]))

        self.keyboard = None
        self.keyboard = Keyboard(self.table.prefabHandler, "kbm",[-1,1.4,0])
        self.table.objects.append(self.keyboard)

        
        self.speakerOut = SpeakerOut(self.table.prefabHandler, "3342",[-1,0.7,0])


        self.audioHandler.speakerStart()
        self.mouseCurve = BezierCurve(self.table.prefabHandler, None, None)

        self.camera = Camera()
        self.background = BackgroundPlane()

        self.pm = PropertyMenu()
        self.addMenu = AddMenu()

        self.activeNode = None
        self.grabbedNode = None

        self.sortedNodes = []

        self.rearrangeNodes()

        self.unDeletableClasses = ["Keyboard","SpeakerOut"]

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
        """
        Recursive function for adding the connected nodes to the sortedNodes list.
        """
        if node in self.sortedNodes:
            return
        self.sortedNodes.append(node)
        for cp in node.connpoints:
            if cp.side == "in":
                if cp.bezier is not None:
                    self.addNodeRecursive(cp.bezier.fromConnPoint.parent)
    def rearrangeNodes(self):
        """
        Creates a list of nodes from the speaker using the connected bezier curves.
        Then it reverses the list creating the required order of nodes for the audio processing.
        """
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
                                        # Check if there are already connected curves to the same connectionpoints
                                        alreadyConnected = []
                                        for bc in self.table.objects+[self.speakerOut]:
                                            if type(bc)==BezierCurve:
                                                if intersectObj == bc.fromConnPoint or intersectObj == bc.toConnPoint or self.mouseCurve.fromConnPoint == bc.fromConnPoint or self.mouseCurve.fromConnPoint == bc.toConnPoint:
                                                    alreadyConnected.append(bc)
                                        for bc in alreadyConnected:
                                            # Remove previously connected curves
                                            bc.fromConnPoint.bezier = None
                                            bc.toConnPoint.bezier = None
                                            self.table.objects.remove(bc)
                                        fromPoint = self.mouseCurve.fromConnPoint
                                        toPoint = intersectObj
                                        if intersectObj.side == "out":
                                            # If its connected from input -> output, then it reverses the connection
                                            fromPoint = intersectObj
                                            toPoint = self.mouseCurve.fromConnPoint
                                        # Adds curve to the table objects
                                        newbc = BezierCurve(self.table.prefabHandler, fromPoint,toPoint)
                                        self.table.objects.append(newbc)
                                        intersectObj.bezier = newbc
                                        self.mouseCurve.fromConnPoint.bezier = newbc
                                        # Recreate the update order graph
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

                if self.inputHandler.mouseXNorm<-0.9:
                    # Clicked on add menu 
                    newNode = self.addMenu.checkAddClick(self.inputHandler.mouseXNorm, self.inputHandler.mouseYNorm)
                    if newNode is not None:
                        if self.grabbedNode != None:
                            self.grabbedNode.isSelected = False
                        if self.activeNode != None:
                            self.activeNode.isSelected = False    
                        self.grabbedNode = newNode(self.table.prefabHandler, "",[-1,0.7,0])
                        self.table.objects.append(self.grabbedNode)
                        self.activeNode = self.grabbedNode
                        if self.grabbedNode is not None:
                            self.activeNode.isSelected = True
                    return

                if self.pm.isOpen:
                    if self.inputHandler.mouseXNorm>0.5:
                        # Clicked on property menu
                        self.pm.checkPropertyClick(self.inputHandler.mouseXNorm, self.inputHandler.mouseYNorm)
                        return
                for i in self.table.objects+[self.speakerOut]:
                    if hasattr(i, "checkIntersect"):
                        intersectObj = i.checkIntersect(output.x,output.y)
                        if intersectObj != None:
                            if(type(intersectObj)==ConnectionPoint):
                                if self.inputHandler.isKeyHeldDown(b'x'):
                                    # Delete curve if x pressed down
                                    if intersectObj.bezier is not None:
                                        bc = intersectObj.bezier
                                        self.table.objects.remove(intersectObj.bezier)
                                        bc.toConnPoint.data = None
                                        bc.toConnPoint.bezier = None
                                        bc.fromConnPoint.data = None
                                        bc.fromConnPoint.bezier = None
                                else:
                                    # Start drawing curve from clicked connectionpoint
                                    self.drawingCurve = True
                                    self.mouseCurve.fromConnPoint = intersectObj
                            else:
                                # Grab object
                                self.grabbedNode = intersectObj
                            break
                if self.activeNode != self.grabbedNode:
                    if self.activeNode is not None:
                        self.activeNode.isSelected = False
                    self.activeNode = self.grabbedNode
                    if self.grabbedNode is not None:
                        self.activeNode.isSelected = True
                    self.pm.selectedProperty = -1
            if args[0] == 2:
                # Right click down
                self.inputHandler.mouseRightDown = True
                self.camera.savedPos = glm.vec3(self.inputHandler.mouseX,self.inputHandler.mouseY,0)/200-self.camera.pos
            if args[0] == 3:
                # Scrollwheel up
                if self.inputHandler.mouseXNorm<-0.9:
                    self.addMenu.scroll(-1)
                else:
                    self.camera.changeZoom(1)
            if args[0] == 4:
                # Scrollwheel down
                if self.inputHandler.mouseXNorm<-0.9:
                    self.addMenu.scroll(1)
                else:
                    self.camera.changeZoom(-1)


    def showScreen(self):
        now = time.perf_counter()
        glutSetWindowTitle("AudioMixer - FPS: "+str(self.FPSCounter.FPS)+", "+str(round(self.FPSCounter.deltaTime*1000,2))+"ms, avg: "+str(round(self.FPSCounter.averageFrameTime*1000,2))+"ms. draw calls: "+str(engine.objectHandler.DRAWCALLCOUNT))

        engine.objectHandler.DRAWCALLCOUNT = 0
    
        self.audioHandler.update()

        self.renderer.Clear()
        self.background.draw(self.shaderHandler,self.renderer,self.camera,self.audioHandler)


        if self.inputHandler.mouseRightDown:
            output = self.inputHandler.screenToWorld(self.proj,self.camera,self.inputHandler.mouseX,self.inputHandler.mouseY)
            self.camera.pos = glm.vec3(self.inputHandler.mouseX,self.inputHandler.mouseY,0)/200 - self.camera.savedPos
            self.camera.update(None,None)
        
        
        # camera
        viewMat = np.matmul(self.proj,self.camera.camModel)


        popupText = str(self.FPSCounter.FPS)
        
        # Drawing and updating the objects
        for i in self.table.objects:
            i.draw(self.shaderHandler,self.renderer,viewMat)
            if hasattr(i, "drawText"):
                i.drawText(self.fontHandler,self.renderer,viewMat)
            if hasattr(i, "update"):
                i.update(self.FPSCounter,self.audioHandler)

        # Sorted update for audio processing
        for i in self.sortedNodes:
            if hasattr(i, "audioUpdate") and not self.audioHandler.dataReady:
                i.audioUpdate(self.FPSCounter,self.audioHandler)
            
        self.speakerOut.draw(self.shaderHandler,self.renderer,viewMat)
        self.speakerOut.update(self.FPSCounter,self.audioHandler)

        classHandler.CPH.DrawWithShader(self.shaderHandler,self.renderer,viewMat)

        #print(self.table.objects[0].model.pos)

        # Get mouseCoordinate in 3D space
        output = self.inputHandler.screenToWorld(self.proj,self.camera,self.inputHandler.mouseX,self.inputHandler.mouseY)

        if self.grabbedNode != None:
            self.grabbedNode.model.SetPosition(glm.vec3(output.x,output.y,0))

        if self.activeNode != None:
            if self.inputHandler.isKeyDown(b'\x7f') and not (type(self.activeNode).__name__ in self.unDeletableClasses):
                # delete key
                self.table.removeNodeObject(self.activeNode)
                self.grabbedNode = None
                self.activeNode = None

        if self.drawingCurve:
            self.mouseCurve.toPos = glm.vec3(output.x, output.y, 0)*50
            self.mouseCurve.update(self.FPSCounter,self.audioHandler)
            self.mouseCurve.draw(self.shaderHandler,self.renderer,viewMat)
        
        self.pm.isOpen = False
        if self.activeNode is not None:
            self.pm.isOpen = True
            self.pm.draw(self.shaderHandler, self.renderer, self.fontHandler)
            self.pm.update(self.FPSCounter,self.audioHandler,self.inputHandler,self.activeNode)
        
        self.addMenu.draw(self.shaderHandler, self.renderer, self.fontHandler)

        glutSwapBuffers()


        if(self.keyboard):
            self.keyboard.updateKeys(self.inputHandler)
        
        self.inputHandler.updateKeysDown()
        
        self.FPSCounter.drawFrame(now)

if __name__ == '__main__':
    g = Game()

