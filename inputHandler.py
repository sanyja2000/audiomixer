from audioop import mul
from OpenGL.GLUT import *
import sys
from cv2 import multiply
import glm

class InputHandler:
    def __init__(self):
        """
        self.keysdown[key]
        - if key is not pressed 0 or KeyError-> care!
        - if key is just pressed 1
        - if key is held down 2
        """
        self.mouseX = 0
        self.mouseY = 0
        self.mouseLeftDown = False
        self.mouseRightDown = False
        self.mouseXNorm = 0
        self.mouseYNorm = 0        
        self.mouseLocked = False
        self.mouseXoffset = 3.14*1.5
        self.windowSize = [0,0]
        self.mouseCatched = True
        self.keysDown = {b'a':0,b's':0,b'd':0,b'w':0,b' ':0}
        self.interactingWith = None
    def passiveMouseEventHandler(self,*args):
        """
        if self.mouseCatched and not self.mouseLocked:
            self.mouseX += args[0]-int(self.windowSize[0]/2)
            self.mouseY += args[1]-int(self.windowSize[1]/2)
            glutWarpPointer(int(self.windowSize[0]/2), int(self.windowSize[1]/2))
        """
        self.mouseX = args[0]
        self.mouseY = args[1]
        self.mouseXNorm = self.mouseX*2/self.windowSize[0]-1
        self.mouseYNorm = -self.mouseY*2/self.windowSize[1]+1
    def screenToWorld(self,proj,camera,x,y):
        output = glm.unProject(glm.vec3(x,y,1.0), camera.camModel, proj, glm.vec4(0,0,self.windowSize[0],self.windowSize[1]))
        output.y*=-1
        multiplier = camera.zoomLevel/-2
        return output/5*multiplier+glm.vec3(camera.pos.x*(0.8+(multiplier-1)*(-0.2)),camera.pos.y*(1.2+(multiplier-1)*(0.2)),0.0)
    def activeMouseEventHandler(self,*args):
        pass
    def keyDownHandler(self, *args):
        self.keysDown[args[0]] = 1
    def keyUpHandler(self, *args):
        self.keysDown[args[0]] = 0
    def isKeyDown(self,key):
        if key in self.keysDown and self.keysDown[key] == 1:
            return True
        return False
    def isKeyHeldDown(self,key):
        if key in self.keysDown and self.keysDown[key] == 2:
            return True
        return False
    def updateKeysDown(self):
        for key in self.keysDown:
            if self.keysDown[key] == 1:
                self.keysDown[key] = 2
    def hideCursor(self):
        pass
    def changeWindowSize(self,size):
        self.windowSize = size
