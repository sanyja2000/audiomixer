from OpenGL.GL import *
from PIL import Image, ImageFont, ImageDraw
from engine.renderer import Shader,VertexArray, VertexBuffer, IndexBuffer, VertexBufferLayout
import numpy as np
from ctypes import sizeof, c_float
import glm

# tmp import
import time

class FontHandler:
    def __init__(self,shader):
        """
        Usage: initialise font handler, with a Shader from the renderer file;
        Preferably from a ShaderHandler.
        """
        self.shader = shader
        self.letterDict = {" ":0,"!":1,'"':2,"#":3,"$":4,"%":5,"&":6,"'":7,"(":8,")":9,"*":10,"+":11,",":12,"-":13,".":14,"/":15,
                           "0":16,"1":17,"2":18,"3":19,"4":20,"5":21,"6":22,"7":23,"8":24,"9":25,":":26,";":27,"<":28,"=":29,">":30,"?":31,
                           "@":32,"A":33,"B":34,"C":35,"D":36,"E":37,"F":38,"G":39,"H":40,"I":41,"J":42,"K":43,"L":44,"M":45,"N":46,"O":47,
                           "P":48,"Q":49,"R":50,"S":51,"T":52,"U":53,"V":54,"W":55,"X":56,"Y":57,"Z":58,"[":59,"\\":60,"]":61,"^":62,"_":63,
                           "`":64,"a":65,"b":66,"c":67,"d":68,"e":69,"f":70,"g":71,"h":72,"i":73,"j":74,"k":75,"l":76,"m":77,"n":78,"o":79,
                           "p":80,"q":81,"r":82,"s":83,"t":84,"u":85,"v":86,"w":87,"x":88,"y":89,"z":90,"{":91,"|":92,"}":93,"~":94," ":95,
                           " ":96," ":97," ":98," ":99," ":100," ":101," ":102," ":103," ":104," ":105," ":106," ":107," ":108," ":109," ":110," ":111,
                           "á":112,"Á":113,"é":114,"É":115,"í":116,"Í":117,"ó":118,"Ó":119,"ö":120,"Ö":121,"ő":122,"Ő":123,"ü":124,"Ü":125,"ű":126,"Ű":127}
        

        #vec2 pos, vec2 uv, float texIndex

        self.max_letter_count = 20
        
        self.indexList = []
        for i in range(self.max_letter_count):
            self.indexList += [0+i*4,1+i*4,2+i*4,2+i*4,3+i*4,0+i*4]
        #self.indices = np.array([0,1,2, 2,3,0, 4,5,6, 6,7,4])
        self.indices = np.array(self.indexList)


        self.values = [[0,0,0,0],[1,0,1/16,0],[1,1,1/16,1/16],[0,1,0,1/16]]

        
        self.va = VertexArray()
        self.vb = VertexBuffer(None, draw=GL_DYNAMIC_DRAW, size=self.max_letter_count*4*sizeof(c_float)*5)
        self.layout = VertexBufferLayout()
        self.layout.PushF(2)
        self.layout.PushF(2)
        self.layout.PushF(1)
        self.va.AddBuffer(self.vb, self.layout)
        self.ib = IndexBuffer(self.indices, len(self.indices))

        self.width = 100
        self.height = 50

        self.textureAtlas = Image.open("res/fontTextureBau2048.png") #256x256px 16x16 chars 16chars/row
        data = self.textureAtlas.tobytes("raw", "RGBA", 0, -1)
        
        self.width, self.height = self.textureAtlas.size

        self.TextureAtlasId = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D,self.TextureAtlasId)
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE)

        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)

        glBindTexture(GL_TEXTURE_2D, 0)
        #self.font = ImageFont.truetype("res/Montserrat-Regular.ttf", 70)
        
    def drawText(self,txt,x,y,scl,renderer):
        """
        Usage: txt as a python string;
        x,y from -1 to 1 left-to-right, bottom-to-top;
        scl a small value around 0.05;
        renderer is an instance from the Renderer class
        """
        
        xoffset = 0.5 #letter-spacing
        subtxt = txt
        longer = False
        if len(subtxt)>self.max_letter_count:
            longer = True
            subtxt = txt[:self.max_letter_count]
        self.shader.Bind()
        self.va.Bind()
        glBindTexture(GL_TEXTURE_2D,self.TextureAtlasId)

        
        self.points = np.zeros((len(subtxt)*self.max_letter_count), np.float32)
        
        for c in range(len(subtxt)):
            for d in range(4):
                self.points[c*self.max_letter_count+d*5] = self.values[d][0]+c*xoffset
                self.points[c*self.max_letter_count+1+d*5] = self.values[d][1]
                self.points[c*self.max_letter_count+2+d*5] = self.values[d][2]
                self.points[c*self.max_letter_count+3+d*5] = self.values[d][3]
                try:
                    self.points[c*self.max_letter_count+4+d*5] = self.letterDict[subtxt[c]]
                except KeyError as ke:
                    self.points[c*self.max_letter_count+4+d*5] = self.letterDict["_"]

        #print(self.points)     
        self.vb.Bind()
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(c_float)*len(self.points),self.points)
           
        self.shader.SetUniform4f("u_PosRotScl",x,y,0,scl)
        self.shader.SetUniform1i("u_Texture",0)
        self.shader.SetUniform1i("u_is3d",0)

        mvp = np.identity(4)        
        self.shader.SetUniformMat4f("u_MVP", mvp)
              
        renderer.Draw(self.va,self.ib,self.shader,drawUntil=len(subtxt)*6)
        glBindTexture(GL_TEXTURE_2D, 0)
        if longer:
            self.drawText(txt[self.max_letter_count:],x+scl*xoffset*self.max_letter_count,y,scl,renderer)
    
    def drawText3D(self,txt,modelMat,scl,viewMat,renderer):
        """
        Usage: txt as a python string;
        pos from -1 to 1 left-to-right, bottom-to-top;
        scl a small value around 0.05;
        renderer is an instance from the Renderer class
        """
        xoffset = 0.5 #letter-spacing
        subtxt = txt
        longer = False
        if len(subtxt)>self.max_letter_count:
            longer = True
            subtxt = txt[:self.max_letter_count]
        self.shader.Bind()
        self.va.Bind()
        glBindTexture(GL_TEXTURE_2D,self.TextureAtlasId)

        
        self.points = np.zeros((len(subtxt)*self.max_letter_count), np.float32)
        
        for c in range(len(subtxt)):
            for d in range(4):
                self.points[c*self.max_letter_count+d*5] = self.values[d][0]+c*xoffset
                self.points[c*self.max_letter_count+1+d*5] = self.values[d][1]
                self.points[c*self.max_letter_count+2+d*5] = self.values[d][2]
                self.points[c*self.max_letter_count+3+d*5] = self.values[d][3]
                try:
                    self.points[c*self.max_letter_count+4+d*5] = self.letterDict[subtxt[c]]
                except KeyError as ke:
                    self.points[c*self.max_letter_count+4+d*5] = self.letterDict["_"]

        #print(self.points)     
        self.vb.Bind()
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(c_float)*len(self.points),self.points)
           
        self.shader.SetUniform4f("u_PosRotScl",0,0,0,scl)
        self.shader.SetUniform1i("u_Texture",0)
        self.shader.SetUniform1i("u_is3d",1)


        mvp = np.transpose(np.matmul(viewMat,modelMat))        
        self.shader.SetUniformMat4f("u_MVP", mvp)

        renderer.Draw(self.va,self.ib,self.shader,drawUntil=len(subtxt)*6)
        glBindTexture(GL_TEXTURE_2D, 0)
        if longer:
            pass
            #self.drawText3D(txt[self.max_letter_count:],modelMat,scl,viewMat,renderer)
            
