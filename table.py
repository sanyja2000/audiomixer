import numpy as np
import json
from classHandler import *
from engine.objectHandler import prefabHandler
from functools import partial

class Table:
    def __init__(self,filename):
        """This class holds all of the map information and objects."""
        print("LOG: Loading mapfile: "+filename)
        self.mapFile = ""
        self.objects = []
        self.prefabHandler = prefabHandler()
        self.puzzle = None
        self.beatLength = 115
        with open(filename, "r") as f:
            self.JSONContent = json.loads("".join(f.readlines()))
            self.type = self.JSONContent["type"]
            for obj in self.JSONContent["objectList"]:
                if obj["type"]=="decoration":
                    self.objects.append(Decoration(self.prefabHandler,obj))
                elif obj["type"]=="comment":
                    pass
                else:
                    print("ERROR: Unknown type for object in json: "+obj["type"])
                    print(obj)
    def loadNewMap(self,filename,player):
        self.__init__(filename,player)
    def getObject(self,objName):
        """Returns an object with name:"objName". If it doesn't exist returns None."""
        for o in self.objects:
            if o.name == objName:
                return o
        return None
