import numpy as np

def processObjFile(filename):
    scale = 1
    with open(filename,"r") as objfile:
        file = objfile.read()
    # vblayout 3pos 2tex 3norm
    vertices = []
    texcoords = []
    normals = []
    vb = []
    indices = []
    lines = file.split("\n")
    faceIndex = -1
    quads = False
    texAvailable = True
    for line in lines:
        if len(line)==0 or line[0] == "#":
            continue
        words = line.split()
        if words[0] == "v":
            vertices.append([float(words[1])*scale,float(words[2])*scale,float(words[3])*scale])
        if words[0] == "vt":
            texcoords.append([float(words[1]),float(words[2])])
        if words[0] == "vn":
            normals.append([float(words[1]),float(words[2]),float(words[3])])
        if words[0] == "f":
            if(len(words) == 5):
                faceIndex = lines.index(line)
                quads = True
                break
            else:
                faceIndex = lines.index(line)
                quads = False
                break
    if len(texcoords) == 0:
        print("No texture coordinates found for file:"+filename)
        texAvailable = False
    if quads:
        n = 0
        for line in lines[faceIndex:]:
            if len(line) > 0:
                if line[0] == "f":
                    words = line.split()[1:]
                    for part in words:
                        ind = part.split("/")
                        if texAvailable:
                            vb += vertices[int(ind[0])-1]+texcoords[int(ind[1])-1]+normals[int(ind[2])-1]
                        else:
                            vb += vertices[int(ind[0])-1]+[0,0]+normals[int(ind[2])-1]
                    indices+=[n,n+1,n+2,n+2,n+3,n]
                    n+=4
                elif line[0] == "s":
                    pass
                else:
                    break
    else:
        n = 0
        for line in lines[faceIndex:]:
            if len(line) > 0:
                if line[0] == "f":
                    words = line.split()[1:]
                    for part in words:
                        ind = part.split("/")
                        if texAvailable:
                            vb += vertices[int(ind[0])-1]+texcoords[int(ind[1])-1]+normals[int(ind[2])-1]
                        else:
                            vb += vertices[int(ind[0])-1]+[0,0]+normals[int(ind[2])-1]
                    indices+=[n,n+1,n+2]
                    n+=3
                elif line[0] == "s":
                    pass
                else:
                    break
    
    return [np.array(vb,dtype='float32'),np.array(indices)]
