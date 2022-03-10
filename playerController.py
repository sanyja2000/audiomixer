import math,pyrr, numpy as np
from classHandler import dist

def lerp(f, t, n):
    return f*(1-n)+t*n

def lerpVec3(f,t,n):
    out = []
    for x in range(3):
        out.append(f[x]*(1-n)+t[x]*n)
    return np.array(out)

