import numpy as np
import math
import matplotlib
import matplotlib.pyplot as mp
import random as rd


#algebra________________________________________

def GenMat(h,w):
    A = np.zeros((h,w),dtype=np.float32)
    for i in range(h):
        for e in range(w):
            A[i,e] = rd.randint(1,100)

    return A

def MatrixMult(A,B):
    C = np.zeros((A.shape[0],B.shape[1]),dtype=np.float32)
    
    for i in range(C.shape[0]):
        for e in range(C.shape[1]):
            C[i,e] = np.dot(A[i,:],B[:,e])
    
    return C

def ReverseVec(A):
    n = len(A)
    C = np.zeros(n,dtype = np.float32)
    B = np.zeros((n,n),dtype = np.float32)
    
    for i in range(n): B[i,n-i-1] = 1

    for i in range(n):
        C[i] = np.dot(A,B[:,i])

    return C


#vector drawing_________________________

VO =np.array([0, 0, 0],dtype=np.float64)
i = np.array([1, 0, 0],dtype=np.float64)
j = np.array([0, 1, 0],dtype=np.float64)
k = np.array([0, 0, 1],dtype=np.float64)

def drawLine(grf,Point1,Point2,style="solid",color="gray"):
    grf.plot([Point1[0],Point2[0]],[Point1[1],Point2[1]],[Point1[2],Point2[2]],linestyle=style,color=color)

def drawSurface(grf,Point1,Point2,Direction,Width,color="gray"):
    for i in range(Width*10+1):
        drawLine(grf,Point1+(i/10)*Direction,Point2+(i/10)*Direction,color=color)
    return Point1+Direction*Width,Point2+Direction*Width    

def drawAxes(grf,tr1=-10,tr2=10):
    
    grf.set_xlim([-10,10])
    grf.set_ylim([-10,10])
    grf.set_zlim([-10,10])

    grf.plot([tr1,tr2],[0,0],[0,0],linestyle="dashed",color="gray")
    grf.plot([0,0],[tr1,tr2],[0,0],linestyle="dashed",color="gray")
    grf.plot([0,0],[0,0],[tr1,tr2],linestyle="dashed",color="gray")

def drawVec(grf,org,vec,col="purple"):
    grf.quiver(*org,*vec,color=col)

def getLen(vec):
    return math.sqrt(np.dot(vec,vec))

def getAngle(vec1,vec2):
    x=np.dot(vec1,vec2)
    print(x)
    y=getLen(vec1)*getLen(vec2)
    print(y)
    x=x/y
    x=np.arccos(x/y)
    return x/math.pi

def ProjMat(support):
    x = np.zeros((3,3),dtype=np.float32)
    for i in range(3):
        for e in range(3):
            x[i,e] = support[i] * support[e]

    x *= 1/np.dot(support,support)
    return x

def ProjMatP(support):    
    tr = np.transpose(support.copy())
    sq = MatrixMult(tr.copy(),support.copy())
    sq = np.linalg.inv(sq)
    support = MatrixMult(support,sq)
    return MatrixMult(support,tr)

def ProjOnLine(vec,support):
    P1 = MatrixMult(ProjMat(support),np.vstack(vec))
    P1 = [P1[0,0],P1[1,0],P1[2,0]]
    E1 = vec-P1
    return P1,E1

def ProjOnPlane(vec,support):
    Pp = MatrixMult(ProjMatP(support),np.vstack(vec))
    Pp = [Pp[0,0],Pp[1,0],Pp[2,0]]
    Ep = vec-Pp
    return Pp,Ep

def SetupEnv():
    return mp.axes(projection="3d")
