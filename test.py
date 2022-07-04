import Alg_Grf as ag
import numpy as np
import math

def main():

    grf = ag.SetupEnv()
    ag.drawAxes(grf)
    v = np.array([5,0,0], dtype=np.float16)
    u = np.array([1,4,5], dtype=np.float16)
    w = np.array([2,5,0], dtype=np.float16)

    
    ag.drawVec(grf, ag.VO, v)
    ag.drawVec(grf, ag.VO, u)
    print("angle=", ag.getAngle(v,u))
    print("cos=", np.cos(ag.getAngle(u,v)*math.pi/180))

    ag.ShowEnv()

if __name__ == "__main__":
    main()
