import time
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def gb3(mat, coord, size):
    assert len(mat.shape) == 3
    def iv3(mat):
        return np.sum(mat)
    w, h, l = size
    # w,h,l = l,h,w
    w, h, l = w+1, h+1, l+1
    x, y, t = coord
    # x,y,t=t,y,x
    gb = (iv3(mat[0:x+w, 0:y+h, 0:t+l]) - iv3(mat[0:x, 0:y+h, 0:t+l]) - iv3(mat[0:x+w, 0:y, 0:t+l]) + iv3(mat[0:x, 0:y, 0:t+l])) - (iv3(mat[0:x+w, 0:y+h, 0:t]) - iv3(mat[0:x, 0:y+h, 0:t]) - iv3(mat[0:x+w, 0:y, 0:t]) + iv3(mat[0:x, 0:y, 0:t]))  
    return gb


def hog3d(stcube, label = True, SIZE = (10, 4), STEP = (10, 4), BC_DIV = 2, IF_SHOW_PATCH = False, IF_PLOT_HOG_FEATURE = True):
    # ---------------------------------------------------------------------/
    CSIZE, TSIZE = SIZE # set cell size

    CSTEP, TSTEP = STEP # set spatio and temporal step for main blocks

    THRES = 1.29107

    PHI = 0.5 * (1 + np.sqrt(5))
    # this is projection matrix
    PROJ = np.array([[1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1], [-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1], 
    [0,1/PHI,PHI], [0,1/PHI,-PHI], [0,-1/PHI,PHI], [0,-1/PHI,-PHI], 
    [1/PHI,PHI,0], [1/PHI,-PHI,0], [-1/PHI,PHI,0], [-1/PHI,-PHI,0], 
    [PHI,0,1/PHI], [PHI,0,-1/PHI], [-PHI,0,1/PHI], [-PHI,0,-1/PHI]])

    # CALC HOG3d IN EACH ST_CUBE
    #{
    
    [dt, dy, dx] = np.gradient(stcube, 1, 1, 1) # NAIVE grad in 3-dims

    t_bgrid = np.arange(0, stcube.shape[0], TSTEP)   
    y_bgrid = np.arange(0, stcube.shape[1], CSTEP)
    x_bgrid = np.arange(0, stcube.shape[2], CSTEP)
    
    # BC_DIV = 2
    
    w, h, l = int(CSIZE / BC_DIV), int(CSIZE/BC_DIV), int(TSIZE/BC_DIV)
    fhog = np.array([])
    # in this looop we process each cell
    for xb in x_bgrid:
        for yb in y_bgrid:
            for tb in t_bgrid:
                # each subblock
                Hc = np.zeros((20,))
                for xc in range(BC_DIV):
                    for yc in range(BC_DIV):
                        for tc in range(BC_DIV):
                            x = xb + xc*w
                            y = yb + yc*h
                            t = tb + tc*l
                            gb_x = gb3(dx, (t, y, x), (l, h, w))
                            gb_y = gb3(dy, (t, y, x), (l, h, w))
                            gb_t = gb3(dt, (t, y, x), (l, h, w))
                            
                            gb = np.array([gb_x, gb_y, gb_t])

                            # print(x, y) # checkpoint
                            
                            qb = np.matmul(PROJ, gb) / np.linalg.norm(gb) #--(5)

                            qb = qb - THRES
                            qb[qb < 0] = 0

                            qb = np.linalg.norm(gb) * qb / np.linalg.norm(qb) 

                            Hc = Hc + qb
                            
                            Hc  = Hc / np.linalg.norm(Hc)

                # print(Hc)
                fhog = np.concatenate((fhog, Hc.flatten()))
    if IF_PLOT_HOG_FEATURE:
        plt.plot(fhog);plt.title(label);plt.show()

    return fhog
