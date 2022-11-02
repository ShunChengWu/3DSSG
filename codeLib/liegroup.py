import numpy as np

def InV_SO3(x:np.array):
    return x.transpose()

def InV_SE3(x):
    R = x[:3,:3].transpose()
    t = -R@x[0:3,3]
    output =np.eye(4)
    output[:3,:3]=R
    output[0:3,3]=t
    return output