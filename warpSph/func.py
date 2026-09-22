from .lib import *

@wp.func
def sphKerSpline(dist:wp.float32, h:wp.float32, sig:wp.float32):
    q = dist / h
    tmp = 0.0
    if q > 0.5 and q < 1.0:
        tmp = 2.0 * wp.pow((1.0 - q), 3.0) * sig
    elif q <= 0.5:
        tmp = (6.0 * (wp.pow(q, 3.0) - wp.pow(q, 2.0)) + 1.0) * sig
    return tmp

@wp.func
def sphKerSplineGrad(dist:wp.float32, h:wp.float32, sigInvH:wp.float32):
    q = wp.float32(dist / h)
    tmp = 0.0
    if q > 0.5 and q < 1.0:
        tmp = -6.0 * (wp.pow(1.0 - q, 2.0)) * sigInvH
    elif q <= 0.5:
        tmp = 6.0 * (3.0 * wp.pow(q, 2.0) - (2.0 * q)) * sigInvH
    return tmp

@wp.func
def sphLaplacian(dist:wp.float32, dim:wp.int32, grad:wp.float32, Vj:wp.float32, xij:Any, Aij:Any):
    return 2.0 * (2.0 + float(dim)) * Vj * grad * xij * wp.dot(Aij, xij) / (dist**3.0)
