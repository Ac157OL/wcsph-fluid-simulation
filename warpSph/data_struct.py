from .lib import *

dim = wp.constant(3)
vecxf = wp.types.vector(length=dim, dtype=wp.float32)

class Config():
    def __init__(self, 
                 dim:wp.constant,
                 partSize:wp.float32, 
                 simTime:wp.float32=30.0,
                 kViscosity:wp.float32=1e-2,
                 artificialViscosityAlpha:wp.float32=0.0,
                 restDensity:wp.float32=1000.0,
                 soundSpeed:wp.float32=1500.0,
                 lb:vecxf=vecxf(0),
                 rt:vecxf=vecxf(1),
                 ):
        self.dim            = wp.constant(dim)
        self.partSize       = wp.float32(partSize)
        self.simTime        = wp.float32(simTime)
        self.kViscosity     = wp.float32(kViscosity)
        self.artificialViscosityAlpha = wp.float32(artificialViscosityAlpha)
        self.restDensity    = wp.float32(restDensity)
        self.soundSpeed     = wp.float32(soundSpeed)
        self.lb             = vecxf(lb)
        self.rt             = vecxf(rt)

        self.partSizeInv    = wp.float32(wp.float32(1.0)/self.partSize)
        self.partVolume     = wp.float32(wp.pow(self.partSize, wp.float32(3.0)))
        self.cflDt          = wp.float32(0.4 * (2.0 * self.partSize) / self.soundSpeed)
        self.dt             = wp.float32(min(float(self.partSize/wp.float32(DT_WEIGHT)),
                                             0.8 * float(self.cflDt)))
        self.dtSqr          = wp.float32(self.dt * self.dt)
        self.dtInv          = wp.float32(1.0/self.dt)
        self.dtSqrInv       = wp.float32(1.0/self.dtSqr)
        self.gravity        = wp.types.vector(length=dim, dtype=wp.float32)(0)
        self.gravity[1]     = -9.8   

class Particle():
    def __init__(self, shape:int, config:Config):
        self.shape = shape
        self.mass           = wp.zeros(shape=self.shape, dtype=wp.float32)
        self.pressure       = wp.zeros(shape=self.shape, dtype=wp.float32)
        self.position       = wp.zeros(shape=self.shape, dtype=wp.types.vector(length=config.dim, dtype=wp.float32))
        self.velocity       = wp.zeros(shape=self.shape, dtype=wp.types.vector(length=config.dim, dtype=wp.float32))
        self.acceleration   = wp.zeros(shape=self.shape, dtype=wp.types.vector(length=config.dim, dtype=wp.float32))

        self.sphDensity     = wp.zeros(shape=self.shape, dtype=wp.float32)
        self.sphCompressionRatio = wp.zeros(shape=self.shape, dtype=wp.float32)

        self.sphSig         = wp.zeros(shape=self.shape, dtype=wp.float32)
        self.sphSigInvH     = wp.zeros(shape=self.shape, dtype=wp.float32)

        self.sphDfAlpha     = wp.zeros(shape=self.shape, dtype=wp.float32)
        self.sphDfAlpha1    = wp.zeros(shape=self.shape, dtype=wp.types.vector(length=config.dim, dtype=wp.float32))
        self.sphDfAlpha2    = wp.zeros(shape=self.shape, dtype=wp.float32)
        self.sphDfDelta     = wp.zeros(shape=self.shape, dtype=wp.float32)
        self.sphDfVelAdv    = wp.zeros(shape=self.shape, dtype=wp.types.vector(length=config.dim, dtype=wp.float32))
        self.sphDfKappa0    = wp.zeros(shape=self.shape, dtype=wp.float32)
        self.sphDfKappa1    = wp.zeros(shape=self.shape, dtype=wp.float32)

class SPH():
    def __init__(self, 
                 config:Config,
                 ):
        self.h              = wp.float32(config.partSize * wp.float32(2.0))
        self.sig            = wp.float32(self.computeSig(config.dim))
        self.sigInvH        = wp.float32(self.sig / self.h)
        self.wcsphB         = wp.float32(config.restDensity * config.soundSpeed *
                                         config.soundSpeed / WC_GAMMA)

    def computeSig(self, dim):
        sig = 0.0
        if dim == 3:
            sig = 8.0 / wp.pi 
        elif dim == 2:
            sig = 40.0 / 7.0 / wp.pi
        elif dim == 1:
            sig = 4.0 / 3.0
        sig = sig / wp.pow(self.h, wp.float32(dim))
        return sig
    
        
