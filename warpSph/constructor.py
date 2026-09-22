from .lib import *
from .data_struct import vecxf

class CubeData():
    def __init__(self,
                 span:wp.float32,
                 dim:wp.int32,
                 lb:vecxf,
                 rt:vecxf,
                 ):
        self.span   = wp.float32(span)
        self.dim    = wp.int32(dim)
        self.lb     = vecxf(lb)
        self.rt     = vecxf(rt)

        self.shape   = np.ceil((self.rt - self.lb) / span).astype(np.int32)
        self.partPos = None
        self.partNum = None
        
        pos_frac = []
        index_frac = []
        for i in range(self.dim):
            pos_frac.append(np.linspace(self.lb[i], self.lb[i]+span*wp.float32(self.shape[i]), self.shape[i]+1))
            index_frac.append(np.linspace(0,self.shape[i], self.shape[i]+1).astype(np.int32))
        
        # returned values
        self.partPos    = np.array(np.meshgrid(*pos_frac)).T.reshape(-1, self.dim) # the pos array takes the form of (num, dim)
        self.partIndex  = np.array(np.meshgrid(*index_frac)).T.reshape(-1, self.dim) # the index array takes the form of (num, dim)
        self.partNum    = self.partPos.shape[0]


class SphereData():
    """Generate a uniformly spaced solid sphere of SPH particles."""
    def __init__(self,
                 span: wp.float32,
                 center: vecxf,
                 radius: wp.float32,
                 ):
        self.span = float(span)
        self.center = np.array([float(center[i]) for i in range(3)], dtype=np.float32)
        self.radius = float(radius)

        axes = [np.arange(c - self.radius, c + self.radius + 0.5 * self.span,
                          self.span, dtype=np.float32) for c in self.center]
        candidates = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        inside = np.sum((candidates - self.center) ** 2, axis=1) <= self.radius ** 2
        self.partPos = np.ascontiguousarray(candidates[inside], dtype=np.float32)
        self.partNum = self.partPos.shape[0]

class PoolData():
    def __init__(self,
                    containerHeight:wp.float32,
                    containerWidth:wp.float32,
                    containerLayer:wp.int32,
                    fluidHeight:wp.float32,
                    span:wp.float32,
                 ):
        self.containerHeight = float(containerHeight)
        self.containerWidth  = float(containerWidth)
        self.containerLayer  = int(containerLayer)
        self.fluidHeight     = float(fluidHeight)
        self.span            = float(span)

        self.fluidEmptyHeight = float(self.containerHeight - self.fluidHeight/2.0)
        self.grid_x, self.grid_y, self.grid_z = \
            np.mgrid[\
                -self.containerWidth/2.0 : self.containerWidth/2.0 : self.span, \
                -self.containerHeight/2.0 : self.containerHeight/2.0 : self.span,\
                -self.containerWidth/2.0 : self.containerWidth/2.0 : self.span,]
        
        self.mask_inner_space =  (self.grid_x > -self.containerWidth/2.0 + self.span*wp.float32(self.containerLayer)) & (self.grid_x < self.containerWidth/2.0 - self.span*wp.float32(self.containerLayer)) & \
                            (self.grid_y > -self.containerHeight/2.0 + self.span*wp.float32(self.containerLayer)) & (self.grid_y < self.containerHeight/2.0 - self.span*wp.float32(self.containerLayer)) & \
                            (self.grid_z > -self.containerWidth/2.0 + self.span*wp.float32(self.containerLayer)) & (self.grid_z < self.containerWidth/2.0 - self.span*wp.float32(self.containerLayer))
        
        self.mask_bound = ~self.mask_inner_space
        self.mask_fluid = self.mask_inner_space & (self.grid_y < self.fluidEmptyHeight) \
            # & (self.grid_x < 0)

        self.fluid_position_x = self.grid_x[self.mask_fluid]
        self.fluid_position_y = self.grid_y[self.mask_fluid]
        self.fluid_position_z = self.grid_z[self.mask_fluid]

        self.bound_position_x = self.grid_x[self.mask_bound]
        self.bound_position_y = self.grid_y[self.mask_bound]
        self.bound_position_z = self.grid_z[self.mask_bound]
        
        self.fluidPartPos = np.stack((self.fluid_position_x.reshape(-1), self.fluid_position_y.reshape(-1), self.fluid_position_z.reshape(-1)), -1)
        self.fluidPartNum = self.fluidPartPos.shape[0]
        self.poolPartPos  = np.stack((self.bound_position_x.reshape(-1), self.bound_position_y.reshape(-1), self.bound_position_z.reshape(-1)), -1)
        self.poolPartNum  = self.poolPartPos.shape[0]
