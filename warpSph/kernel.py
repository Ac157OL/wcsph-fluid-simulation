from .lib import *
from .func import *
from .data_struct import vecxf

@wp.kernel
def set(a:wp.array(dtype=Any), val:Any):
    i = wp.tid()
    a[i] = val

@wp.kernel
def copy(toArr:wp.array(dtype=Any), fromArr:wp.array(dtype=Any)):
    i = wp.tid()
    toArr[i] = fromArr[i]

@wp.kernel
def integral(intArr:wp.array(dtype=Any), delta:Any, arr:wp.array(dtype=Any)):
    i = wp.tid()
    arr[i] += intArr[i] * delta

@wp.kernel
def levelUp(arr:wp.array(dtype=Any), level:Any):
    i = wp.tid()
    if arr[i] < level: 
        arr[i] = level
@wp.kernel
def levelDown(arr:wp.array(dtype=Any), level:Any):
    i = wp.tid()
    if arr[i] > level: 
        arr[i] = level

@wp.kernel
def accumulate(arr:wp.array(dtype=Any), intArr:wp.array(dtype=Any)):
    i = wp.tid()
    arr[i] += intArr[i]


@wp.kernel
def sphAccumulateDensity(position: wp.array(dtype=vecxf),
                         density: wp.array(dtype=wp.float32),
                         neighbor_position: wp.array(dtype=vecxf),
                         neighbor_mass: wp.array(dtype=wp.float32),
                         neighbor_grid: wp.uint64,
                         h: wp.float32,
                         sig: wp.float32):
    """Accumulate rho_i = sum_j m_j W_ij over one particle set."""
    i = wp.tid()
    xi = position[i]
    query = wp.hash_grid_query(neighbor_grid, xi, h)
    j = wp.int32(0)
    while wp.hash_grid_query_next(query, j):
        distance = wp.length(xi - neighbor_position[j])
        if distance < h:
            density[i] += neighbor_mass[j] * sphKerSpline(distance, h, sig)

@wp.kernel
def sphGravityAcceleration(acceleration: wp.array(dtype=vecxf), gravity: vecxf):
    i = wp.tid()
    acceleration[i] += gravity

@wp.kernel
def sphWcPressure(density: wp.array(dtype=wp.float32),
                  pressure: wp.array(dtype=wp.float32),
                  rest_density: wp.float32,
                  pressure_B: wp.float32):
    """Tait equation of state, P=B*((rho/rho0)^gamma-1), gamma=7."""
    i = wp.tid()
    ratio = density[i] / rest_density
    pressure[i] = pressure_B * (wp.pow(ratio, WC_GAMMA) - 1.0)

@wp.kernel
def sphWcPressureAcceleration(position: wp.array(dtype=vecxf),
                              velocity: wp.array(dtype=vecxf),
                              density: wp.array(dtype=wp.float32),
                              pressure: wp.array(dtype=wp.float32),
                              acceleration: wp.array(dtype=vecxf),
                              neighbor_position: wp.array(dtype=vecxf),
                              neighbor_velocity: wp.array(dtype=vecxf),
                              neighbor_density: wp.array(dtype=wp.float32),
                              neighbor_pressure: wp.array(dtype=wp.float32),
                              neighbor_mass: wp.array(dtype=wp.float32),
                              neighbor_grid: wp.uint64,
                              h: wp.float32,
                              sig_inv_h: wp.float32,
                              viscosity: wp.float32,
                              artificial_viscosity_alpha: wp.float32,
                              sound_speed: wp.float32):
    """Symmetric pressure force, Morris viscosity, and paper artificial viscosity."""
    i = wp.tid()
    xi = position[i]
    vi = velocity[i]
    rhoi = density[i]
    pi = pressure[i]
    ai = vecxf(0.0)

    query = wp.hash_grid_query(neighbor_grid, xi, h)
    j = wp.int32(0)
    while wp.hash_grid_query_next(query, j):
        xij = xi - neighbor_position[j]
        distance = wp.length(xij)
        if distance > INF_SMALL and distance < h:
            rhoj = neighbor_density[j]
            radial_grad = sphKerSplineGrad(distance, h, sig_inv_h)
            grad_w = radial_grad * xij / distance
            pressure_term = pi / (rhoi * rhoi) + neighbor_pressure[j] / (rhoj * rhoj)
            ai -= neighbor_mass[j] * pressure_term * grad_w

            # Becker & Teschner / Monaghan artificial viscosity.  It is active
            # only for approaching pairs, so it dissipates compression waves
            # and impacts without applying an unconditional global drag.
            vij = vi - neighbor_velocity[j]
            approaching = wp.dot(vij, xij)
            if approaching < 0.0 and artificial_viscosity_alpha > 0.0:
                nu_bar = (2.0 * artificial_viscosity_alpha * h * sound_speed) / (rhoi + rhoj)
                pi_ij = -nu_bar * approaching / (distance * distance + 0.01 * h * h)
                ai -= neighbor_mass[j] * pi_ij * grad_w

            # Morris-style viscosity; xij.gradW is non-negative for this kernel.
            volume_j = neighbor_mass[j] / rhoj
            visc_scale = 10.0 * volume_j  # 2*(d+2), d=3
            visc_scale *= wp.dot(xij, grad_w) / (distance * distance + 0.01 * h * h)
            # xij·grad(W) <= 0, so (vi-vj) makes this term dissipative.
            ai += viscosity * visc_scale * vij
    acceleration[i] += ai
            
