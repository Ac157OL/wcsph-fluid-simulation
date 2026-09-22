from .lib import *
from .data_struct import *
from .func import *
from .kernel import *

def stepWCSPH(objList: List[Particle], 
              dynamicList: List[bool], 
              gridList: List[wp.HashGrid], 
              config: Config, 
              sph: SPH,):
    objNum = len(objList)

    for obj in objList:
        wp.launch(set, dim=obj.shape, inputs=[obj.sphDensity, 0.0])
        wp.launch(set, dim=obj.shape, inputs=[obj.acceleration, wp.vec3f(0.0)])

    for objIndex in range(objNum):
        obj = objList[objIndex]
        for neighbObjIndex in range(objNum):
            neighbObj = objList[neighbObjIndex]
            neighbObjGrid = gridList[neighbObjIndex]

            wp.launch(sphAccumulateDensity, dim=obj.shape,
                      inputs=[obj.position, obj.sphDensity,
                              neighbObj.position, neighbObj.mass,
                              neighbObjGrid.id, sph.h, sph.sig])
        
        wp.launch(levelUp, dim=obj.shape, inputs=[obj.sphDensity, config.restDensity])
    
    for objIndex in range(objNum):
        obj = objList[objIndex]
        # Boundary particles need an equation-of-state pressure as well.  They
        # stay kinematic, but their pressure participates in the symmetric
        # fluid-boundary force during impact.
        wp.launch(sphWcPressure, dim=obj.shape,
                  inputs=[obj.sphDensity, obj.pressure,
                          config.restDensity, sph.wcsphB])
        if dynamicList[objIndex]:
            wp.launch(sphGravityAcceleration, dim=obj.shape,
                      inputs=[obj.acceleration, config.gravity])
                        


    for objIndex in range(objNum):
        obj = objList[objIndex]
        if not dynamicList[objIndex]:
            continue
        for neighbObjIndex in range(objNum):
            neighbObj = objList[neighbObjIndex]
            neighbObjGrid = gridList[neighbObjIndex]

            wp.launch(sphWcPressureAcceleration, dim=obj.shape,
                      inputs=[obj.position, obj.velocity, obj.sphDensity,
                              obj.pressure, obj.acceleration,
                              neighbObj.position, neighbObj.velocity,
                              neighbObj.sphDensity, neighbObj.pressure,
                              neighbObj.mass, neighbObjGrid.id,
                              sph.h, sph.sigInvH, config.kViscosity,
                              config.artificialViscosityAlpha, config.soundSpeed])

    
    for objIndex in range(objNum):
        if dynamicList[objIndex]:
            obj = objList[objIndex]
            wp.launch(integral, dim=obj.shape,
                      inputs=[obj.acceleration, config.dt],
                      outputs=[obj.velocity])
            wp.launch(integral, dim=obj.shape,
                      inputs=[obj.velocity, config.dt],
                      outputs=[obj.position])

            
