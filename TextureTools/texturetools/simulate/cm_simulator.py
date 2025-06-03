import numpy as np
import trimesh
import fcl

def get_fcl_obj(mesh:trimesh.Trimesh):
    if mesh.is_convex:
        obj = trimesh.collision.mesh_to_convex(mesh)
    else:
        obj = trimesh.collision.mesh_to_BVH(mesh)
    return obj

if __name__ == '__main__':
    m1 = trimesh.creation.uv_sphere(1.0)
    t10 = np.array([-10.0, 0.0, 0.0])
    t11 = np.array([1.0, 0.0, 0.0])
    m2 = trimesh.creation.uv_sphere(1.0)
    t20 = np.array([10.0, 0.0, 0.0])
    t21 = np.array([1.0, 0.0, 0.0])

    request = fcl.ContinuousCollisionRequest()
    result = fcl.ContinuousCollisionResult()
    ret = fcl.continuousCollide(
        fcl.CollisionObject(
            get_fcl_obj(m1), 
            fcl.Transform(t10)
        ), 
        fcl.Transform(t11), 
        fcl.CollisionObject(
            get_fcl_obj(m2), 
            fcl.Transform(t20)
        ), 
        fcl.Transform(t21), 
        request, 
        result,
    )
    t = result.time_of_contact
    m1.apply_translation(t10 + t * (t11 - t10))
    m2.apply_translation(t20 + t * (t21 - t20))
    s = trimesh.Scene({'m1': m1, 'm2': m2})
    s.export('debug.glb')


