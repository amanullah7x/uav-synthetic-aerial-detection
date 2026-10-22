"""Give the Isaac Sim characters (101-joint 'RL_BoneRoot' rig, bind pose = T-pose) a relaxed standing pose.

The stock animation clips use a different rig and need retargeting, so instead a one-frame SkelAnimation is authored
from the skeleton's rest transforms with the upper arms rotated down (and a small elbow bend). The animation prim
lives outside the referenced asset, so the asset files are not modified.
"""
from pxr import Gf, Sdf, UsdSkel, Vt


def find_skeleton(stage, root_path):
    for p in stage.Traverse():
        if p.GetPath().HasPrefix(root_path) and p.GetTypeName() == "Skeleton":
            return UsdSkel.Skeleton(p)
    return None


def _rot(axis, deg):
    return Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(*axis), deg))


def relax_arms(stage, root_path, anim_path, arm_deg=-80.0, elbow_deg=-10.0, axis=(0, 0, 1)):
    """Author anim_path (a SkelAnimation) and bind it to the character's skeleton. Returns False if no rig found."""
    skel = find_skeleton(stage, root_path)
    if skel is None:
        return False
    joints = skel.GetJointsAttr().Get()
    rest = skel.GetRestTransformsAttr().Get()
    mats = [Gf.Matrix4d(m) for m in rest]
    for i, j in enumerate(joints):
        name = str(j).rsplit("/", 1)[-1]
        sign = 1.0 if name.startswith("L_") else -1.0
        if name in ("L_Upperarm", "R_Upperarm"):
            mats[i] = _rot(axis, sign * arm_deg) * mats[i]
        elif name in ("L_Forearm", "R_Forearm"):
            mats[i] = _rot(axis, sign * elbow_deg) * mats[i]
    anim = UsdSkel.Animation.Define(stage, anim_path)
    anim.GetJointsAttr().Set(joints)
    tr, ro, sc = [], [], []
    for m in mats:
        t, r, s = Gf.Vec3f(m.ExtractTranslation()), Gf.Quatf(m.ExtractRotationQuat()), Gf.Vec3h(1, 1, 1)
        tr.append(t), ro.append(r), sc.append(s)
    anim.GetTranslationsAttr().Set(Vt.Vec3fArray(tr))
    anim.GetRotationsAttr().Set(Vt.QuatfArray(ro))
    anim.GetScalesAttr().Set(Vt.Vec3hArray(sc))
    UsdSkel.BindingAPI.Apply(skel.GetPrim()).CreateAnimationSourceRel().SetTargets([Sdf.Path(anim_path)])
    return True
