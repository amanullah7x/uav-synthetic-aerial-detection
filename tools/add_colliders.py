#!/usr/bin/env python3
"""add_colliders.py - Mark the ground meshes of a converted Baylands USD as static triangle-mesh colliders.

    python add_colliders.py baylands.usd Textures/      (matches each mesh's texture against the source textures)
Trees, roofs and tables are left without collision so PhysX does not cook ~700k triangles at every start.
"""
import os
import sys

import numpy as np
from PIL import Image
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

GROUND = {"Grass", "DirtPath", "Asphalt", "Road", "Sand", "ParkingLotMedian", "ParkingMarkings", "ConcreteGround_01",
          "ConcreteGround_02", "ConcretePebbled", "Sidewalk", "Curb", "Curb_Red", "Concrete", "Tree_Concrete",
          "MurkyWater", "Checker", "Rubber"}

usd, src = sys.argv[1], sys.argv[2]
ref = {os.path.splitext(f)[0]: np.asarray(Image.open(os.path.join(src, f)).convert("RGB").resize((16, 16)), float)
       for f in os.listdir(src)}
stage = Usd.Stage.Open(usd)
d = os.path.dirname(os.path.abspath(usd))
for p in stage.Traverse():
    if not p.IsA(UsdGeom.Mesh):
        continue
    mat = UsdShade.MaterialBindingAPI(p).GetDirectBinding().GetMaterial()
    tex = stage.GetPrimAtPath(mat.GetPath().AppendChild("tex"))
    if not tex:
        continue
    im = np.asarray(Image.open(os.path.join(d, tex.GetAttribute("inputs:file").Get().path)).convert("RGB")
                    .resize((16, 16)), float)
    name = min(ref, key=lambda k: np.abs(ref[k] - im).mean())
    if name in GROUND:
        UsdPhysics.CollisionAPI.Apply(p)
        UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("none")
        print(f"collider: {p.GetName()} ({name})")
stage.GetRootLayer().Save()
