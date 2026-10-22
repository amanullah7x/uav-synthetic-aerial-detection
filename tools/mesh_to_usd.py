#!/usr/bin/env python3
"""
mesh_to_usd.py - Convert an OBJ / Collada (DAE) mesh (e.g. a Gazebo Fuel model) to a Z-up, metre-scale USD with
UsdPreviewSurface materials and the textures copied next to it. Runs outside Isaac Sim (needs usd-core, trimesh,
pycollada, pillow).

    python mesh_to_usd.py in.obj out_dir/name.usd [--scale 0.0254] [--rotz 90] [--yup] [--center] [--ground]
"""
import argparse
import os
import shutil

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="mesh files, last argument is the output .usd")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--rotz", type=float, default=0.0, help="extra rotation about Z (deg)")
    ap.add_argument("--yup", action="store_true", help="source is Y-up: rotate +90 about X")
    ap.add_argument("--center", action="store_true", help="center X/Y on the origin")
    ap.add_argument("--ground", action="store_true", help="put the lowest point at Z=0")
    ap.add_argument("--z-offset", type=float, default=0.0)
    a = ap.parse_args()
    *srcs, out = a.inputs
    out_dir = os.path.dirname(os.path.abspath(out))
    tex_dir = os.path.join(out_dir, "textures")
    os.makedirs(tex_dir, exist_ok=True)

    T = np.eye(4)
    if a.yup:
        T = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]) @ T
    T = trimesh.transformations.rotation_matrix(np.radians(a.rotz), [0, 0, 1]) @ T
    T = trimesh.transformations.scale_matrix(a.scale) @ T

    geoms = []
    for src in srcs:
        scene = trimesh.load(src, force="scene", process=False)
        for node in scene.graph.nodes_geometry:
            M, gname = scene.graph[node]
            g = scene.geometry[gname].copy()
            g.apply_transform(T @ M)
            geoms.append((f"{os.path.splitext(os.path.basename(src))[0]}_{len(geoms)}", g))

    allv = np.vstack([g.vertices for _, g in geoms])
    lo, hi = allv.min(0), allv.max(0)
    shift = np.zeros(3)
    if a.center:
        shift[:2] = -(lo[:2] + hi[:2]) / 2
    if a.ground:
        shift[2] = -lo[2]
    shift[2] += a.z_offset
    print(f"bounds before shift: {lo.round(2)} .. {hi.round(2)}  size {np.round(hi - lo, 2)}  shift {shift.round(2)}")

    stage = Usd.Stage.CreateNew(out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Scope.Define(stage, "/Root/Looks")

    mats = {}
    img_keys = {}

    def material_for(g):
        vis = g.visual
        mat = getattr(vis, "material", None)
        img = None
        color = (0.6, 0.6, 0.6)
        if mat is not None:
            img = getattr(mat, "image", None) or getattr(mat, "baseColorTexture", None)
            c = getattr(mat, "diffuse", None)
            if c is None:
                c = getattr(mat, "baseColorFactor", None)
            if c is not None:
                c = np.asarray(c, dtype=float).ravel()
                if c.max() > 1.0:
                    c = c / 255.0
                color = tuple(c[:3])
        if img is not None:
            if id(img) not in img_keys:
                import hashlib
                img_keys[id(img)] = hashlib.md5(img.resize((32, 32)).tobytes()).hexdigest() + str(img.size)
            key = (img_keys[id(img)], None)
        else:
            key = (None, tuple(round(x, 3) for x in color))
        if key in mats:
            return mats[key]
        mp = f"/Root/Looks/mat_{len(mats)}"
        m = UsdShade.Material.Define(stage, mp)
        sh = UsdShade.Shader.Define(stage, mp + "/Surface")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
        if img is not None:
            fn = f"tex_{len(mats)}.png"
            img.convert("RGBA" if img.mode in ("RGBA", "LA", "P") else "RGB").save(os.path.join(tex_dir, fn))
            st = UsdShade.Shader.Define(stage, mp + "/st")
            st.CreateIdAttr("UsdPrimvarReader_float2")
            st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            tx = UsdShade.Shader.Define(stage, mp + "/tex")
            tx.CreateIdAttr("UsdUVTexture")
            tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./textures/{fn}")
            tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
            tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
            if img.mode == "RGBA" and np.asarray(img)[:, :, 3].min() < 128:   # cut-out (leaves, markings)
                tx.CreateOutput("a", Sdf.ValueTypeNames.Float)
                sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(tx.ConnectableAPI(), "a")
                sh.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.5)
        else:
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        mats[key] = m
        return m

    for name, g in geoms:
        if not isinstance(g, trimesh.Trimesh) or len(g.faces) == 0:
            continue
        path = "/Root/" + "".join(ch if ch.isalnum() else "_" for ch in name)
        mesh = UsdGeom.Mesh.Define(stage, path)
        v = g.vertices + shift
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(g.faces), 3, np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(g.faces.astype(np.int32).ravel()))
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateExtentAttr([Gf.Vec3f(*v.min(0)), Gf.Vec3f(*v.max(0))])
        uv = getattr(g.visual, "uv", None)
        if uv is not None and len(uv) == len(g.vertices):
            pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                         UsdGeom.Tokens.vertex)
            pv.Set(Vt.Vec2fArray.FromNumpy(np.asarray(uv, np.float32)))
        try:
            n = g.vertex_normals
            mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(n.astype(np.float32)))
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        except Exception:
            pass
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material_for(g))
    stage.GetRootLayer().Save()
    print(f"wrote {out}: {len(geoms)} meshes, {len(mats)} materials")


if __name__ == "__main__":
    main()
