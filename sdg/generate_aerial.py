"""Synthetic aerial detection data from the Baylands park scene with Isaac Sim Replicator.

Run with Isaac Sim's Python (Isaac Sim 6.0):

    ~/isaacsim/python.sh sdg/generate_aerial.py --frames 5000 --out data/synthetic

Every frame is a fresh random draw of
  * camera: altitude 8..120 m above the ground it looks at, pitch 30..89 deg below the horizon, any yaw,
    horizontal FOV 55..95 deg, 1280x720;
  * objects: up to 18 cars (4 models, random body tint) and up to 30 people (Gazebo Fuel + Isaac Sim characters),
    dropped on the terrain where random image rays hit the ground, people partly in small groups;
  * lighting: sun elevation 8..85 deg, any azimuth, colour temperature 3000..7500 K, sky-fill strength
    (clear to overcast), plus a global tint on the park's ground textures.

Labels come from Replicator's tight 2D boxes (visible pixels only) and are written as YOLO txt with VisDrone class
ids: person -> 0 (pedestrian), car -> 3 (car). Boxes under 3 px or more than 75 % occluded are dropped.
Output: <out>/images/*.jpg, <out>/labels/*.txt, <out>/meta.csv (per-frame camera/light parameters).
"""

import argparse
import csv
import glob
import math
import os
import random
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=100)
parser.add_argument("--out", default="data/synthetic")
parser.add_argument("--start", type=int, default=0, help="first frame index (to resume or add to a run)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--rt-subframes", type=int, default=4, help="render subframes per capture (settles lighting)")
parser.add_argument("--baylands", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "baylands", "baylands.usd"))
parser.add_argument("--assets", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets"))
parser.add_argument("--max-cars", type=int, default=18)
parser.add_argument("--max-people", type=int, default=30)
parser.add_argument("--headless", action="store_true", default=True)
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless, "renderer": "RaytracedLighting",
                                "width": args.width, "height": args.height})

import numpy as np
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.core.experimental.utils.semantics import add_labels
from omni.physics.core import get_physics_scene_query_interface
from PIL import Image
from pxr import Gf, PhysicsSchemaTools, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import posing  # noqa: E402

CLASS_ID = {"pedestrian": 0, "car": 3}  # VisDrone ids
rng = random.Random(args.seed)
np_rng = np.random.default_rng(args.seed)

omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")  # only used for terrain ray casts

# ---------------------------------------------------------------- environment
park = stage.DefinePrim("/World/Baylands", "Xform")
park.GetReferences().AddReference(os.path.abspath(args.baylands))
sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
sun.CreateAngleAttr(1.0)
sun.CreateEnableColorTemperatureAttr(True)
dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
# Large earth-coloured plane below the park so views past its edge show land, not the empty background
outer = UsdGeom.Mesh.Define(stage, "/World/OuterGround")
outer.CreatePointsAttr([(-3000, -3000, -13), (3000, -3000, -13), (3000, 3000, -13), (-3000, 3000, -13)])
outer.CreateFaceVertexCountsAttr([4])
outer.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
outer_color = outer.CreateDisplayColorAttr([(0.33, 0.36, 0.22)])
simulation_app.update()

# Ground meshes already carry colliders (tools/add_colliders.py). Trees, roofs and other props get triangle-mesh
# colliders here too (in memory only), so a ray from the camera tells ground from canopy: objects are only dropped
# where the camera actually sees ground, never hidden under a tree or inside a building.
GROUND = set()
for p in stage.Traverse():
    if p.GetPath().HasPrefix("/World/Baylands") and p.GetTypeName() == "Mesh":
        if p.HasAPI(UsdPhysics.CollisionAPI):
            GROUND.add(str(p.GetPath()))
        else:
            UsdPhysics.CollisionAPI.Apply(p)
            UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("none")
print(f"[sdg] {len(GROUND)} ground meshes", flush=True)

# Ground texture shaders of the park (for a global tint per frame)
park_textures = [UsdShade.Shader(p) for p in stage.Traverse()
                 if p.GetPath().HasPrefix("/World/Baylands") and p.GetTypeName() == "Shader"
                 and UsdShade.Shader(p).GetIdAttr().Get() == "UsdUVTexture"]


# ---------------------------------------------------------------- object pool
def find_usds(pattern):
    return sorted(glob.glob(os.path.join(args.assets, pattern)))


CAR_USDS = find_usds("car_*/*.usd")
PERSON_USDS = find_usds("person_*/*.usd") + [u for u in find_usds("isaac_people/*/*.usd") if "/Animations/" not in u]
assert CAR_USDS and PERSON_USDS, f"no assets found under {args.assets} (run tools/fetch_assets.sh)"
print(f"[sdg] {len(CAR_USDS)} car models, {len(PERSON_USDS)} person models, {len(park_textures)} park textures",
      flush=True)


def body_textures(root):
    """UsdUVTexture shaders under an asset that are not wheels/eyes etc. (those keep their colour)."""
    out = []
    for p in stage.Traverse():
        if not p.GetPath().HasPrefix(root.GetPath()) or p.GetTypeName() != "Shader":
            continue
        sh = UsdShade.Shader(p)
        if sh.GetIdAttr().Get() != "UsdUVTexture":
            continue
        f = sh.GetInput("file")
        name = os.path.basename(str(f.Get().path)).lower() if f and f.Get() else ""
        if any(k in name for k in ("wheel", "interior", "eye", "lash", "brow", "tongue", "teeth", "normal")):
            continue
        out.append(sh)
    return out


class Obj:
    def __init__(self, path, usd, label):
        # wrapper Xform we pose; the asset (which may carry its own xform ops) is referenced one level down
        self.prim = stage.DefinePrim(path, "Xform")
        stage.DefinePrim(path + "/asset", "Xform").GetReferences().AddReference(os.path.abspath(usd))
        add_labels(self.prim, labels=[label], taxonomy="class")
        self.xf = UsdGeom.Xformable(self.prim)
        self.xf.ClearXformOpOrder()
        self.t = self.xf.AddTranslateOp()
        self.r = self.xf.AddRotateZOp()
        self.s = self.xf.AddScaleOp()
        self.label = label
        self.textures = []
        self.hide()

    def place(self, pos, yaw, scale):
        self.t.Set(Gf.Vec3d(*pos))
        self.r.Set(float(yaw))
        self.s.Set(Gf.Vec3f(scale, scale, scale))
        UsdGeom.Imageable(self.prim).MakeVisible()

    def hide(self):
        UsdGeom.Imageable(self.prim).MakeInvisible()

    def tint(self, rgb):
        for sh in self.textures:
            inp = sh.GetInput("scale") or sh.CreateInput("scale", Sdf.ValueTypeNames.Float4)
            inp.Set(Gf.Vec4f(rgb[0], rgb[1], rgb[2], 1.0))


cars = [Obj(f"/World/Cars/car_{i:02d}", CAR_USDS[i % len(CAR_USDS)], "car") for i in range(args.max_cars)]
people = [Obj(f"/World/People/person_{i:02d}", PERSON_USDS[i % len(PERSON_USDS)], "pedestrian")
          for i in range(args.max_people)]
simulation_app.update()
for o in cars:
    o.textures = body_textures(o.prim)
posed = sum(posing.relax_arms(stage, o.prim.GetPath().AppendChild("asset"), o.prim.GetPath().AppendChild("pose"),
                              arm_deg=-rng.uniform(72, 86), elbow_deg=-rng.uniform(0, 25)) for o in people)
print(f"[sdg] posed {posed} rigged characters", flush=True)

# Car paint palette (multiplies the body texture): white/silver/grey/black dominate real traffic
PAINT = [(1, 1, 1), (0.85, 0.85, 0.85), (0.55, 0.55, 0.57), (0.18, 0.18, 0.2), (0.08, 0.08, 0.09),
         (0.75, 0.12, 0.1), (0.15, 0.25, 0.6), (0.2, 0.4, 0.25), (0.8, 0.7, 0.45), (0.5, 0.3, 0.2)]

# ---------------------------------------------------------------- camera + annotators
CAM = "/World/UAV/Camera"
cam = UsdGeom.Camera.Define(stage, CAM)
H_AP = 20.955
cam.GetHorizontalApertureAttr().Set(H_AP)
cam.GetVerticalApertureAttr().Set(H_AP * args.height / args.width)
cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.5, 3000.0))
cam_xf = UsdGeom.Xformable(cam)
cam_xf.ClearXformOpOrder()
cam_op = cam_xf.AddTransformOp()

rp = rep.create.render_product(CAM, (args.width, args.height))
rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
bbox_annot = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
rgb_annot.attach([rp])
bbox_annot.attach([rp])
rep.orchestrator.set_capture_on_play(False)

timeline = omni.timeline.get_timeline_interface()
timeline.play()  # starts PhysX so the terrain colliders can be ray cast
for _ in range(10):
    simulation_app.update()
query = get_physics_scene_query_interface()


def ground_hit(origin, direction, dist=5000.0):
    """First hit along the ray if it is park ground (not a tree, roof or prop), else None."""
    n = np.linalg.norm(direction)
    hit, info = query.raycast_closest(tuple(map(float, origin)), tuple(map(float, direction / n)), dist, True)
    if not hit or str(PhysicsSchemaTools.intToSdfPath(info.collision)) not in GROUND:
        return None
    return np.array(info.position, dtype=float)


# Park area with ground under it (bounds of the converted Baylands tiles, minus a margin)
X_RANGE, Y_RANGE = (-330.0, 300.0), (-330.0, 140.0)


def look_basis(yaw, pitch):
    f = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch)])
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return f, r, u


def set_camera(pos, f, r, u):
    m = Gf.Matrix4d(r[0], r[1], r[2], 0, u[0], u[1], u[2], 0, -f[0], -f[1], -f[2], 0, pos[0], pos[1], pos[2], 1)
    cam_op.Set(m)


def pixel_ray(px, py, f, r, u, tan_h, tan_v):
    x = (2.0 * px / args.width - 1.0) * tan_h
    y = (1.0 - 2.0 * py / args.height) * tan_v
    return f + x * r + y * u


def far_enough(p, placed, d):
    return all(np.hypot(p[0] - q[0], p[1] - q[1]) >= d for q in placed)


def randomize_frame():
    # camera
    for _ in range(50):
        focus = ground_hit((rng.uniform(*X_RANGE), rng.uniform(*Y_RANGE), 300.0), np.array([0, 0, -1.0]))
        if focus is not None:
            break
    alt = math.exp(rng.uniform(math.log(8.0), math.log(120.0)))
    pitch = math.radians(rng.uniform(30.0, 89.0))
    yaw = rng.uniform(-math.pi, math.pi)
    hfov = math.radians(rng.uniform(55.0, 95.0))
    f, r, u = look_basis(yaw, pitch)
    pos = focus - f * (alt / math.sin(pitch))
    set_camera(pos, f, r, u)
    cam.GetFocalLengthAttr().Set(H_AP / (2.0 * math.tan(hfov / 2)))
    tan_h = math.tan(hfov / 2)
    tan_v = tan_h * args.height / args.width

    def ground_at_pixel(px, py):
        return ground_hit(pos, pixel_ray(px, py, f, r, u, tan_h, tan_v))

    # cars: scattered or in a parked row
    n_cars = rng.choice([0, 1, 2, 3, 4, 6, 8, 10, 12, 15, args.max_cars])
    n_cars = min(n_cars, args.max_cars)
    placed = []
    car_items = list(cars)
    rng.shuffle(car_items)
    row = rng.random() < 0.35
    row_yaw = rng.uniform(0, 360)
    for o in car_items:
        o.hide()
    for o in car_items[:n_cars]:
        for _ in range(15):
            g = ground_at_pixel(rng.uniform(0, args.width), rng.uniform(0, args.height))
            if g is not None and far_enough(g, placed, 4.5):
                yaw_c = row_yaw + rng.choice([0, 180]) + rng.gauss(0, 4) if row else rng.uniform(0, 360)
                o.place(g, yaw_c, rng.uniform(0.92, 1.08))
                o.tint(rng.choice(PAINT) if rng.random() < 0.85 else [rng.uniform(0.1, 1) for _ in range(3)])
                placed.append(g)
                break

    # people: some in groups of 2..6, the rest alone
    n_people = min(rng.choice([0, 1, 2, 4, 6, 8, 12, 16, 20, 25, args.max_people]), args.max_people)
    ppl = list(people)
    rng.shuffle(ppl)
    for o in ppl:
        o.hide()
    i = 0
    while i < n_people:
        group = rng.choice([1, 1, 1, 2, 3, 4, 6])
        centre = None
        for _ in range(15):
            centre = ground_at_pixel(rng.uniform(0, args.width), rng.uniform(0, args.height))
            if centre is not None:
                break
        if centre is None:
            break
        for _ in range(group):
            if i >= n_people:
                break
            for _ in range(10):
                off = np.array([rng.gauss(0, 1.2), rng.gauss(0, 1.2), 0.0]) if group > 1 else np.zeros(3)
                g = ground_hit(centre + off + [0, 0, 20.0], np.array([0, 0, -1.0]))
                if g is not None and far_enough(g, placed, 0.7):
                    ppl[i].place(g, rng.uniform(0, 360), rng.uniform(0.9, 1.08))
                    placed.append(g)
                    break
            i += 1

    # light
    elev = rng.uniform(8.0, 85.0)
    azim = rng.uniform(0.0, 360.0)
    overcast = rng.random() < 0.25
    sun.GetIntensityAttr().Set(rng.uniform(600, 1500) if overcast else rng.uniform(2000, 5000))
    sun.GetColorTemperatureAttr().Set(rng.uniform(3000, 4500) if elev < 20 else rng.uniform(4500, 7500))
    # DistantLight shines along its -Z; rotate so the light comes from (elev, azim)
    UsdGeom.XformCommonAPI(sun).SetRotate(Gf.Vec3f(90.0 - elev, 0.0, azim), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    dome.GetIntensityAttr().Set(rng.uniform(1200, 2500) if overcast else rng.uniform(300, 1200))
    b = rng.uniform(0.85, 1.0)
    dome.GetColorAttr().Set(Gf.Vec3f(rng.uniform(0.7, 0.9) * b, rng.uniform(0.8, 0.95) * b, b))

    outer_color.Set([Gf.Vec3f(rng.uniform(0.25, 0.42), rng.uniform(0.3, 0.4), rng.uniform(0.18, 0.28))])

    # ground texture tint (season / wet / dry look)
    base = rng.uniform(0.7, 1.15)
    tint = Gf.Vec4f(base * rng.uniform(0.9, 1.1), base * rng.uniform(0.9, 1.1), base * rng.uniform(0.85, 1.1), 1.0)
    for sh in park_textures:
        inp = sh.GetInput("scale") or sh.CreateInput("scale", Sdf.ValueTypeNames.Float4)
        inp.Set(tint)

    return {"alt_m": alt, "pitch_deg": math.degrees(pitch), "yaw_deg": math.degrees(yaw),
            "hfov_deg": math.degrees(hfov), "sun_elev_deg": elev, "sun_azim_deg": azim, "overcast": int(overcast),
            "n_cars": n_cars, "n_people": n_people}


def yolo_lines(bbox_data):
    data, info = bbox_data["data"], bbox_data["info"]
    id_to_label = info.get("idToLabels", {})
    lines = []
    for row in data:
        lab = id_to_label.get(str(row["semanticId"]), id_to_label.get(int(row["semanticId"]), {}))
        name = lab.get("class") if isinstance(lab, dict) else None
        if name not in CLASS_ID:
            continue
        x0, y0, x1, y1 = (float(row[k]) for k in ("x_min", "y_min", "x_max", "y_max"))
        x0, y0 = max(x0, 0.0), max(y0, 0.0)
        x1, y1 = min(x1 + 1, args.width), min(y1 + 1, args.height)
        w, h = x1 - x0, y1 - y0
        if w < 3 or h < 3 or float(row["occlusionRatio"]) > 0.75:
            continue
        lines.append(f"{CLASS_ID[name]} {(x0 + w / 2) / args.width:.6f} {(y0 + h / 2) / args.height:.6f} "
                     f"{w / args.width:.6f} {h / args.height:.6f}")
    return lines


out = os.path.abspath(args.out)
os.makedirs(os.path.join(out, "images"), exist_ok=True)
os.makedirs(os.path.join(out, "labels"), exist_ok=True)
meta_path = os.path.join(out, "meta.csv")
new_meta = not os.path.exists(meta_path)
meta_f = open(meta_path, "a", newline="")
meta = None

# warm up the renderer (first frames come out dark while textures stream in)
randomize_frame()
for _ in range(3):
    rep.orchestrator.step(rt_subframes=8, delta_time=0.0, pause_timeline=False)

t0 = time.time()
for k in range(args.start, args.start + args.frames):
    params = randomize_frame()
    rep.orchestrator.step(rt_subframes=args.rt_subframes, delta_time=0.0, pause_timeline=False)
    rgb = rgb_annot.get_data()
    lines = yolo_lines(bbox_annot.get_data())
    name = f"syn_{k:06d}"
    Image.fromarray(np.asarray(rgb)[:, :, :3]).save(os.path.join(out, "images", name + ".jpg"),
                                                    quality=rng.randint(75, 95))
    with open(os.path.join(out, "labels", name + ".txt"), "w") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))
    params.update(image=name, n_boxes=len(lines))
    if meta is None:
        meta = csv.DictWriter(meta_f, fieldnames=list(params))
        if new_meta:
            meta.writeheader()
    meta.writerow({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in params.items()})
    done = k - args.start + 1
    if done % 50 == 0 or done == args.frames:
        meta_f.flush()
        print(f"[sdg] {done}/{args.frames} frames, {done / (time.time() - t0):.2f} fps", flush=True)

meta_f.close()
timeline.stop()
simulation_app.close()
