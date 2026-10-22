"""README figures: per-class AP50 bar chart and synthetic-only predictions on real VisDrone images.

    python tools/make_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "images")
VAL = os.path.expanduser("~/datasets/VisDrone/VisDrone2019-DET-val/images")

m = json.load(open(os.path.join(ROOT, "results", "metrics.json")))
order = ["baseline", "real_ft", "real_syn", "syn_only"]
labels = ["baseline\n(best.pt)", "real only\n+15 epochs", "real + synthetic\n+15 epochs", "synthetic only\n(COCO start)"]
fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
w = 0.38
for j, (cls, col) in enumerate([("car", "#2f6db5"), ("pedestrian", "#d0573f")]):
    vals = [m[k]["per_class"][cls]["ap50"] for k in order]
    xs = [i + (j - 0.5) * w for i in range(len(order))]
    ax.bar(xs, vals, w, label=cls, color=col)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
ax.set_xticks(range(len(order)), labels, fontsize=8)
ax.set_ylabel("AP50 on VisDrone val (real)")
ax.set_ylim(0, 0.9)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8)
ax.set_title("Synthetic data matches real-only fine-tuning; synthetic alone transfers partly", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "ap50_by_model.png"))

# synthetic-only model on real images
model = YOLO(os.path.join(ROOT, "runs", "train", "syn_only", "weights", "best.pt"))
names = ["0000271_00601_d_0000371.jpg", "0000330_00001_d_0000809.jpg"]
names = [n for n in names if os.path.exists(os.path.join(VAL, n))] or sorted(os.listdir(VAL))[100:102]
tiles = []
for n in names:
    r = model.predict(os.path.join(VAL, n), conf=0.3, imgsz=640, verbose=False)[0]
    tiles.append(Image.fromarray(r.plot(line_width=2, font_size=10)[:, :, ::-1]).resize((800, 450)))
sheet = Image.new("RGB", (800 * len(tiles), 450))
for i, t in enumerate(tiles):
    sheet.paste(t, (800 * i, 0))
sheet.save(os.path.join(OUT, "syn_only_on_real.jpg"), quality=88)
print("wrote", os.listdir(OUT))
