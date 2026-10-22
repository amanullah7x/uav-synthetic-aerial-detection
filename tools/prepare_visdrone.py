"""Unzip VisDrone2019-DET train/val and convert the annotations to YOLO labels (same rules as Ultralytics' VisDrone.yaml).

    python tools/prepare_visdrone.py ~/datasets/visdrone_raw ~/datasets/VisDrone

Output: <out>/VisDrone2019-DET-{train,val}/{images,labels}. Ignored regions (class 0) and 'others' (class 11) are dropped,
classes 1..10 become 0..9 (pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor).
"""
import os
import sys
import zipfile

from PIL import Image


def convert(split_dir):
    os.makedirs(os.path.join(split_dir, "labels"), exist_ok=True)
    n = 0
    for fn in sorted(os.listdir(os.path.join(split_dir, "annotations"))):
        stem = os.path.splitext(fn)[0]
        w, h = Image.open(os.path.join(split_dir, "images", stem + ".jpg")).size
        lines = []
        for row in open(os.path.join(split_dir, "annotations", fn)):
            r = row.strip().split(",")
            if len(r) < 6 or r[4] == "0":  # score 0 = ignored region
                continue
            x, y, bw, bh, cls = int(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[5]) - 1
            if not 0 <= cls <= 9 or bw <= 0 or bh <= 0:
                continue
            lines.append(f"{cls} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
        with open(os.path.join(split_dir, "labels", stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        n += 1
    return n


def main():
    raw, out = (os.path.expanduser(p) for p in sys.argv[1:3])
    os.makedirs(out, exist_ok=True)
    for split in ("train", "val"):
        z = os.path.join(raw, f"VisDrone2019-DET-{split}.zip")
        d = os.path.join(out, f"VisDrone2019-DET-{split}")
        if not os.path.isdir(d):
            zipfile.ZipFile(z).extractall(out)
        print(split, convert(d), "images")


if __name__ == "__main__":
    main()
