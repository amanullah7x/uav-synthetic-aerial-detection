"""Draw YOLO labels on images and tile them into one preview: python tools/draw_labels.py <dataset dir> <out.jpg> [n]"""
import os
import sys

from PIL import Image, ImageDraw

COLORS = {0: (255, 60, 60), 3: (60, 200, 255)}


def main():
    root, out = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    names = sorted(os.listdir(os.path.join(root, "images")))[:n]
    tiles = []
    for name in names:
        im = Image.open(os.path.join(root, "images", name)).convert("RGB")
        w, h = im.size
        d = ImageDraw.Draw(im)
        lab = os.path.join(root, "labels", os.path.splitext(name)[0] + ".txt")
        for line in open(lab) if os.path.exists(lab) else []:
            c, x, y, bw, bh = line.split()
            c, x, y, bw, bh = int(c), float(x) * w, float(y) * h, float(bw) * w, float(bh) * h
            d.rectangle([x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2], outline=COLORS.get(c, (255, 255, 0)), width=2)
        tiles.append(im.resize((640, int(640 * h / w))))
    cols = 2
    th = tiles[0].height
    sheet = Image.new("RGB", (640 * cols, th * ((len(tiles) + cols - 1) // cols)))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * 640, (i // cols) * th))
    sheet.save(out, quality=90)


if __name__ == "__main__":
    main()
