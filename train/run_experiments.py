"""Train and evaluate the detector variants on the real VisDrone validation set.

    python train/run_experiments.py [--epochs 15] [--only real_ft,real_syn,syn_only]

Variants (all YOLOv8s, 640 px):
  baseline   models/best.pt as delivered (VisDrone fine-tune), evaluated only
  real_ft    best.pt fine-tuned further on VisDrone train            (controls for the extra training)
  real_syn   best.pt fine-tuned on VisDrone train + synthetic frames  (same schedule as real_ft)
  syn_only   COCO yolov8s.pt trained on synthetic frames only         (sim-to-real gap)
Results go to results/metrics.json and results/metrics.csv.
"""
import argparse
import csv
import json
import os

from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.path.expanduser(os.environ.get("BASE_MODEL", "~/drone_px4_isaac/models/best.pt"))
DATA = {k: os.path.join(HERE, "data", f"{v}.yaml") for k, v in
        [("real", "visdrone_real"), ("real_syn", "visdrone_plus_synthetic"), ("syn", "synthetic_only")]}
RUNS = os.path.join(ROOT, "runs")


def evaluate(weights, name):
    m = YOLO(weights)
    r = m.val(data=DATA["real"], imgsz=640, batch=16, project=os.path.join(RUNS, "val"), name=name, exist_ok=True,
              plots=True, verbose=False)
    names = r.names
    per_class = {names[c]: {"ap50": float(r.box.ap50[i]), "ap50_95": float(r.box.ap[i])}
                 for i, c in enumerate(r.box.ap_class_index)}
    return {"model": name, "weights": weights, "map50": float(r.box.map50), "map50_95": float(r.box.map),
            "per_class": per_class}


def train(name, data, init, epochs, lr0):
    m = YOLO(init)
    m.train(data=data, epochs=epochs, imgsz=640, batch=16, workers=8, lr0=lr0, lrf=0.1, warmup_epochs=1,
            close_mosaic=3, project=os.path.join(RUNS, "train"), name=name, exist_ok=True, seed=0, deterministic=False,
            plots=True)
    return os.path.join(RUNS, "train", name, "weights", "best.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--syn-epochs", type=int, default=30)
    ap.add_argument("--only", default="baseline,real_ft,real_syn,syn_only")
    a = ap.parse_args()
    out_json = os.path.join(ROOT, "results", "metrics.json")
    results = json.load(open(out_json)) if os.path.exists(out_json) else {}
    for v in a.only.split(","):
        if v == "baseline":
            w = BASE
        elif v == "real_ft":
            w = train(v, DATA["real"], BASE, a.epochs, 0.002)
        elif v == "real_syn":
            w = train(v, DATA["real_syn"], BASE, a.epochs, 0.002)
        elif v == "syn_only":
            w = train(v, DATA["syn"], "yolov8s.pt", a.syn_epochs, 0.01)
        else:
            raise SystemExit(f"unknown variant {v}")
        results[v] = evaluate(w, v)
        json.dump(results, open(out_json, "w"), indent=2)
        print(f"[exp] {v}: mAP50 {results[v]['map50']:.3f}  mAP50-95 {results[v]['map50_95']:.3f}", flush=True)
    with open(os.path.join(ROOT, "results", "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "mAP50", "mAP50-95", "car AP50", "car AP50-95", "pedestrian AP50", "pedestrian AP50-95"])
        for k, r in results.items():
            pc = r["per_class"]
            w.writerow([k, f"{r['map50']:.3f}", f"{r['map50_95']:.3f}",
                        f"{pc['car']['ap50']:.3f}", f"{pc['car']['ap50_95']:.3f}",
                        f"{pc['pedestrian']['ap50']:.3f}", f"{pc['pedestrian']['ap50_95']:.3f}"])


if __name__ == "__main__":
    main()
