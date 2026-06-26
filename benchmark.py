#!/usr/bin/env python
"""Benchmark Claude vs the local VLM on BioVL-QR step localization.

For each video, the protocol CSV gives both the step sentences (input) and the
ground-truth [start,end] of each step (held out). We sample frames, run each
engine, and score predicted vs ground-truth time-ranges (temporal IoU, ±Ns start
accuracy, ordering). Results are written to benchmark_results.json, which the web
app's /benchmark page renders as a leaderboard.

Usage:
    ANTHROPIC_API_KEY_FILE=/path/to/anthropic_key.txt \
    python benchmark.py --data /home/jovyan/workbench/BioVL-QR_zip \
        --categories extractdna gel electrophoresis purifydna --per-cat 1 --frames 24
"""
import argparse
import csv
import json
import os
import time

import app          # reuse sample_frames, align (Claude), MODEL, VLM info
import metrics

try:
    import vlm_engine
except Exception:
    vlm_engine = None


def load_protocol_csv(path):
    """Return (steps, gt_intervals): step sentences + [(start,end),...] from a BioVL-QR CSV."""
    steps, gt = [], []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        for row in reader:
            if not row or not row[0].strip():
                continue
            try:
                s, e = float(row[1]), float(row[2])
            except (IndexError, ValueError):
                s, e = 0.0, 0.0
            steps.append(row[0].strip())
            gt.append((s, e))
    return steps, gt


def run_one(engine, steps, frames):
    t0 = time.time()
    if engine == "claude":
        res = app.align(steps, frames)
    else:
        res = vlm_engine.align_vlm(steps, frames)
    res["_latency_s"] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/jovyan/workbench/BioVL-QR_zip")
    ap.add_argument("--categories", nargs="+",
                    default=["extractdna", "gel", "electrophoresis", "purifydna"])
    ap.add_argument("--per-cat", type=int, default=1, help="videos per category")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--engines", nargs="+", default=["claude", "vlm"])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "benchmark_results.json"))
    args = ap.parse_args()

    runs = []
    for cat in args.categories:
        vdir = os.path.join(args.data, "videos", cat)
        pdir = os.path.join(args.data, "protocols", cat)
        if not os.path.isdir(vdir):
            print(f"!! skip {cat}: no {vdir}", flush=True)
            continue
        vids = sorted(f for f in os.listdir(vdir) if f.endswith(".mp4"))[:args.per_cat]
        for vf in vids:
            stem = os.path.splitext(vf)[0]
            csv_path = os.path.join(pdir, stem + ".csv")
            if not os.path.isfile(csv_path):
                print(f"!! skip {stem}: no protocol csv", flush=True)
                continue
            steps, gt = load_protocol_csv(csv_path)
            print(f"\n== {cat}/{stem}: {len(steps)} steps ==", flush=True)
            t0 = time.time()
            frames, total, dur, fps = app.sample_frames(os.path.join(vdir, vf), args.frames)
            print(f"   sampled {len(frames)} frames ({dur:.0f}s clip) in {time.time()-t0:.1f}s", flush=True)
            for eng in args.engines:
                if eng == "vlm" and vlm_engine is None:
                    continue
                try:
                    res = run_one(eng, steps, frames)
                    m = metrics.score_vs_groundtruth(res["steps"], gt)
                    perstep = [{"gt": list(gt[i]),
                                "pred": [res["steps"][i].get("start_time_s", 0),
                                         res["steps"][i].get("end_time_s", 0)]
                                        if res["steps"][i].get("best_frame_index", -1) >= 0 else None}
                               for i in range(min(len(gt), len(res["steps"])))]
                    runs.append({"category": cat, "video": stem, "engine": eng,
                                 "latency_s": res["_latency_s"], "n_flags": res.get("n_flags", 0),
                                 **m, "perstep": perstep})
                    print(f"   [{eng:6}] IoU={m['mean_iou']:.3f}  ±10s={m['start_within_10s']:.2f}  "
                          f"order={m['ordering_acc']:.2f}  {res['_latency_s']}s", flush=True)
                except Exception as e:
                    print(f"   [{eng:6}] ERROR: {e}", flush=True)

    # aggregate: overall per engine + per (engine, category)
    agg = {}
    by_cat = {}
    for eng in args.engines:
        er = [r for r in runs if r["engine"] == eng]
        if er:
            agg[eng] = metrics.aggregate(er)
        for cat in args.categories:
            cr = [r for r in er if r["category"] == cat]
            if cr:
                by_cat.setdefault(eng, {})[cat] = metrics.aggregate(cr)

    out = {"config": {"data": args.data, "categories": args.categories,
                      "per_cat": args.per_cat, "frames": args.frames,
                      "claude_model": app.MODEL, "vlm_model": app.VLM_LABEL},
           "runs": runs, "aggregate": agg, "by_category": by_cat}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {args.out}  ({len(runs)} runs)", flush=True)
    for eng, a in agg.items():
        print(f"  {eng:6}: meanIoU={a['mean_iou']}  ±10s={a['start_within_10s']}  "
              f"localized={a['localized_frac']}  lat={a['latency_s']}s  (n={a['n_videos']})", flush=True)


if __name__ == "__main__":
    main()
