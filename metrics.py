"""Metrics for protocol↔video step alignment.

Two kinds:
  - Ground-truth localization metrics: compare an engine's predicted per-step
    [start,end] against ground-truth [start,end] (e.g. the BioVL-QR protocol CSVs).
  - Cross-engine agreement: how much Claude and the VLM agree on the same video
    (no ground truth needed — usable on any uploaded clip).
"""


def _iou(a0, a1, b0, b1):
    """Temporal intersection-over-union of two [start,end] intervals (seconds)."""
    a0, a1 = (a0, a1) if a1 >= a0 else (a1, a0)
    b0, b1 = (b0, b1) if b1 >= b0 else (b1, b0)
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return (inter / union) if union > 0 else 0.0


def _localized(step):
    """A step is 'localized' if the engine gave a real frame/time (best_frame_index>=0).
    Tolerant of None / float / bool values an engine might emit."""
    bfi = step.get("best_frame_index")
    return isinstance(bfi, (int, float)) and not isinstance(bfi, bool) and bfi >= 0


def score_vs_groundtruth(pred_steps, gt_intervals, tol=(5, 10)):
    """pred_steps: engine output [{step_number,start_time_s,end_time_s,best_frame_index,...}].
    gt_intervals: ground-truth [(start,end), ...] in protocol order (1:1 with steps).
    Steps matched by position. Unlocalized predictions score IoU 0 (counts against the engine).
    Returns a dict of aggregate metrics for this single run."""
    n = min(len(pred_steps), len(gt_intervals))
    if n == 0:
        return {"n_steps": 0, "mean_iou": 0.0, "localized_frac": 0.0,
                **{f"start_within_{t}s": 0.0 for t in tol}, "ordering_acc": 0.0}

    ious, start_err, localized, pred_starts = [], [], 0, []
    for i in range(n):
        ps = pred_steps[i]
        gs, ge = gt_intervals[i]
        if _localized(ps):
            localized += 1
            p0, p1 = float(ps.get("start_time_s", 0)), float(ps.get("end_time_s", 0))
            ious.append(_iou(p0, p1, gs, ge))
            start_err.append(abs(p0 - gs))
            pred_starts.append(p0)
        else:
            ious.append(0.0)
            start_err.append(float("inf"))
            pred_starts.append(None)

    out = {
        "n_steps": n,
        "mean_iou": round(sum(ious) / n, 3),
        "localized_frac": round(localized / n, 3),
    }
    for t in tol:
        out[f"start_within_{t}s"] = round(sum(1 for e in start_err if e <= t) / n, 3)

    # ordering: of consecutive localized pairs, fraction with non-decreasing predicted start
    seq = [s for s in pred_starts if s is not None]
    pairs = list(zip(seq, seq[1:]))
    out["ordering_acc"] = round(sum(1 for a, b in pairs if b >= a) / len(pairs), 3) if pairs else 1.0
    return out


def aggregate(runs, keys=("mean_iou", "start_within_5s", "start_within_10s",
                          "localized_frac", "ordering_acc", "latency_s", "n_flags")):
    """Average a list of per-run metric dicts. Missing keys are skipped per-run."""
    out = {}
    for k in keys:
        vals = [r[k] for r in runs if isinstance(r.get(k), (int, float))]
        out[k] = round(sum(vals) / len(vals), 3) if vals else None
    out["n_videos"] = len(runs)
    return out


def cross_engine_agreement(a_steps, b_steps):
    """How much two engines agree on the same video (no ground truth).
    Returns mean per-step IoU between their predicted ranges, flag agreement, and
    mean absolute start-time difference (over steps both engines localized)."""
    n = min(len(a_steps), len(b_steps))
    if n == 0:
        return {"n_steps": 0, "mean_iou": 0.0, "flag_agreement": 0.0, "mean_start_diff_s": None}
    ious, flag_agree, start_diffs = [], 0, []
    for i in range(n):
        a, b = a_steps[i], b_steps[i]
        if _localized(a) and _localized(b):
            ious.append(_iou(float(a.get("start_time_s", 0)), float(a.get("end_time_s", 0)),
                             float(b.get("start_time_s", 0)), float(b.get("end_time_s", 0))))
            start_diffs.append(abs(float(a.get("start_time_s", 0)) - float(b.get("start_time_s", 0))))
        flag_agree += 1 if bool(a.get("flag")) == bool(b.get("flag")) else 0
    return {
        "n_steps": n,
        "mean_iou": round(sum(ious) / len(ious), 3) if ious else 0.0,
        "flag_agreement": round(flag_agree / n, 3),
        "mean_start_diff_s": round(sum(start_diffs) / len(start_diffs), 1) if start_diffs else None,
    }
