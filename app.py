#!/usr/bin/env python
"""
Protocol → video step alignment.

Paste a protocol (one step per line) and upload a video of someone performing it.
The app samples timestamped frames, sends them (in order, each labeled with its
time) plus the protocol to Claude, and Claude returns, for each step, the
time-range and the single most representative frame where that step happens.

Claude's API takes images, not video — so frames are the unit of alignment.

API key: put it (one line) in  video_app/anthropic_key.txt  — read automatically.
($ANTHROPIC_API_KEY, $ANTHROPIC_API_KEY_FILE, .anthropic_key, .env also honored.)

Run:
    pip install flask anthropic opencv-python
    echo "sk-ant-..." > anthropic_key.txt && chmod 600 anthropic_key.txt
    python app.py            # http://0.0.0.0:7860
    # laptop:  ssh -L 7860:localhost:7860 <host>  -> open localhost:7860
"""
import base64
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import anthropic
from flask import Flask, request, jsonify, render_template_string

MODEL = "claude-opus-4-8"
MAX_FRAMES = 30
FRAME_WIDTH = 1280       # higher res so small tube/reagent labels are legible
PORT = int(os.environ.get("PORT", "7860"))

app = Flask(__name__)
HERE = os.path.dirname(os.path.abspath(__file__))
KEY_FILES = [os.environ.get("ANTHROPIC_API_KEY_FILE"),
             os.path.join(HERE, "anthropic_key.txt"),
             os.path.join(HERE, ".anthropic_key"),
             os.path.join(HERE, ".env")]


def load_api_key():
    env = os.environ.get("ANTHROPIC_API_KEY")
    if env and env.strip():
        return env.strip()
    for path in KEY_FILES:
        if not path or not os.path.isfile(path):
            continue
        for line in open(path):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                if k.strip().upper().endswith("ANTHROPIC_API_KEY"):
                    return v.strip().strip('"').strip("'")
            else:
                return line.strip('"').strip("'")
    return None


_client = None
def get_client():
    global _client
    if _client is None:
        key = load_api_key()
        if not key:
            raise RuntimeError(
                "No Anthropic API key found. Put your key in "
                f"{os.path.join(HERE, 'anthropic_key.txt')} (one line), "
                "or set ANTHROPIC_API_KEY.")
        _client = anthropic.Anthropic(api_key=key)
    return _client


def parse_steps(text):
    """One step per non-empty line. Returns list of step strings (numbering stripped)."""
    steps = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        # strip a leading "1." / "1)" / "- " marker if present
        import re
        s = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", s).strip()
        if s:
            steps.append(s)
    return steps


def sample_frames(path, n):
    """Return (frames, total, duration, fps) where frames=[{idx,t,b64}] chronological."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open this video (try an .mp4/.mov file).")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    n = max(1, min(n, MAX_FRAMES))
    raw_idxs = [0] if total <= 1 else (
        [round(i * (total - 1) / (n - 1)) for i in range(n)] if n > 1 else [0])

    frames = []
    for fi in raw_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        big = cv2.resize(frame, (FRAME_WIDTH, int(h * FRAME_WIDTH / w))) if w > FRAME_WIDTH else frame
        okb, buf = cv2.imencode(".jpg", big, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        bh, bw = big.shape[:2]
        TW = 420
        small = cv2.resize(big, (TW, int(bh * TW / bw))) if bw > TW else big
        okt, tbuf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if okb:
            frames.append({"idx": len(frames),
                           "t": round(fi / fps, 1) if fps else 0.0,
                           "b64": base64.b64encode(buf).decode(),                       # 1280px → model
                           "thumb": base64.b64encode(tbuf).decode() if okt else ""})     # 420px → browser
    cap.release()
    if not frames:
        raise ValueError("Could not read any frames from this video.")
    duration = (total / fps) if fps else 0
    return frames, total, duration, fps


ALIGN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "observed_summary": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "step_number": {"type": "integer"},
                    "step_text": {"type": "string"},
                    "level": {"type": "string", "enum": ["green", "yellow", "red"]},
                    "start_time_s": {"type": "number"},
                    "end_time_s": {"type": "number"},
                    "best_frame_index": {"type": "integer"},
                    "note": {"type": "string"},
                },
                "required": ["step_number", "step_text", "level",
                             "start_time_s", "end_time_s", "best_frame_index", "note"],
            },
        },
    },
    "required": ["observed_summary", "steps"],
}

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday"]
# 0 = Monday (strictest) … 4 = Friday (most relaxed). Only the green↔yellow↔red
# threshold changes; the "don't penalize unverifiable amounts" rule is fixed.
STRICTNESS = {
    0: ("CALIBRATION — STRICT AUDIT (Monday): Be a demanding auditor. Mark 'green' ONLY when you "
        "can clearly see the action happen. If the action is not clearly visible, mark 'yellow'; if "
        "a step's action is not clearly visible where you'd expect it in the sequence, lean 'red'. "
        "Scrutinize order and completeness."),
    1: ("CALIBRATION — fairly strict (Tuesday): Prefer 'yellow' over 'green' when the evidence is "
        "only partial. Mark 'green' only when the action is reasonably clear; flag anything you're "
        "unsure about."),
    2: ("CALIBRATION — balanced (Wednesday): Mark 'green' when the action is visible or clearly "
        "plausible, 'yellow' when genuinely unsure, 'red' when an expected action is clearly absent."),
    3: ("CALIBRATION — lenient (Thursday): Give the benefit of the doubt. Mark 'green' when the "
        "action plausibly happened; use 'yellow' only for real doubt and 'red' only for clear omissions."),
    4: ("CALIBRATION — relaxed (Friday): Assume good faith. Mark 'green' whenever the action plausibly "
        "happened even if not fully confirmable; reserve 'yellow' for genuine uncertainty and 'red' only "
        "for obvious, unmistakable omissions. Don't nitpick."),
}


def align(steps, frames, strictness=4):
    """Advisory traffic-light check at a given strictness (0=Mon strict … 4=Fri relaxed)."""
    proto = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    header = (
        "You are assisting a human reviewing whether a lab protocol was followed in a video. "
        "You give a traffic-light assessment per step, not a pass/fail verdict.\n\n"
        "PROTOCOL:\n" + proto + "\n\n"
        f"Below are {len(frames)} frames sampled in chronological order from the video, each "
        "preceded by its index and timestamp. Treat them as one continuous clip."
    )
    instruction = (
        "DO NOT penalize what video can't show. You cannot read exact quantities/volumes/"
        "concentrations (µL, mL, ng) from frames, and you often can't confirm which specific labeled "
        "tube a pipette drew from. NEVER downgrade a step for an unverifiable amount or reagent "
        "identity. Judge each step by whether its ACTION (adding a reagent, mixing, spinning, placing "
        "the tube, etc.) appears to take place.\n\n"
        + STRICTNESS.get(strictness, STRICTNESS[4]) + "\n\n"
        "First, in observed_summary, describe what you actually see (tools/tubes/actions, order), "
        "independently of the protocol.\n\n"
        "Then assign each protocol step a traffic-light level:\n"
        "- 'green': done, or probably done — the action is seen or clearly plausible.\n"
        "- 'yellow': genuinely doubtful — can't tell if it happened, or it looks slightly off / out of order.\n"
        "- 'red': the step appears COMPLETELY MISSED — the action is absent where expected.\n"
        "Apply the calibration above to decide how readily to use yellow/red vs green.\n\n"
        "Also give start_time_s / end_time_s / best_frame_index when the step is visible (else 0 and "
        "-1), and note: one short, plain, non-accusatory sentence. Return every step."
    )

    content = [{"type": "text", "text": header}]
    for f in frames:
        content.append({"type": "text", "text": f"[frame {f['idx']} | t={f['t']}s]"})
        content.append({"type": "image",
                         "source": {"type": "base64", "media_type": "image/jpeg", "data": f["b64"]}})
    content.append({"type": "text", "text": instruction})

    resp = get_client().messages.create(
        model=MODEL,
        max_tokens=12000,
        thinking={"type": "adaptive"},
        output_config={"format": {"type": "json_schema", "schema": ALIGN_SCHEMA}},
        messages=[{"role": "user", "content": content}],
    )
    text = next((b.text for b in resp.content if b.type == "text"), "")
    data = json.loads(text)
    steps_out = data.get("steps", [])
    levels = [s.get("level") for s in steps_out]
    return {"observed_summary": data.get("observed_summary", ""),
            "n_red": levels.count("red"),
            "n_yellow": levels.count("yellow"),
            "n_green": levels.count("green"),
            "steps": steps_out}


BENCHMARK_DATA = {"config":{"data":"/home/jovyan/workbench/BioVL-QR_zip","categories":["extractdna","gel","electrophoresis","purifydna"],"per_cat":1,"frames":24,"claude_model":"claude-opus-4-8","vlm_model":"Qwen2.5-VL-7B-Instruct (4-bit)  &  Qwen2.5-VL-3B-Instruct","engines":["claude","Qwen2.5-VL-7B-Instruct (4-bit)","Qwen2.5-VL-3B-Instruct"]},"runs":[{"category":"extractdna","video":"extractdna_1","engine":"claude","latency_s":38.7,"n_flags":1,"n_steps":5,"mean_iou":0.485,"localized_frac":1.0,"start_within_5s":0.4,"start_within_10s":0.6,"ordering_acc":0.75,"perstep":[{"gt":[1.0,37.0],"pred":[13.1,39.2]},{"gt":[40.0,47.0],"pred":[78.4,85.0]},{"gt":[55.0,67.0],"pred":[58.8,71.9]},{"gt":[77.0,113.0],"pred":[85.0,104.6]},{"gt":[122.0,146.0],"pred":[124.2,150.3]}]},{"category":"extractdna","video":"extractdna_1","engine":"Qwen2.5-VL-3B-Instruct","latency_s":36.1,"n_flags":0,"n_steps":5,"mean_iou":0.372,"localized_frac":1.0,"start_within_5s":0.0,"start_within_10s":0.4,"ordering_acc":1.0,"perstep":[{"gt":[1.0,37.0],"pred":[12.0,31.5]},{"gt":[40.0,47.0],"pred":[32.7,52.3]},{"gt":[55.0,67.0],"pred":[65.4,85.0]},{"gt":[77.0,113.0],"pred":[98.0,117.6]},{"gt":[122.0,146.0],"pred":[130.7,150.3]}]},{"category":"gel","video":"gel_1","engine":"claude","latency_s":24.3,"n_flags":3,"n_steps":6,"mean_iou":0.356,"localized_frac":1.0,"start_within_5s":0.167,"start_within_10s":0.333,"ordering_acc":1.0,"perstep":[{"gt":[2.0,32.0],"pred":[22.7,22.7]},{"gt":[33.0,48.0],"pred":[45.3,45.3]},{"gt":[52.0,188.0],"pred":[45.3,158.7]},{"gt":[203.0,221.0],"pred":[181.4,204.1]},{"gt":[229.0,458.0],"pred":[226.8,453.5]},{"gt":[460.0,518.0],"pred":[476.2,498.8]}]},{"category":"gel","video":"gel_1","engine":"Qwen2.5-VL-3B-Instruct","latency_s":111.4,"n_flags":0,"n_steps":6,"mean_iou":0.0,"localized_frac":0.0,"start_within_5s":0.0,"start_within_10s":0.0,"ordering_acc":1.0,"perstep":[{"gt":[2.0,32.0],"pred":None},{"gt":[33.0,48.0],"pred":None},{"gt":[52.0,188.0],"pred":None},{"gt":[203.0,221.0],"pred":None},{"gt":[229.0,458.0],"pred":None},{"gt":[460.0,518.0],"pred":None}]},{"category":"electrophoresis","video":"electrophoresis_1","engine":"claude","latency_s":27.8,"n_flags":3,"n_steps":7,"mean_iou":0.156,"localized_frac":1.0,"start_within_5s":0.143,"start_within_10s":0.286,"ordering_acc":0.833,"perstep":[{"gt":[3.0,10.0],"pred":[63.0,84.0]},{"gt":[17.0,79.0],"pred":[21.0,42.0]},{"gt":[93.0,236.0],"pred":[273.0,315.0]},{"gt":[244.0,356.0],"pred":[336.0,336.1]},{"gt":[357.0,396.0],"pred":[378.1,420.1]},{"gt":[400.0,408.0],"pred":[420.1,441.1]},{"gt":[436.0,481.0],"pred":[441.1,462.1]}]},{"category":"electrophoresis","video":"electrophoresis_1","engine":"Qwen2.5-VL-3B-Instruct","latency_s":42.8,"n_flags":0,"n_steps":7,"mean_iou":0.114,"localized_frac":1.0,"start_within_5s":0.286,"start_within_10s":0.286,"ordering_acc":1.0,"perstep":[{"gt":[3.0,10.0],"pred":[0.0,12.0]},{"gt":[17.0,79.0],"pred":[12.0,31.5]},{"gt":[93.0,236.0],"pred":[31.5,42.0]},{"gt":[244.0,356.0],"pred":[42.0,105.0]},{"gt":[357.0,396.0],"pred":[105.0,210.0]},{"gt":[400.0,408.0],"pred":[210.0,252.1]},{"gt":[436.0,481.0],"pred":[252.1,420.1]}]},{"category":"purifydna","video":"purifydna_1","engine":"claude","latency_s":41.6,"n_flags":3,"n_steps":19,"mean_iou":0.14,"localized_frac":1.0,"start_within_5s":0.211,"start_within_10s":0.368,"ordering_acc":0.833,"perstep":[{"gt":[5.0,39.0],"pred":[19.0,56.9]},{"gt":[42.0,50.0],"pred":[37.9,75.8]},{"gt":[60.0,91.0],"pred":[151.7,170.7]},{"gt":[92.0,102.0],"pred":[94.8,113.8]},{"gt":[104.0,111.0],"pred":[113.8,113.8]},{"gt":[112.0,118.0],"pred":[113.8,132.7]},{"gt":[123.0,131.0],"pred":[170.7,208.6]},{"gt":[134.0,164.0],"pred":[208.6,246.5]},{"gt":[174.0,200.0],"pred":[265.5,303.4]},{"gt":[209.0,214.0],"pred":[303.4,322.4]},{"gt":[215.0,218.0],"pred":[303.4,322.4]},{"gt":[220.0,257.0],"pred":[246.5,417.2]},{"gt":[267.0,282.0],"pred":[322.4,341.3]},{"gt":[299.0,304.0],"pred":[341.3,360.3]},{"gt":[305.0,307.0],"pred":[341.3,360.3]},{"gt":[314.0,335.0],"pred":[322.4,360.3]},{"gt":[359.0,368.0],"pred":[360.3,398.2]},{"gt":[368.0,403.0],"pred":[379.3,417.2]},{"gt":[423.0,433.0],"pred":[417.2,436.2]}]},{"category":"purifydna","video":"purifydna_1","engine":"Qwen2.5-VL-3B-Instruct","latency_s":110.5,"n_flags":0,"n_steps":19,"mean_iou":0.0,"localized_frac":1.0,"start_within_5s":0.053,"start_within_10s":0.053,"ordering_acc":1.0,"perstep":[{"gt":[5.0,39.0],"pred":[0.0,5.0]},{"gt":[42.0,50.0],"pred":[5.0,10.0]},{"gt":[60.0,91.0],"pred":[10.0,15.0]},{"gt":[92.0,102.0],"pred":[15.0,20.0]},{"gt":[104.0,111.0],"pred":[20.0,25.0]},{"gt":[112.0,118.0],"pred":[25.0,30.0]},{"gt":[123.0,131.0],"pred":[30.0,35.0]},{"gt":[134.0,164.0],"pred":[35.0,40.0]},{"gt":[174.0,200.0],"pred":[40.0,45.0]},{"gt":[209.0,214.0],"pred":[45.0,50.0]},{"gt":[215.0,218.0],"pred":[50.0,55.0]},{"gt":[220.0,257.0],"pred":[55.0,60.0]},{"gt":[267.0,282.0],"pred":[60.0,65.0]},{"gt":[299.0,304.0],"pred":[65.0,70.0]},{"gt":[305.0,307.0],"pred":[70.0,75.0]},{"gt":[314.0,335.0],"pred":[75.0,80.0]},{"gt":[359.0,368.0],"pred":[80.0,85.0]},{"gt":[368.0,403.0],"pred":[85.0,90.0]},{"gt":[423.0,433.0],"pred":[90.0,95.0]}]},{"category":"extractdna","video":"extractdna_1","engine":"Qwen2.5-VL-7B-Instruct (4-bit)","latency_s":64.1,"n_flags":5,"n_steps":5,"mean_iou":0.2,"localized_frac":1.0,"start_within_5s":0.2,"start_within_10s":0.2,"ordering_acc":1.0,"perstep":[{"gt":[1.0,37.0],"pred":[0.0,19.6]},{"gt":[40.0,47.0],"pred":[19.6,32.7]},{"gt":[55.0,67.0],"pred":[32.7,52.3]},{"gt":[77.0,113.0],"pred":[52.3,85.0]},{"gt":[122.0,146.0],"pred":[85.0,150.3]}]},{"category":"gel","video":"gel_1","engine":"Qwen2.5-VL-7B-Instruct (4-bit)","latency_s":56.2,"n_flags":6,"n_steps":6,"mean_iou":0.0,"localized_frac":0.0,"start_within_5s":0.0,"start_within_10s":0.0,"ordering_acc":1.0,"perstep":[{"gt":[2.0,32.0],"pred":None},{"gt":[33.0,48.0],"pred":None},{"gt":[52.0,188.0],"pred":None},{"gt":[203.0,221.0],"pred":None},{"gt":[229.0,458.0],"pred":None},{"gt":[460.0,518.0],"pred":None}]},{"category":"electrophoresis","video":"electrophoresis_1","engine":"Qwen2.5-VL-7B-Instruct (4-bit)","latency_s":76.4,"n_flags":7,"n_steps":7,"mean_iou":0.126,"localized_frac":1.0,"start_within_5s":0.143,"start_within_10s":0.143,"ordering_acc":1.0,"perstep":[{"gt":[3.0,10.0],"pred":[0.0,42.0]},{"gt":[17.0,79.0],"pred":[42.0,105.0]},{"gt":[93.0,236.0],"pred":[105.0,147.0]},{"gt":[244.0,356.0],"pred":[147.0,210.0]},{"gt":[357.0,396.0],"pred":[210.0,252.1]},{"gt":[400.0,408.0],"pred":[252.1,315.0]},{"gt":[436.0,481.0],"pred":[315.0,357.1]}]},{"category":"purifydna","video":"purifydna_1","engine":"Qwen2.5-VL-7B-Instruct (4-bit)","latency_s":167.1,"n_flags":0,"n_steps":19,"mean_iou":0.0,"localized_frac":0.0,"start_within_5s":0.0,"start_within_10s":0.0,"ordering_acc":1.0,"perstep":[{"gt":[5.0,39.0],"pred":None},{"gt":[42.0,50.0],"pred":None},{"gt":[60.0,91.0],"pred":None},{"gt":[92.0,102.0],"pred":None},{"gt":[104.0,111.0],"pred":None},{"gt":[112.0,118.0],"pred":None},{"gt":[123.0,131.0],"pred":None},{"gt":[134.0,164.0],"pred":None},{"gt":[174.0,200.0],"pred":None},{"gt":[209.0,214.0],"pred":None},{"gt":[215.0,218.0],"pred":None},{"gt":[220.0,257.0],"pred":None},{"gt":[267.0,282.0],"pred":None},{"gt":[299.0,304.0],"pred":None},{"gt":[305.0,307.0],"pred":None},{"gt":[314.0,335.0],"pred":None},{"gt":[359.0,368.0],"pred":None},{"gt":[368.0,403.0],"pred":None},{"gt":[423.0,433.0],"pred":None}]}],"aggregate":{"claude":{"mean_iou":0.284,"start_within_5s":0.23,"start_within_10s":0.397,"localized_frac":1.0,"ordering_acc":0.854,"latency_s":33.1,"n_flags":2.5,"n_videos":4},"Qwen2.5-VL-7B-Instruct (4-bit)":{"mean_iou":0.082,"start_within_5s":0.086,"start_within_10s":0.086,"localized_frac":0.5,"ordering_acc":1.0,"latency_s":90.95,"n_flags":4.5,"n_videos":4},"Qwen2.5-VL-3B-Instruct":{"mean_iou":0.121,"start_within_5s":0.085,"start_within_10s":0.185,"localized_frac":0.75,"ordering_acc":1.0,"latency_s":75.2,"n_flags":0.0,"n_videos":4}},"by_category":{"claude":{"extractdna":{"mean_iou":0.485,"start_within_5s":0.4,"start_within_10s":0.6,"localized_frac":1.0,"ordering_acc":0.75,"latency_s":38.7,"n_flags":1.0,"n_videos":1},"gel":{"mean_iou":0.356,"start_within_5s":0.167,"start_within_10s":0.333,"localized_frac":1.0,"ordering_acc":1.0,"latency_s":24.3,"n_flags":3.0,"n_videos":1},"electrophoresis":{"mean_iou":0.156,"start_within_5s":0.143,"start_within_10s":0.286,"localized_frac":1.0,"ordering_acc":0.833,"latency_s":27.8,"n_flags":3.0,"n_videos":1},"purifydna":{"mean_iou":0.14,"start_within_5s":0.211,"start_within_10s":0.368,"localized_frac":1.0,"ordering_acc":0.833,"latency_s":41.6,"n_flags":3.0,"n_videos":1}},"Qwen2.5-VL-7B-Instruct (4-bit)":{"extractdna":{"mean_iou":0.2,"start_within_5s":0.2,"start_within_10s":0.2,"localized_frac":1.0,"ordering_acc":1.0,"latency_s":64.1,"n_flags":5.0,"n_videos":1},"gel":{"mean_iou":0.0,"start_within_5s":0.0,"start_within_10s":0.0,"localized_frac":0.0,"ordering_acc":1.0,"latency_s":56.2,"n_flags":6.0,"n_videos":1},"electrophoresis":{"mean_iou":0.126,"start_within_5s":0.143,"start_within_10s":0.143,"localized_frac":1.0,"ordering_acc":1.0,"latency_s":76.4,"n_flags":7.0,"n_videos":1},"purifydna":{"mean_iou":0.0,"start_within_5s":0.0,"start_within_10s":0.0,"localized_frac":0.0,"ordering_acc":1.0,"latency_s":167.1,"n_flags":0.0,"n_videos":1}},"Qwen2.5-VL-3B-Instruct":{"extractdna":{"mean_iou":0.372,"start_within_5s":0.0,"start_within_10s":0.4,"localized_frac":1.0,"ordering_acc":1.0,"latency_s":36.1,"n_flags":0.0,"n_videos":1},"gel":{"mean_iou":0.0,"start_within_5s":0.0,"start_within_10s":0.0,"localized_frac":0.0,"ordering_acc":1.0,"latency_s":111.4,"n_flags":0.0,"n_videos":1},"electrophoresis":{"mean_iou":0.114,"start_within_5s":0.286,"start_within_10s":0.286,"localized_frac":1.0,"ordering_acc":1.0,"latency_s":42.8,"n_flags":0.0,"n_videos":1},"purifydna":{"mean_iou":0.0,"start_within_5s":0.053,"start_within_10s":0.053,"localized_frac":1.0,"ordering_acc":1.0,"latency_s":110.5,"n_flags":0.0,"n_videos":1}}}}


@app.route("/")
def index():
    return render_template_string(PAGE, model=MODEL, max_frames=MAX_FRAMES)


@app.route("/benchmark-data")
def benchmark_data():
    return jsonify(BENCHMARK_DATA)


@app.route("/align", methods=["POST"])
def align_route():
    pfiles = [p for p in request.files.getlist("protocol") if p and p.filename]
    vfiles = [v for v in request.files.getlist("videos") if v and v.filename]
    if not pfiles:
        return jsonify(error="Attach a protocol text file (.txt, one step per line)."), 400
    proto_text = ""
    for p in pfiles:
        try:
            proto_text += p.read().decode("utf-8", "replace") + "\n"
        except Exception:
            pass
    steps = parse_steps(proto_text)
    if not steps:
        return jsonify(error="Couldn't read any steps from the protocol file(s)."), 400
    if not vfiles:
        return jsonify(error="Attach at least one video."), 400
    try:
        n = int(request.form.get("frames", 20))
    except ValueError:
        n = 20

    # 1) sample each video's frames once (local, fast)
    vids = []
    for v in vfiles:
        suffix = os.path.splitext(v.filename)[1] or ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            v.save(tmp.name)
            tmp.close()
            frames, total, duration, fps = sample_frames(tmp.name, n)
            vids.append({"filename": v.filename, "frames": frames, "duration": round(duration, 1)})
        except Exception as e:
            vids.append({"filename": v.filename, "error": str(e)})
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    # 2) run all (video × strictness-day) inferences in parallel
    jobs = [(vi, di) for vi, vd in enumerate(vids) if "error" not in vd for di in range(5)]
    grid = {}
    if jobs:
        with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as ex:
            futs = {ex.submit(align, steps, vids[vi]["frames"], di): (vi, di) for vi, di in jobs}
            for fut in as_completed(futs):
                vi, di = futs[fut]
                try:
                    grid[(vi, di)] = fut.result()
                except anthropic.APIError as e:
                    grid[(vi, di)] = {"error": f"Claude API error: {e}"}
                except json.JSONDecodeError:
                    grid[(vi, di)] = {"error": "Claude returned an unparseable result (try fewer frames)."}
                except Exception as e:
                    grid[(vi, di)] = {"error": str(e)}

    # 3) assemble per-video results: 5 days + the shared frame thumbnails
    results = []
    for vi, vd in enumerate(vids):
        if "error" in vd:
            results.append({"filename": vd["filename"], "error": vd["error"]})
            continue
        results.append({
            "filename": vd["filename"],
            "n_frames": len(vd["frames"]),
            "duration": vd["duration"],
            "thumbs": [f["thumb"] for f in vd["frames"]],
            "days": {DAYS[di]: grid.get((vi, di), {"error": "no result"}) for di in range(5)},
        })
    return jsonify(steps_parsed=steps, results=results)


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Protocol ↔ Video Alignment</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:920px;
      margin:36px auto;padding:0 20px;color:#17323a;background:#f7fafb}
 h1{font-size:23px;margin-bottom:2px} .sub{color:#5c7079;margin-top:0;font-size:14px}
 .card{background:#fff;border-radius:12px;padding:20px;box-shadow:0 2px 10px rgba(11,32,39,.07);margin-top:16px}
 label{display:block;font-weight:600;font-size:13px;margin:12px 0 4px}
 input,textarea{width:100%;padding:9px;border:1px solid #d9e6e8;border-radius:8px;box-sizing:border-box;font-size:14px}
 textarea{min-height:150px;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;line-height:1.5}
 .row{display:flex;gap:16px;align-items:flex-end} .row>div:first-child{flex:1}
 button{margin-top:16px;background:#0e7c86;color:#fff;border:0;border-radius:8px;padding:11px 22px;
      font-size:15px;font-weight:600;cursor:pointer} button:disabled{opacity:.5;cursor:default}
 .step{display:flex;gap:14px;padding:12px 0;border-top:1px solid #eef4f5}
 .step:first-child{border-top:0}
 .step img{width:150px;border-radius:8px;border:1px solid #d9e6e8;flex:none;object-fit:cover}
 .step .noimg{width:150px;height:90px;border-radius:8px;background:#f0f4f5;color:#9bb1b8;
      display:flex;align-items:center;justify-content:center;font-size:12px;flex:none;text-align:center}
 .st{font-weight:600} .tm{font-weight:600;font-size:13px;margin:2px 0}
 .ev{color:#5c7079;font-size:13px}
 .banner{padding:11px 14px;border-radius:8px;font-weight:600;margin-bottom:10px}
 .ok{background:#e2f4f0;color:#0e7c86} .warn{background:#fff4df;color:#9a6b15} .alert{background:#fde7e4;color:#c0392b}
 .badge{font-size:11px;font-weight:700;padding:2px 8px;border-radius:10px;margin-left:6px;vertical-align:1px}
 .b-green{background:#e2f4f0;color:#0e7c86} .b-yellow{background:#fff1d6;color:#a9701b} .b-red{background:#fde7e4;color:#c0392b}
 .l-yellow{outline:2px solid #f3d28a} .l-red{outline:2px solid #f0b3aa}
 .l-yellow,.l-red{outline-offset:-2px;border-radius:8px;padding-left:10px}
 .legend{color:#5c7079;font-size:12px;margin-top:6px}
 .mood{font-weight:700;font-size:16px;margin:10px 0 2px}
 input[type=range]{width:100%;accent-color:#0e7c86;cursor:pointer}
 .ticks{display:flex;justify-content:space-between;font-size:11px;color:#5c7079;margin-top:3px}
 .ticks span{flex:1;text-align:center}
 details{margin-top:10px;font-size:13px;color:#5c7079} summary{cursor:pointer;font-weight:600}
 .meta{color:#5c7079;font-size:12px;margin-bottom:6px} .vid-h{font-weight:700;font-size:15px;margin:0 0 6px}
 .spin{display:inline-block;width:15px;height:15px;border:2px solid #fff;border-top-color:transparent;
      border-radius:50%;animation:s .7s linear infinite;vertical-align:-2px;margin-right:7px}
 @keyframes s{to{transform:rotate(360deg)}}
 /* tabs */
 .tabs{display:flex;gap:4px;margin-bottom:0;margin-top:0}
 .tab{padding:10px 22px;border-radius:10px 10px 0 0;font-weight:600;font-size:14px;cursor:pointer;
      background:#e4eef0;color:#5c7079;border:none;transition:background .15s}
 .tab.active{background:#0e7c86;color:#fff}
 .tab-panel{display:none} .tab-panel.active{display:block}
 /* benchmark */
 .bm-section{margin-top:20px}
 .bm-title{font-weight:700;font-size:17px;margin-bottom:12px;color:#17323a}
 .bm-sub{font-size:13px;color:#5c7079;margin-bottom:16px}
 table{width:100%;border-collapse:collapse;font-size:13px}
 th{background:#f0f6f7;font-weight:700;text-align:left;padding:8px 10px;border-bottom:2px solid #d9e6e8}
 td{padding:7px 10px;border-bottom:1px solid #eef4f5;vertical-align:middle}
 tr:last-child td{border-bottom:none}
 .eng-claude{color:#0e7c86;font-weight:700} .eng-7b{color:#7c5e0e;font-weight:700} .eng-3b{color:#5e0e7c;font-weight:700}
 .bar-wrap{background:#eef4f5;border-radius:6px;height:10px;width:120px;display:inline-block;vertical-align:middle}
 .bar-fill{height:10px;border-radius:6px;display:block}
 .bar-claude{background:#0e7c86} .bar-7b{background:#c8960e} .bar-3b{background:#8e24aa}
 .cat-tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px}
 .cat-tab{padding:5px 14px;border-radius:20px;font-size:12px;font-weight:600;cursor:pointer;
          background:#eef4f5;color:#5c7079;border:none;transition:background .15s}
 .cat-tab.active{background:#0e7c86;color:#fff}
 .metric-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px}
 .metric-card{background:#f7fafb;border:1px solid #d9e6e8;border-radius:10px;padding:12px}
 .metric-label{font-size:11px;font-weight:600;color:#5c7079;text-transform:uppercase;letter-spacing:.5px}
 .metric-val{font-size:22px;font-weight:700;margin:2px 0 0}
 .mc-claude{color:#0e7c86} .mc-7b{color:#c8960e} .mc-3b{color:#8e24aa}
 .winner{font-size:11px;margin-top:2px;color:#5c7079}
 .timeline-wrap{overflow-x:auto;margin-top:8px}
 .tl-row{display:flex;align-items:center;gap:8px;margin-bottom:5px;font-size:12px}
 .tl-label{width:90px;flex-shrink:0;color:#5c7079;text-align:right}
 .tl-bar{position:relative;height:18px;border-radius:4px;flex:1;background:#eef4f5}
 .tl-gt{position:absolute;height:100%;border-radius:4px;background:rgba(14,124,134,.18);border:1.5px solid #0e7c86}
 .tl-pred{position:absolute;height:8px;top:5px;border-radius:3px;opacity:.85}
 .tl-pred-claude{background:#0e7c86} .tl-pred-7b{background:#c8960e} .tl-pred-3b{background:#8e24aa}
 .tl-miss{font-size:11px;color:#c0392b;margin-left:4px}
</style></head><body>
<h1>🎬↔📋 Protocol–Video Review</h1>
<p class="sub">Attach a protocol file and one or more videos; Claude (<b>{{model}}</b>) reviews each video against the protocol, step by step.</p>

<div class="tabs">
 <button class="tab active" onclick="switchTab('review',this)">📋 Review</button>
 <button class="tab" onclick="switchTab('benchmark',this)">📊 Benchmark</button>
</div>

<div id="tab-review" class="tab-panel active">
<div class="card">
 <form id="f">
  <label>Protocol file(s) — a .txt with one step per line (multiple files are merged)</label>
  <input type="file" name="protocol" accept=".txt,.md,text/plain" multiple required>
  <div class="row">
   <div><label>Video file(s) — one or more .mp4/.mov (each reviewed separately)</label>
        <input type="file" name="videos" accept="video/*" multiple required></div>
   <div><label>Frames / video (max {{max_frames}})</label>
        <input type="number" name="frames" value="20" min="2" max="{{max_frames}}" style="width:100px"></div>
  </div>
  <button id="b" type="submit">Review video(s)</button>
 </form>
</div>
<div id="out"></div>
</div>

<div id="tab-benchmark" class="tab-panel">
<div class="card bm-section">
 <div class="bm-title">VLM Benchmark — Protocol Step Localization</div>
 <div class="bm-sub">Comparing <span class="eng-claude">Claude Opus 4.8</span>, <span class="eng-7b">Qwen2.5-VL-7B (4-bit)</span>, and <span class="eng-3b">Qwen2.5-VL-3B</span> on 4 lab procedure categories · 24 frames/video</div>

 <div class="cat-tabs" id="catTabs">
  <button class="cat-tab active" onclick="setCat('all',this)">All categories</button>
  <button class="cat-tab" onclick="setCat('extractdna',this)">extractdna</button>
  <button class="cat-tab" onclick="setCat('gel',this)">gel</button>
  <button class="cat-tab" onclick="setCat('electrophoresis',this)">electrophoresis</button>
  <button class="cat-tab" onclick="setCat('purifydna',this)">purifydna</button>
 </div>

 <div class="metric-grid" id="metricGrid"></div>

 <div style="font-weight:700;font-size:14px;margin:16px 0 8px">All metrics by engine</div>
 <div id="bmTable"></div>

 <div style="font-weight:700;font-size:14px;margin:20px 0 8px">Step timeline — predicted vs ground truth <span style="font-size:12px;font-weight:400;color:#5c7079">(select a category above)</span></div>
 <div id="bmTimeline"></div>
</div>
</div>

<script>
// ── tab switching ──
function switchTab(id,btn){
 document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
 document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
 document.getElementById('tab-'+id).classList.add('active');
 btn.classList.add('active');
 if(id==='benchmark' && !BM) loadBenchmark();
}

// ── review tab ──
const f=document.getElementById('f'),b=document.getElementById('b'),out=document.getElementById('out');
function fmt(t){t=Math.round(t);return (t<60?t+'s':Math.floor(t/60)+'m'+String(t%60).padStart(2,'0')+'s');}
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
const DAYS=['monday','tuesday','wednesday','thursday','friday'];
const MOOD=['🧐 Monday — strict audit','🤨 Tuesday — picky','🙂 Wednesday — balanced','😌 Thursday — easygoing','😎 Friday — relaxed'];
const DOT={green:'🟢',yellow:'🟡',red:'🔴'},LBL={green:'done / probably done',yellow:'uncertain',red:'appears missed'};
let DATA=null, dayIdx=4;
function renderStep(v,s){
 const lvl=(s.level in DOT)?s.level:'yellow';
 const thumb=(s.best_frame_index>=0 && v.thumbs[s.best_frame_index])?v.thumbs[s.best_frame_index]:null;
 const img=thumb?`<img src="data:image/jpeg;base64,${thumb}">`:`<div class="noimg">${DOT[lvl]} ${lvl=='red'?'not seen':'no clear frame'}</div>`;
 const tm=(s.best_frame_index>=0)?`<div class="tm" style="color:#0e7c86">⏱ ${fmt(s.start_time_s)} – ${fmt(s.end_time_s)}</div>`:'';
 const badge=`<span class="badge b-${lvl}">${DOT[lvl]} ${LBL[lvl]}</span>`;
 return `<div class="step l-${lvl}">${img}<div><div class="st">${s.step_number}. ${esc(s.step_text)}${badge}</div>`+
        `${tm}<div class="ev">${esc(s.note)}</div></div></div>`;}
function renderVideo(v){
 if(v.error) return `<div class="card"><div class="vid-h">🎬 ${esc(v.filename)}</div><div>⚠️ ${esc(v.error)}</div></div>`;
 const day=v.days[DAYS[dayIdx]]||{};
 if(day.error) return `<div class="card"><div class="vid-h">🎬 ${esc(v.filename)}</div><div>⚠️ ${esc(day.error)}</div></div>`;
 const banner=day.n_red?`<div class="banner alert">🔴 ${day.n_red} step${day.n_red>1?'s':''} appear missed`+
                        (day.n_yellow?` · 🟡 ${day.n_yellow} uncertain`:'')+`</div>`
   :day.n_yellow?`<div class="banner warn">🟡 ${day.n_yellow} step${day.n_yellow>1?'s':''} uncertain — worth a glance</div>`
   :`<div class="banner ok">🟢 All steps look done</div>`;
 const obs=day.observed_summary?`<details><summary>What Claude observed</summary><div>${esc(day.observed_summary)}</div></details>`:'';
 return `<div class="card"><div class="vid-h">🎬 ${esc(v.filename)}</div>`+
        `<div class="meta">${v.n_frames} frames · ~${fmt(v.duration)}</div>${banner}`+
        (day.steps||[]).map(s=>renderStep(v,s)).join('')+obs+`</div>`;}
function renderVids(){document.getElementById('vids').innerHTML=DATA.results.map(renderVideo).join('');}
function renderAll(){
 const proto=`<div class="card"><div class="vid-h">📋 Protocol (${DATA.steps_parsed.length} steps)</div>`+
   `<ol style="margin:0;padding-left:20px;font-size:13px;color:#39535b">`+
   DATA.steps_parsed.map(s=>`<li>${esc(s)}</li>`).join('')+`</ol>`+
   `<div class="legend">🟢 done / probably done · 🟡 uncertain · 🔴 appears missed</div></div>`;
 const slider=`<div class="card"><div class="vid-h">🗓️ Strictness dial</div>`+
   `<input type="range" id="day" min="0" max="4" value="${dayIdx}">`+
   `<div class="ticks"><span>Mon</span><span>Tue</span><span>Wed</span><span>Thu</span><span>Fri</span></div>`+
   `<div class="mood" id="mood">${MOOD[dayIdx]}</div>`+
   `<div class="legend">Monday = strict, Friday = relaxed. All 5 levels were evaluated in parallel — slide to compare instantly.</div></div>`;
 out.innerHTML=proto+slider+`<div id="vids"></div>`;
 const day=document.getElementById('day');
 day.oninput=()=>{dayIdx=+day.value;document.getElementById('mood').textContent=MOOD[dayIdx];renderVids();};
 renderVids();}
f.onsubmit=async e=>{e.preventDefault();
 b.disabled=true;b.innerHTML='<span class="spin"></span>Reviewing (5 strictness levels in parallel)…';
 out.innerHTML='<div class="card">Sampling frames and running all 5 strictness levels per video in parallel — please wait…</div>';
 try{const r=await fetch('/align',{method:'POST',body:new FormData(f)});const d=await r.json();
  if(!r.ok){out.innerHTML=`<div class="card">⚠️ ${esc(d.error||'Error')}</div>`;b.disabled=false;b.textContent='Review video(s)';return;}
  DATA=d;renderAll();
 }catch(err){out.innerHTML=`<div class="card">⚠️ ${esc(String(err))}</div>`;}
 b.disabled=false;b.textContent='Review video(s)';};

// ── benchmark tab ──
let BM=null, activeCat='all';
const ENGINES=['claude','Qwen2.5-VL-7B-Instruct (4-bit)','Qwen2.5-VL-3B-Instruct'];
const ENG_SHORT={'claude':'Claude','Qwen2.5-VL-7B-Instruct (4-bit)':'Qwen 7B','Qwen2.5-VL-3B-Instruct':'Qwen 3B'};
const ENG_CLS={'claude':'claude','Qwen2.5-VL-7B-Instruct (4-bit)':'7b','Qwen2.5-VL-3B-Instruct':'3b'};
const METRICS=[
 {key:'mean_iou',label:'Mean IoU',fmt:v=>(v*100).toFixed(1)+'%',best:'high'},
 {key:'localized_frac',label:'Localized',fmt:v=>(v*100).toFixed(0)+'%',best:'high'},
 {key:'start_within_10s',label:'Start ≤10s',fmt:v=>(v*100).toFixed(0)+'%',best:'high'},
 {key:'ordering_acc',label:'Order Acc.',fmt:v=>(v*100).toFixed(0)+'%',best:'high'},
 {key:'latency_s',label:'Latency',fmt:v=>v.toFixed(1)+'s',best:'low'},
 {key:'n_flags',label:'Avg Flags',fmt:v=>v.toFixed(1),best:'low'},
];

async function loadBenchmark(){
 const r=await fetch('/benchmark-data');
 BM=await r.json();
 renderBenchmark();
}

function getStats(cat){
 if(cat==='all') return BM.aggregate;
 const out={};
 ENGINES.forEach(e=>{out[e]=(BM.by_category[e]||{})[cat]||null;});
 return out;
}

function pct(v,max){return Math.round(Math.min(100,v/max*100));}

function renderMetricGrid(stats){
 const grid=document.getElementById('metricGrid');
 grid.innerHTML=METRICS.map(m=>{
  const vals=ENGINES.map(e=>stats[e]?stats[e][m.key]:null).filter(v=>v!==null);
  const best=m.best==='high'?Math.max(...vals):Math.min(...vals);
  const winner=ENGINES.find(e=>stats[e]&&stats[e][m.key]===best);
  return `<div class="metric-card">
   <div class="metric-label">${m.label}</div>
   ${ENGINES.map(e=>{
    const v=stats[e]?stats[e][m.key]:null;
    const cls='mc-'+ENG_CLS[e];
    return v!==null?`<div class="metric-val ${cls}" style="font-size:15px">${ENG_SHORT[e]}: ${m.fmt(v)}</div>`:''
   }).join('')}
   <div class="winner">Best: <b>${ENG_SHORT[winner]||'—'}</b></div>
  </div>`;
 }).join('');
}

function renderTable(stats){
 const COLS=[
  {key:'mean_iou',label:'Mean IoU',fmt:v=>(v*100).toFixed(1)+'%',bar:true,max:1},
  {key:'localized_frac',label:'Localized %',fmt:v=>(v*100).toFixed(0)+'%',bar:true,max:1},
  {key:'start_within_5s',label:'Start≤5s',fmt:v=>(v*100).toFixed(0)+'%',bar:false},
  {key:'start_within_10s',label:'Start≤10s',fmt:v=>(v*100).toFixed(0)+'%',bar:false},
  {key:'ordering_acc',label:'Order Acc.',fmt:v=>(v*100).toFixed(0)+'%',bar:false},
  {key:'latency_s',label:'Latency (s)',fmt:v=>v.toFixed(1),bar:true,max:200,invert:true},
  {key:'n_flags',label:'Avg Flags',fmt:v=>v.toFixed(1),bar:false},
 ];
 let h=`<table><tr><th>Engine</th>${COLS.map(c=>`<th>${c.label}</th>`).join('')}</tr>`;
 ENGINES.forEach(e=>{
  const s=stats[e];
  if(!s){h+=`<tr><td class="eng-${ENG_CLS[e]}">${ENG_SHORT[e]}</td>${COLS.map(()=>`<td>—</td>`).join('')}</tr>`;return;}
  h+=`<tr><td class="eng-${ENG_CLS[e]}">${ENG_SHORT[e]}</td>`;
  COLS.forEach(c=>{
   const v=s[c.key];
   if(c.bar){
    const pct2=c.invert?Math.round((1-v/c.max)*100):Math.round(v/c.max*100);
    h+=`<td>${c.fmt(v)} <span class="bar-wrap"><span class="bar-fill bar-${ENG_CLS[e]}" style="width:${pct2}%"></span></span></td>`;
   } else {
    h+=`<td>${c.fmt(v)}</td>`;
   }
  });
  h+=`</tr>`;
 });
 h+=`</table>`;
 document.getElementById('bmTable').innerHTML=h;
}

function renderTimeline(cat){
 const tl=document.getElementById('bmTimeline');
 if(cat==='all'){tl.innerHTML='<div style="color:#5c7079;font-size:13px">Select a category to see step-level timelines.</div>';return;}
 const runs=BM.runs.filter(r=>r.category===cat);
 if(!runs.length){tl.innerHTML='<div style="color:#5c7079;font-size:13px">No data.</div>';return;}
 const nSteps=runs[0].n_steps;
 const maxT=Math.max(...runs.flatMap(r=>r.perstep.flatMap(p=>[p.gt[1],p.pred?p.pred[1]:0])));
 let html='<div class="timeline-wrap">';
 for(let i=0;i<nSteps;i++){
  html+=`<div style="margin-bottom:12px"><div style="font-size:12px;font-weight:600;color:#17323a;margin-bottom:4px">Step ${i+1}</div>`;
  runs.forEach(r=>{
   const p=r.perstep[i];
   const cls=ENG_CLS[r.engine];
   const gtL=pct(p.gt[0],maxT),gtW=Math.max(1,pct(p.gt[1]-p.gt[0],maxT));
   let predBar='';
   if(p.pred){
    const pL=pct(p.pred[0],maxT),pW=Math.max(1,pct(p.pred[1]-p.pred[0],maxT));
    predBar=`<span class="tl-pred tl-pred-${cls}" style="left:${pL}%;width:${pW}%"></span>`;
   } else {
    predBar=`<span class="tl-miss">✗ not localized</span>`;
   }
   html+=`<div class="tl-row"><div class="tl-label">${ENG_SHORT[r.engine]}</div>`+
    `<div class="tl-bar"><span class="tl-gt" style="left:${gtL}%;width:${gtW}%"></span>${p.pred?predBar:''}</div>`+
    (p.pred?'':`<span class="tl-miss" style="font-size:11px;color:#c0392b">✗</span>`)+
   `</div>`;
  });
  html+='</div>';
 }
 html+=`<div style="font-size:11px;color:#5c7079;margin-top:6px">
   <span style="display:inline-block;width:28px;height:10px;background:rgba(14,124,134,.18);border:1.5px solid #0e7c86;border-radius:3px;vertical-align:middle"></span> ground truth &nbsp;
   <span style="display:inline-block;width:28px;height:8px;background:#0e7c86;border-radius:3px;vertical-align:middle"></span> Claude &nbsp;
   <span style="display:inline-block;width:28px;height:8px;background:#c8960e;border-radius:3px;vertical-align:middle"></span> Qwen 7B &nbsp;
   <span style="display:inline-block;width:28px;height:8px;background:#8e24aa;border-radius:3px;vertical-align:middle"></span> Qwen 3B
  </div></div>`;
 tl.innerHTML=html;
}

function renderBenchmark(){
 const stats=getStats(activeCat);
 renderMetricGrid(stats);
 renderTable(stats);
 renderTimeline(activeCat);
}

function setCat(cat,btn){
 activeCat=cat;
 document.querySelectorAll('.cat-tab').forEach(t=>t.classList.remove('active'));
 btn.classList.add('active');
 if(BM) renderBenchmark();
}
</script></body></html>"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
