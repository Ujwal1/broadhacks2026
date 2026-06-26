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


@app.route("/")
def index():
    return render_template_string(PAGE, model=MODEL, max_frames=MAX_FRAMES)


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
 body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:860px;
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
</style></head><body>
<h1>🎬↔📋 Protocol–Video Review</h1>
<p class="sub">Attach a protocol file and one or more videos; Claude (<b>{{model}}</b>) reviews each video against the protocol, step by step.</p>
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
<script>
const f=document.getElementById('f'),b=document.getElementById('b'),out=document.getElementById('out');
function fmt(t){t=Math.round(t);return (t<60?t+'s':Math.floor(t/60)+'m'+String(t%60).padStart(2,'0')+'s');}
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
const DAYS=['monday','tuesday','wednesday','thursday','friday'];
const MOOD=['🧐 Monday — strict audit','🤨 Tuesday — picky','🙂 Wednesday — balanced','😌 Thursday — easygoing','😎 Friday — relaxed'];
const DOT={green:'🟢',yellow:'🟡',red:'🔴'},LBL={green:'done / probably done',yellow:'uncertain',red:'appears missed'};
let DATA=null, dayIdx=4;   // default Friday (most relaxed)
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
</script></body></html>"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
