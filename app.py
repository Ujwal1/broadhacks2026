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
        if w > FRAME_WIDTH:
            frame = cv2.resize(frame, (FRAME_WIDTH, int(h * FRAME_WIDTH / w)))
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            frames.append({"idx": len(frames),
                           "t": round(fi / fps, 1) if fps else 0.0,
                           "b64": base64.b64encode(buf).decode()})
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
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "flag": {"type": "boolean"},
                    "warning": {"type": "string"},
                    "start_time_s": {"type": "number"},
                    "end_time_s": {"type": "number"},
                    "best_frame_index": {"type": "integer"},
                    "note": {"type": "string"},
                },
                "required": ["step_number", "step_text", "confidence", "flag",
                             "warning", "start_time_s", "end_time_s", "best_frame_index", "note"],
            },
        },
    },
    "required": ["observed_summary", "steps"],
}


def align(steps, frames):
    """Advisory check: per-step confidence + gentle 'worth reviewing' warnings (not a pass/fail verdict)."""
    proto = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    header = (
        "You are assisting a human who is reviewing whether a lab protocol was followed in a video. "
        "You are NOT issuing a pass/fail verdict — you give a confidence estimate per step and raise "
        "gentle, suggestive warnings worth a quick human look.\n\n"
        "PROTOCOL:\n" + proto + "\n\n"
        f"Below are {len(frames)} frames sampled in chronological order from the video, each "
        "preceded by its index and timestamp. Treat them as one continuous clip."
    )
    instruction = (
        "First, in observed_summary, describe what you actually see in the video (tubes/reagents/"
        "tools handled, actions, order) — independently of the protocol.\n\n"
        "Then for EACH protocol step provide:\n"
        "- confidence: your confidence the step was performed (and performed correctly) from the "
        "visual evidence — 'high' (clearly see it), 'medium' (consistent with it but not certain), "
        "'low' (can't really tell).\n"
        "- flag: true ONLY if the step is genuinely worth a human double-check — it looks possibly "
        "skipped, out of order, or done incorrectly, OR you truly cannot confirm a critical action. "
        "Do NOT flag steps that clearly or plausibly happened. Keep flags light-touch — this is a "
        "warning to glance at, not a strict audit.\n"
        "- warning: if flag is true, one short, non-accusatory sentence phrased as a suggestion to "
        "check (e.g. 'Couldn't clearly confirm template DNA from tube T was added — worth a look.'). "
        "Empty string otherwise.\n"
        "- start_time_s / end_time_s / best_frame_index: when the step is visible (else 0 and -1).\n"
        "- note: brief, plain description of what you saw for this step.\n\n"
        "Be helpful, not pedantic: reserve flags for real concerns. Return every step."
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
    steps_out = []
    for s in data.get("steps", []):
        bi = s.get("best_frame_index", -1)
        thumb = frames[bi]["b64"] if isinstance(bi, int) and 0 <= bi < len(frames) else None
        steps_out.append({**s, "thumb": thumb})
    n_flags = sum(1 for s in steps_out if s.get("flag"))
    return {"observed_summary": data.get("observed_summary", ""),
            "n_flags": n_flags,
            "steps": steps_out}


@app.route("/")
def index():
    return render_template_string(PAGE, model=MODEL, max_frames=MAX_FRAMES)


@app.route("/align", methods=["POST"])
def align_route():
    f = request.files.get("video")
    if not f or not f.filename:
        return jsonify(error="No video uploaded."), 400
    steps = parse_steps(request.form.get("protocol", ""))
    if not steps:
        return jsonify(error="Paste a protocol (one step per line)."), 400
    try:
        n = int(request.form.get("frames", 16))
    except ValueError:
        n = 16

    suffix = os.path.splitext(f.filename)[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        f.save(tmp.name)
        tmp.close()
        frames, total, duration, fps = sample_frames(tmp.name, n)
        result = align(steps, frames)
        return jsonify(n_frames=len(frames), duration=round(duration, 1), **result)
    except anthropic.APIError as e:
        return jsonify(error=f"Claude API error: {e}"), 502
    except json.JSONDecodeError:
        return jsonify(error="Claude returned an unparseable result; try fewer frames or a shorter protocol."), 502
    except Exception as e:
        return jsonify(error=str(e)), 400
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


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
 .ok{background:#e2f4f0;color:#0e7c86} .warn{background:#fff4df;color:#9a6b15}
 .warntext{color:#9a6b15;font-size:13px;font-weight:500;margin:2px 0}
 .badge{font-size:11px;font-weight:700;padding:2px 7px;border-radius:10px;margin-left:6px;vertical-align:1px}
 .b-high{background:#e2f4f0;color:#0e7c86} .b-medium{background:#eaf2f4;color:#3a7a86}
 .b-low{background:#eef0f1;color:#7a8a90} .b-flag{background:#fff1d6;color:#a9701b}
 .review{outline:2px solid #f3d28a;outline-offset:-2px;border-radius:8px;padding-left:10px}
 details{margin-top:10px;font-size:13px;color:#5c7079} summary{cursor:pointer;font-weight:600}
 #meta{color:#5c7079;font-size:12px;margin-bottom:6px}
 .spin{display:inline-block;width:15px;height:15px;border:2px solid #fff;border-top-color:transparent;
      border-radius:50%;animation:s .7s linear infinite;vertical-align:-2px;margin-right:7px}
 @keyframes s{to{transform:rotate(360deg)}}
</style></head><body>
<h1>🎬↔📋 Protocol–Video Alignment</h1>
<p class="sub">Paste a protocol and upload a video; Claude (<b>{{model}}</b>) finds the frames for each step.</p>
<div class="card">
 <form id="f">
  <label>Protocol — one step per line</label>
  <textarea name="protocol" placeholder="1. Add Master Mix to the tube&#10;2. Add forward primer&#10;3. Add reverse primer&#10;4. Add template DNA&#10;5. Flick to mix and quick-spin&#10;6. Place in thermocycler"></textarea>
  <div class="row">
   <div><label>Video file (.mp4 / .mov)</label><input type="file" name="video" accept="video/*" required></div>
   <div><label>Frames (max {{max_frames}})</label><input type="number" name="frames" value="20" min="2" max="{{max_frames}}" style="width:90px"></div>
  </div>
  <button id="b" type="submit">Review video</button>
 </form>
</div>
<div class="card" id="result" style="display:none">
 <div id="meta"></div><div id="banner"></div><div id="steps"></div>
 <details id="obs" style="display:none"><summary>What Claude observed in the video</summary><div id="obstext"></div></details>
</div>
<script>
const f=document.getElementById('f'),b=document.getElementById('b'),res=document.getElementById('result'),
      steps=document.getElementById('steps'),meta=document.getElementById('meta'),banner=document.getElementById('banner'),
      obs=document.getElementById('obs'),obstext=document.getElementById('obstext');
function fmt(t){t=Math.round(t);return (t<60?t+'s':Math.floor(t/60)+'m'+String(t%60).padStart(2,'0')+'s');}
const CONF={high:'likely done',medium:'probably done',low:'unconfirmed'};
f.onsubmit=async e=>{e.preventDefault();
 b.disabled=true;b.innerHTML='<span class="spin"></span>Checking…';
 res.style.display='block';meta.textContent='';banner.innerHTML='';obs.style.display='none';
 steps.innerHTML='Sampling frames and reviewing the protocol against the video…';
 try{const r=await fetch('/align',{method:'POST',body:new FormData(f)});const d=await r.json();
  if(!r.ok){steps.innerHTML='⚠️ '+(d.error||'Error');b.disabled=false;b.textContent='Review video';return;}
  meta.textContent=`${d.n_frames} frames sampled · ~${fmt(d.duration)} clip`;
  if(!d.n_flags){banner.innerHTML='<div class="banner ok">✓ Looks complete — no steps flagged for review</div>';}
  else{banner.innerHTML=`<div class="banner warn">⚠ ${d.n_flags} step${d.n_flags>1?'s':''} worth a quick review (suggestions, not failures)</div>`;}
  steps.innerHTML='';
  d.steps.forEach(s=>{const div=document.createElement('div');div.className='step'+(s.flag?' review':'');
   const img=s.thumb?`<img src="data:image/jpeg;base64,${s.thumb}">`:`<div class="noimg">no clear frame</div>`;
   const tm=(s.best_frame_index>=0)?`<div class="tm" style="color:#0e7c86">⏱ ${fmt(s.start_time_s)} – ${fmt(s.end_time_s)}</div>`:'';
   const badge=s.flag?`<span class="badge b-flag">⚠ worth a review</span>`
                     :`<span class="badge b-${s.confidence}">${CONF[s.confidence]||s.confidence}</span>`;
   const warn=s.flag&&s.warning?`<div class="warntext">💡 ${s.warning}</div>`:'';
   div.innerHTML=`${img}<div><div class="st">${s.step_number}. ${s.step_text}${badge}</div>`+
                 `${tm}${warn}<div class="ev">${s.note||''}</div></div>`;
   steps.appendChild(div);});
  if(d.observed_summary){obstext.textContent=d.observed_summary;obs.style.display='block';}
 }catch(err){steps.innerHTML='⚠️ '+err;}
 b.disabled=false;b.textContent='Review video';};
</script></body></html>"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
