#!/usr/bin/env python
"""
Protocol → video step alignment, with a Claude vs open-source-VLM comparison.

Paste a protocol (one step per line) and upload a video of someone performing it.
The app samples timestamped frames and, for each protocol step, returns a confidence,
a time-range, and the most representative frame — as an advisory review (not pass/fail).

You can run the SAME task with two engines and see them side by side:
  - Claude   (claude-opus-4-8, via the Anthropic API)
  - VLM      (Qwen2.5-VL, a local open-source vision-language model on the GPU)
Both get the identical sampled frames + protocol, so the comparison is fair.

Claude's API takes images, not video — so frames are the unit of alignment.

API key: put it (one line) in  anthropic_key.txt  — read automatically.
($ANTHROPIC_API_KEY, $ANTHROPIC_API_KEY_FILE, .anthropic_key, .env also honored.)

Run:
    pip install -r requirements.txt
    echo "sk-ant-..." > anthropic_key.txt && chmod 600 anthropic_key.txt
    python app.py            # http://0.0.0.0:7860
    # laptop:  ssh -L 7860:localhost:7860 <host>  -> open localhost:7860
The local VLM is optional: if torch/CUDA/transformers aren't available the app still
runs Claude-only and the VLM option is disabled in the UI.
"""
import base64
import json
import os
import tempfile
import time

import cv2
import anthropic
from flask import Flask, request, jsonify, render_template_string

MODEL = "claude-opus-4-8"
MAX_FRAMES = 30
FRAME_WIDTH = 1280       # higher res so small tube/reagent labels are legible
PORT = int(os.environ.get("PORT", "7860"))

# --- optional local VLM engine (mirrors align()'s schema) -------------------
try:
    import torch
    _CUDA = torch.cuda.is_available()
except Exception:
    _CUDA = False
try:
    import vlm_engine
    VLM_AVAILABLE = bool(_CUDA)
except Exception:
    vlm_engine = None
    VLM_AVAILABLE = False
VLM_LABEL = vlm_engine.vlm_name() if vlm_engine else "VLM"

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


def run_engine(engine, steps, frames):
    """Run one engine and wrap it with a label + timing + error capture (so one engine
    failing never kills the other in a side-by-side comparison)."""
    t0 = time.time()
    try:
        if engine == "claude":
            res = align(steps, frames)
            label = f"Claude · {MODEL}"
        elif engine == "vlm":
            if not VLM_AVAILABLE:
                raise RuntimeError("Local VLM not available (needs CUDA + transformers).")
            res = vlm_engine.align_vlm(steps, frames)
            label = f"VLM · {VLM_LABEL}"
        else:
            raise ValueError(f"unknown engine {engine}")
        res["label"] = label
        res["elapsed_s"] = round(time.time() - t0, 1)
        return res
    except Exception as e:
        return {"label": {"claude": f"Claude · {MODEL}", "vlm": f"VLM · {VLM_LABEL}"}.get(engine, engine),
                "error": str(e), "elapsed_s": round(time.time() - t0, 1),
                "observed_summary": "", "n_flags": 0, "steps": []}


@app.route("/")
def index():
    return render_template_string(PAGE, model=MODEL, max_frames=MAX_FRAMES,
                                  vlm_available=VLM_AVAILABLE, vlm_label=VLM_LABEL)


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

    # which engine(s): claude | vlm | both
    engine = request.form.get("engine", "both")
    if engine == "both":
        engines = ["claude", "vlm"] if VLM_AVAILABLE else ["claude"]
    elif engine == "vlm":
        engines = ["vlm"] if VLM_AVAILABLE else ["claude"]
    else:
        engines = ["claude"]

    suffix = os.path.splitext(f.filename)[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        f.save(tmp.name)
        tmp.close()
        frames, total, duration, fps = sample_frames(tmp.name, n)
        results = {eng: run_engine(eng, steps, frames) for eng in engines}  # same frames -> fair
        return jsonify(n_frames=len(frames), duration=round(duration, 1),
                       engines=engines, results=results)
    except anthropic.APIError as e:
        return jsonify(error=f"Claude API error: {e}"), 502
    except Exception as e:
        return jsonify(error=str(e)), 400
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Protocol ↔ Video Alignment — Claude vs VLM</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1180px;
      margin:36px auto;padding:0 20px;color:#17323a;background:#f7fafb}
 h1{font-size:23px;margin-bottom:2px} .sub{color:#5c7079;margin-top:0;font-size:14px}
 .card{background:#fff;border-radius:12px;padding:20px;box-shadow:0 2px 10px rgba(11,32,39,.07);margin-top:16px}
 label{display:block;font-weight:600;font-size:13px;margin:12px 0 4px}
 input,textarea{width:100%;padding:9px;border:1px solid #d9e6e8;border-radius:8px;box-sizing:border-box;font-size:14px}
 textarea{min-height:140px;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;line-height:1.5}
 .row{display:flex;gap:16px;align-items:flex-end;flex-wrap:wrap} .row>div:first-child{flex:1;min-width:220px}
 .engsel{display:flex;gap:8px;margin-top:4px} .engsel label{display:flex;align-items:center;gap:6px;font-weight:500;
      margin:0;padding:8px 12px;border:1px solid #d9e6e8;border-radius:8px;cursor:pointer;font-size:13px;background:#fff}
 .engsel input{width:auto} .engsel label:has(input:checked){border-color:#0e7c86;background:#e8f5f3;font-weight:600}
 button{margin-top:16px;background:#0e7c86;color:#fff;border:0;border-radius:8px;padding:11px 22px;
      font-size:15px;font-weight:600;cursor:pointer} button:disabled{opacity:.5;cursor:default}
 .cols{display:flex;gap:18px;align-items:flex-start} .cols>div{flex:1;min-width:0}
 .enghead{font-size:15px;font-weight:700;padding:10px 14px;border-radius:8px 8px 0 0;color:#fff}
 .enghead.claude{background:#0e7c86} .enghead.vlm{background:#5b54c9}
 .engbody{border:1px solid #e3ebec;border-top:0;border-radius:0 0 8px 8px;padding:14px}
 .step{display:flex;gap:12px;padding:11px 0;border-top:1px solid #eef4f5}
 .step:first-child{border-top:0}
 .step img{width:128px;border-radius:8px;border:1px solid #d9e6e8;flex:none;object-fit:cover}
 .step .noimg{width:128px;height:78px;border-radius:8px;background:#f0f4f5;color:#9bb1b8;
      display:flex;align-items:center;justify-content:center;font-size:11px;flex:none;text-align:center}
 .st{font-weight:600;font-size:14px} .tm{font-weight:600;font-size:12px;margin:2px 0}
 .ev{color:#5c7079;font-size:12.5px;line-height:1.4}
 .banner{padding:9px 12px;border-radius:8px;font-weight:600;margin-bottom:10px;font-size:13px}
 .ok{background:#e2f4f0;color:#0e7c86} .warn{background:#fff4df;color:#9a6b15} .err{background:#fde8e8;color:#b3261e}
 .warntext{color:#9a6b15;font-size:12.5px;font-weight:500;margin:2px 0}
 .badge{font-size:10.5px;font-weight:700;padding:2px 7px;border-radius:10px;margin-left:6px;vertical-align:1px}
 .b-high{background:#e2f4f0;color:#0e7c86} .b-medium{background:#eaf2f4;color:#3a7a86}
 .b-low{background:#eef0f1;color:#7a8a90} .b-flag{background:#fff1d6;color:#a9701b}
 .review{outline:2px solid #f3d28a;outline-offset:-2px;border-radius:8px;padding-left:8px}
 details{margin-top:10px;font-size:12.5px;color:#5c7079} summary{cursor:pointer;font-weight:600}
 .elapsed{font-size:12px;font-weight:500;opacity:.85;float:right}
 #meta{color:#5c7079;font-size:12px;margin-bottom:10px}
 .spin{display:inline-block;width:15px;height:15px;border:2px solid #fff;border-top-color:transparent;
      border-radius:50%;animation:s .7s linear infinite;vertical-align:-2px;margin-right:7px}
 @keyframes s{to{transform:rotate(360deg)}}
</style></head><body>
<h1>🎬↔📋 Protocol–Video Alignment</h1>
<p class="sub">Paste a protocol and upload a video. The same frames + protocol are reviewed by
<b>Claude ({{model}})</b>{% if vlm_available %} and a local open VLM (<b>{{vlm_label}}</b>){% endif %},
shown side by side for comparison.</p>
<div class="card">
 <form id="f">
  <label>Protocol — one step per line</label>
  <textarea name="protocol" placeholder="1. Mark four tubes&#10;2. Add 90 µL sample to tube 1, 90 µL buffer to the rest&#10;3. Add 10 µL sample to the 10^-1 tube and mix&#10;4. Serially transfer 10 µL down the dilution series"></textarea>
  <div class="row">
   <div><label>Video file (.mp4 / .mov)</label><input type="file" name="video" accept="video/*" required></div>
   <div><label>Frames (max {{max_frames}})</label><input type="number" name="frames" value="16" min="2" max="{{max_frames}}" style="width:90px"></div>
  </div>
  <label>Engine</label>
  <div class="engsel">
   <label><input type="radio" name="engine" value="both" {% if vlm_available %}checked{% endif %} {% if not vlm_available %}disabled{% endif %}>Compare both</label>
   <label><input type="radio" name="engine" value="claude" {% if not vlm_available %}checked{% endif %}>Claude only</label>
   <label><input type="radio" name="engine" value="vlm" {% if not vlm_available %}disabled{% endif %}>VLM only</label>
  </div>
  {% if not vlm_available %}<div class="sub" style="margin-top:6px">⚠ Local VLM unavailable (no CUDA/transformers) — Claude only.</div>{% endif %}
  <button id="b" type="submit">Review video</button>
 </form>
</div>
<div class="card" id="result" style="display:none">
 <div id="meta"></div><div class="cols" id="cols"></div>
</div>
<script>
const f=document.getElementById('f'),b=document.getElementById('b'),res=document.getElementById('result'),
      cols=document.getElementById('cols'),meta=document.getElementById('meta');
function fmt(t){t=Math.round(t);return (t<60?t+'s':Math.floor(t/60)+'m'+String(t%60).padStart(2,'0')+'s');}
const CONF={high:'likely done',medium:'probably done',low:'unconfirmed'};
function esc(s){const d=document.createElement('div');d.textContent=s==null?'':s;return d.innerHTML;}

function renderEngine(key,r){
 const head=`<div class="enghead ${key}">${esc(r.label)}`+
   (r.elapsed_s!=null?`<span class="elapsed">⏱ ${r.elapsed_s}s</span>`:'')+`</div>`;
 if(r.error){return head+`<div class="engbody"><div class="banner err">⚠ ${esc(r.error)}</div></div>`;}
 let banner = !r.n_flags
   ? `<div class="banner ok">✓ Looks complete — no steps flagged</div>`
   : `<div class="banner warn">⚠ ${r.n_flags} step${r.n_flags>1?'s':''} worth a quick review</div>`;
 let body='';
 (r.steps||[]).forEach(s=>{
  const img=s.thumb?`<img src="data:image/jpeg;base64,${s.thumb}">`:`<div class="noimg">no clear frame</div>`;
  const tm=(s.best_frame_index>=0)?`<div class="tm" style="color:#0e7c86">⏱ ${fmt(s.start_time_s)} – ${fmt(s.end_time_s)}</div>`:'';
  const badge=s.flag?`<span class="badge b-flag">⚠ review</span>`
                    :`<span class="badge b-${s.confidence}">${CONF[s.confidence]||esc(s.confidence)}</span>`;
  const warn=s.flag&&s.warning?`<div class="warntext">💡 ${esc(s.warning)}</div>`:'';
  body+=`<div class="step${s.flag?' review':''}">${img}<div><div class="st">${s.step_number}. ${esc(s.step_text)}${badge}</div>`+
        `${tm}${warn}<div class="ev">${esc(s.note)}</div></div></div>`;
 });
 const obs=r.observed_summary?`<details><summary>What it observed</summary><div>${esc(r.observed_summary)}</div></details>`:'';
 return head+`<div class="engbody">${banner}${body}${obs}</div>`;
}

f.onsubmit=async e=>{e.preventDefault();
 b.disabled=true;b.innerHTML='<span class="spin"></span>Reviewing…';
 res.style.display='block';meta.textContent='';
 cols.innerHTML='<div>Sampling frames and reviewing the protocol against the video… (the local VLM can take ~30–60s)</div>';
 try{const r=await fetch('/align',{method:'POST',body:new FormData(f)});const d=await r.json();
  if(!r.ok){cols.innerHTML='⚠️ '+esc(d.error||'Error');b.disabled=false;b.textContent='Review video';return;}
  meta.textContent=`${d.n_frames} frames sampled · ~${fmt(d.duration)} clip · engines: ${d.engines.join(' vs ')}`;
  cols.innerHTML='';
  d.engines.forEach(k=>{const div=document.createElement('div');div.innerHTML=renderEngine(k,d.results[k]);cols.appendChild(div);});
 }catch(err){cols.innerHTML='⚠️ '+esc(''+err);}
 b.disabled=false;b.textContent='Review video';};
</script></body></html>"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
