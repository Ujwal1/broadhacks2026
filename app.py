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

import prompts
import metrics

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
    header = prompts.build_header(steps, len(frames))
    instruction = prompts.INSTRUCTION

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
        # when both ran, quantify how much they agree (no ground truth needed)
        agreement = None
        if "claude" in results and "vlm" in results \
                and not results["claude"].get("error") and not results["vlm"].get("error"):
            agreement = metrics.cross_engine_agreement(results["claude"]["steps"],
                                                       results["vlm"]["steps"])
        return jsonify(n_frames=len(frames), duration=round(duration, 1),
                       engines=engines, results=results, agreement=agreement)
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
 .agree{display:flex;gap:22px;flex-wrap:wrap;background:#eef3fb;border:1px solid #d8e2f3;border-radius:8px;padding:11px 15px;margin-bottom:12px}
 .agree .m{font-size:12px;color:#5c7079} .agree b{font-size:17px;color:#3a3f9a;display:block;font-weight:700}
 .nav{font-size:13px;margin:8px 0 0} .nav a{color:#0e7c86;text-decoration:none;font-weight:600} .nav a:hover{text-decoration:underline}
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
<p class="nav"><a href="/benchmark">📊 View the Claude vs VLM benchmark →</a> &nbsp;(scored against BioVL-QR ground-truth timestamps)</p>
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
 <div id="meta"></div><div id="agree"></div><div class="cols" id="cols"></div>
</div>
<script>
const f=document.getElementById('f'),b=document.getElementById('b'),res=document.getElementById('result'),
      cols=document.getElementById('cols'),meta=document.getElementById('meta'),agree=document.getElementById('agree');
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
 res.style.display='block';meta.textContent='';agree.innerHTML='';
 cols.innerHTML='<div>Sampling frames and reviewing the protocol against the video… (the local VLM can take ~30–60s)</div>';
 try{const r=await fetch('/align',{method:'POST',body:new FormData(f)});const d=await r.json();
  if(!r.ok){cols.innerHTML='⚠️ '+esc(d.error||'Error');b.disabled=false;b.textContent='Review video';return;}
  meta.textContent=`${d.n_frames} frames sampled · ~${fmt(d.duration)} clip · engines: ${d.engines.join(' vs ')}`;
  if(d.agreement){const a=d.agreement;
   agree.innerHTML='<div class="agree"><div style="width:100%" class="m">Claude ↔ VLM agreement on this clip:</div>'+
    `<div><b>${Math.round(a.mean_iou*100)}%</b><span class="m">time-range overlap (IoU)</span></div>`+
    `<div><b>${Math.round(a.flag_agreement*100)}%</b><span class="m">flag agreement</span></div>`+
    `<div><b>${a.mean_start_diff_s==null?'–':a.mean_start_diff_s+'s'}</b><span class="m">mean start-time gap</span></div>`+
    `<div><b>${a.n_steps}</b><span class="m">steps compared</span></div></div>`;}
  cols.innerHTML='';
  d.engines.forEach(k=>{const div=document.createElement('div');div.innerHTML=renderEngine(k,d.results[k]);cols.appendChild(div);});
 }catch(err){cols.innerHTML='⚠️ '+esc(''+err);}
 b.disabled=false;b.textContent='Review video';};
</script></body></html>"""


@app.route("/benchmark")
def benchmark_page():
    path = os.path.join(HERE, "benchmark_results.json")
    data = None
    if os.path.isfile(path):
        try:
            with open(path) as fh:
                data = json.load(fh)
        except Exception:
            data = None
    return render_template_string(BENCH_PAGE, data_json=json.dumps(data))


BENCH_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Claude vs VLM — Benchmark</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1100px;
      margin:34px auto;padding:0 20px;color:#17323a;background:#f7fafb}
 h1{font-size:23px;margin-bottom:2px} h2{font-size:16px;margin:0 0 12px}
 .sub{color:#5c7079;margin-top:0;font-size:14px} .nav{font-size:13px;margin:8px 0 0}
 .nav a{color:#0e7c86;text-decoration:none;font-weight:600} .nav a:hover{text-decoration:underline}
 .card{background:#fff;border-radius:12px;padding:20px;box-shadow:0 2px 10px rgba(11,32,39,.07);margin-top:16px}
 .meta{color:#5c7079;font-size:13px} code{background:#eef0f1;padding:2px 6px;border-radius:5px}
 table{border-collapse:collapse;width:100%;font-size:13px}
 th,td{text-align:left;padding:8px 10px;border-bottom:1px solid #eef4f5;vertical-align:top}
 th{color:#5c7079;font-weight:600} td:first-child{font-weight:600;white-space:nowrap}
 .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
 .bar{height:5px;background:#eef0f1;border-radius:3px;margin-top:3px;overflow:hidden;min-width:60px}
 .fill{height:100%}
 .vid{margin:14px 0;padding-top:12px;border-top:1px solid #eef4f5}
 .vt{font-weight:600;font-size:13px;margin-bottom:6px} .m{color:#9bb1b8;font-weight:400}
 .trk{display:flex;align-items:center;gap:10px;margin:3px 0}
 .lab{width:158px;font-size:11.5px;color:#5c7079;text-align:right;flex:none}
 .axis{position:relative;height:18px;flex:1;background:#f3f6f7;border-radius:4px}
 .seg{position:absolute;top:0;height:18px;border-radius:3px;color:#fff;font-size:10px;
      line-height:18px;text-align:center;overflow:hidden}
</style></head><body>
<h1>📊 Claude vs VLM — Step-Localization Benchmark</h1>
<p class="sub">Both engines get the same protocol + sampled frames; predicted per-step time-ranges are
scored against BioVL-QR ground-truth timestamps.</p>
<p class="nav"><a href="/">← Back to the live comparison</a></p>
<div id="root"></div>
<script>
const DATA = {{ data_json|safe }};
const root=document.getElementById('root');
const ECPAL=['#5b54c9','#e07b39','#3a9a6b','#b3508a','#9a6b15'];
function ecolor(e,i){return e.toLowerCase().includes('claude')?'#0e7c86':ECPAL[i%ECPAL.length];}
const PAL=['#0e7c86','#e07b39','#5b54c9','#3a9a6b','#b3508a','#9a6b15','#3a7a86','#a9701b','#7a8a90','#b3261e'];
function pct(x){return x==null?'–':Math.round(x*100)+'%';}
function trackHtml(label,segs,dur){
 let s=`<div class="trk"><div class="lab">${label}</div><div class="axis">`;
 segs.forEach(g=>{const left=100*g.s/dur,w=Math.max(1.3,100*(g.e-g.s)/dur);
  s+=`<div class="seg" title="step ${g.i+1}: ${g.s}-${g.e}s" style="left:${left}%;width:${w}%;background:${PAL[g.i%PAL.length]}">${g.i+1}</div>`;});
 return s+`</div></div>`;
}
if(!DATA){
 root.innerHTML='<div class="card">No benchmark results yet. Generate them:<br><br><code>ANTHROPIC_API_KEY_FILE=anthropic_key.txt python benchmark.py --per-cat 1 --frames 24</code></div>';
}else{
 const A=DATA.aggregate,cfg=DATA.config,eng=Object.keys(A);
 const METR=[['mean_iou','Temporal IoU',1],['start_within_10s','Start ±10s',1],['start_within_5s','Start ±5s',1],['localized_frac','Localized',1],['ordering_acc','Ordering',1],['latency_s','Latency (s)',0]];
 let h=`<div class="card"><div class="meta"><b>${cfg.claude_model}</b> vs <b>${cfg.vlm_model}</b> · ${cfg.frames} frames/clip · ${A[eng[0]].n_videos} videos · categories: ${cfg.categories.join(', ')}</div></div>`;
 h+='<div class="card"><h2>Leaderboard — vs BioVL-QR ground truth</h2><table><tr><th>Engine</th>'+METR.map(m=>`<th>${m[1]}</th>`).join('')+'</tr>';
 eng.forEach((e,ei)=>{const ec=ecolor(e,ei);h+=`<tr><td><span class="dot" style="background:${ec}"></span>${e}</td>`;
  METR.forEach(m=>{const v=A[e][m[0]];const disp=m[2]?pct(v):(v==null?'–':v);
   const bar=(m[2]&&v!=null)?`<div class="bar"><div class="fill" style="width:${Math.round(v*100)}%;background:${ec}"></div></div>`:'';
   h+=`<td>${disp}${bar}</td>`;});
  h+='</tr>';});
 h+='</table><div class="meta" style="margin-top:8px">Higher is better except latency. IoU = overlap of predicted vs true step time-range.</div></div>';
 if(DATA.by_category){
  h+='<div class="card"><h2>By category — IoU / Start ±10s</h2><table><tr><th>Category</th>'+eng.map(e=>`<th>${e}</th>`).join('')+'</tr>';
  cfg.categories.forEach(cat=>{h+=`<tr><td>${cat}</td>`;
   eng.forEach(e=>{const c=(DATA.by_category[e]||{})[cat];h+=`<td>${c?pct(c.mean_iou)+' / '+pct(c.start_within_10s):'–'}</td>`;});
   h+='</tr>';});
  h+='</table></div>';
 }
 const byVid={};
 DATA.runs.forEach(r=>{if(!byVid[r.video])byVid[r.video]={category:r.category,engines:{}};byVid[r.video].engines[r.engine]=r;});
 h+='<div class="card"><h2>Per-video timelines — ground truth vs predictions</h2>';
 Object.keys(byVid).forEach(vid=>{const v=byVid[vid],any=Object.values(v.engines)[0],gt=any.perstep.map(p=>p.gt);
  let dur=0;gt.forEach(g=>dur=Math.max(dur,g[1]));
  Object.values(v.engines).forEach(r=>r.perstep.forEach(p=>{if(p.pred)dur=Math.max(dur,p.pred[1]);}));dur=dur||1;
  h+=`<div class="vid"><div class="vt">${v.category} / ${vid} <span class="m">(${gt.length} steps · ${Math.round(dur)}s)</span></div>`;
  h+=trackHtml('ground truth',gt.map((g,i)=>({i:i,s:g[0],e:g[1]})),dur);
  eng.forEach(e=>{const r=v.engines[e];if(!r)return;
   const segs=r.perstep.map((p,i)=>p.pred?{i:i,s:p.pred[0],e:p.pred[1]}:null).filter(Boolean);
   h+=trackHtml(e+' · IoU '+pct(r.mean_iou),segs,dur);});
  h+='</div>';});
 h+='</div>';
 root.innerHTML=h;
}
</script></body></html>"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
