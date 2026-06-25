#!/usr/bin/env python
"""
Video → text web app: upload a video, Claude describes what's in it.

How it works: Claude's API takes images, not video — so we sample frames evenly
across the uploaded clip with OpenCV, send them (in chronological order) to Claude
as a single message, and return Claude's description.

API key: put it (one line) in  video_app/anthropic_key.txt  — the app reads it
automatically. ($ANTHROPIC_API_KEY env var, $ANTHROPIC_API_KEY_FILE, .anthropic_key,
and .env are also honored.)

Run:
    pip install flask anthropic opencv-python   # if not already in the env
    echo "sk-ant-..." > anthropic_key.txt && chmod 600 anthropic_key.txt   # your key
    python app.py                       # serves on http://0.0.0.0:7860
    # from your laptop:  ssh -L 7860:localhost:7860 <this-host>   then open localhost:7860
"""
import base64
import os
import tempfile

import cv2
import anthropic
from flask import Flask, request, jsonify, render_template_string

MODEL = "claude-opus-4-8"
MAX_FRAMES = 20          # hard cap on frames sent to the model
FRAME_WIDTH = 768        # downscale width (keeps payload + tokens reasonable)
PORT = int(os.environ.get("PORT", "7860"))

app = Flask(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
# Where to look for the key, in order. First match wins.
KEY_FILES = [os.environ.get("ANTHROPIC_API_KEY_FILE"),
             os.path.join(HERE, "anthropic_key.txt"),
             os.path.join(HERE, ".anthropic_key"),
             os.path.join(HERE, ".env")]


def load_api_key():
    """Return the API key from $ANTHROPIC_API_KEY, else from a secret file (see KEY_FILES)."""
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
            if "=" in line:                       # KEY=value form (e.g. a .env file)
                k, v = line.split("=", 1)
                if k.strip().upper().endswith("ANTHROPIC_API_KEY"):
                    return v.strip().strip('"').strip("'")
            else:                                  # bare key on its own line
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
                "or set the ANTHROPIC_API_KEY environment variable.")
        _client = anthropic.Anthropic(api_key=key)
    return _client


def sample_frames(path, n):
    """Return up to n evenly-spaced frames as base64 JPEG strings (chronological)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open this video (try an .mp4/.mov file).")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    n = max(1, min(n, MAX_FRAMES))
    idxs = [0] if total <= 1 else [round(i * (total - 1) / (n - 1)) for i in range(n)] if n > 1 else [0]

    frames = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        if w > FRAME_WIDTH:
            frame = cv2.resize(frame, (FRAME_WIDTH, int(h * FRAME_WIDTH / w)))
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            frames.append(base64.b64encode(buf).decode())
    cap.release()
    if not frames:
        raise ValueError("Could not read any frames from this video.")
    duration = (total / fps) if fps else 0
    return frames, total, duration


def describe(frames, question):
    """Send frames + prompt to Claude; return the text description."""
    instruction = (question.strip() if question and question.strip() else
                   "Describe what is happening in this video — the setting, the people or "
                   "objects, the actions and their order, and any notable events or visible text. "
                   "Give a clear, coherent account of the video's content.")
    prompt = (f"The following {len(frames)} images are frames sampled in chronological order "
              f"from a single video (first frame first, last frame last). "
              f"Treat them as one continuous clip, not separate pictures.\n\n{instruction}")

    content = [{"type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": f}}
               for f in frames]
    content.append({"type": "text", "text": prompt})

    resp = get_client().messages.create(
        model=MODEL,
        max_tokens=4000,
        thinking={"type": "adaptive"},   # frame-by-frame reasoning benefits from this
        messages=[{"role": "user", "content": content}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


@app.route("/")
def index():
    return render_template_string(PAGE, model=MODEL, max_frames=MAX_FRAMES)


@app.route("/analyze", methods=["POST"])
def analyze():
    f = request.files.get("video")
    if not f or not f.filename:
        return jsonify(error="No video uploaded."), 400
    try:
        n = int(request.form.get("frames", 10))
    except ValueError:
        n = 10
    question = request.form.get("question", "")

    suffix = os.path.splitext(f.filename)[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        f.save(tmp.name)
        tmp.close()
        frames, total, duration = sample_frames(tmp.name, n)
        text = describe(frames, question)
        return jsonify(text=text, frames=len(frames),
                       total_frames=total, duration=round(duration, 1))
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
<title>Video → Text (Claude)</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:760px;
      margin:40px auto;padding:0 20px;color:#17323a;background:#f7fafb}
 h1{font-size:24px;margin-bottom:4px} .sub{color:#5c7079;margin-top:0;font-size:14px}
 .card{background:#fff;border-radius:12px;padding:22px;box-shadow:0 2px 10px rgba(11,32,39,.07);margin-top:18px}
 label{display:block;font-weight:600;font-size:13px;margin:14px 0 4px}
 input[type=file],input[type=text],input[type=number]{width:100%;padding:9px;border:1px solid #d9e6e8;
      border-radius:8px;box-sizing:border-box;font-size:14px}
 .row{display:flex;gap:16px} .row>div{flex:1}
 button{margin-top:18px;background:#0e7c86;color:#fff;border:0;border-radius:8px;padding:11px 20px;
      font-size:15px;font-weight:600;cursor:pointer} button:disabled{opacity:.5;cursor:default}
 #out{white-space:pre-wrap;line-height:1.5;font-size:15px} #meta{color:#5c7079;font-size:12px;margin-bottom:10px}
 .spin{display:inline-block;width:15px;height:15px;border:2px solid #fff;border-top-color:transparent;
      border-radius:50%;animation:s .7s linear infinite;vertical-align:-2px;margin-right:7px}
 @keyframes s{to{transform:rotate(360deg)}}
</style></head><body>
<h1>🎬 Video → Text</h1>
<p class="sub">Upload a video; Claude (<b>{{model}}</b>) samples frames and describes what's in it.</p>
<div class="card">
 <form id="f">
  <label>Video file (.mp4 / .mov recommended)</label>
  <input type="file" name="video" accept="video/*" required>
  <div class="row">
   <div><label>Frames to sample (max {{max_frames}})</label>
        <input type="number" name="frames" value="10" min="1" max="{{max_frames}}"></div>
  </div>
  <label>Question (optional — leave blank for a general description)</label>
  <input type="text" name="question" placeholder="e.g. What procedure is being performed? Any mistakes?">
  <button id="b" type="submit">Analyze video</button>
 </form>
</div>
<div class="card" id="result" style="display:none">
 <div id="meta"></div><div id="out"></div>
</div>
<script>
const f=document.getElementById('f'),b=document.getElementById('b'),
      res=document.getElementById('result'),out=document.getElementById('out'),meta=document.getElementById('meta');
f.onsubmit=async e=>{e.preventDefault();
 b.disabled=true;b.innerHTML='<span class="spin"></span>Analyzing…';
 res.style.display='block';meta.textContent='';out.textContent='Sampling frames and asking Claude…';
 try{const r=await fetch('/analyze',{method:'POST',body:new FormData(f)});
  const d=await r.json();
  if(!r.ok){out.textContent='⚠️ '+(d.error||'Error');}
  else{meta.textContent=`${d.frames} frames sampled · ${d.total_frames} total frames · ~${d.duration}s`;
       out.textContent=d.text;}
 }catch(err){out.textContent='⚠️ '+err;}
 b.disabled=false;b.textContent='Analyze video';};
</script></body></html>"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
