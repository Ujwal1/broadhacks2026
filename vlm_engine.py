"""Open-source VLM (Qwen2.5-VL) engine that mirrors app.py's Claude align() — same schema,
so Claude vs VLM can be shown side-by-side. Lazy-loads the model on first use."""
import base64, io, json, os, re, threading
from PIL import Image

import prompts

# Model is configurable via env so we can A/B different VLMs without code changes.
# Default is the 3B: our benchmark found it BEATS the 7B-4bit here (higher IoU, 2x faster) —
# the bigger model in 4-bit was more conservative and slower. Try the 7B with
# VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct (auto-loads in 4-bit; needs bitsandbytes).
VLM_MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
VLM_4BIT = os.environ.get("VLM_4BIT", "auto")   # "1"/"0"/"auto" (auto -> quantize >=7B)
VLM_MAX_FRAMES = int(os.environ.get("VLM_MAX_FRAMES", "10"))   # keep vision tokens in GPU budget
VLM_FRAME_PX = 640           # resize long side before sending
VLM_MAX_NEW_TOKENS = 2000

_model = None
_processor = None
_lock = threading.Lock()


def _use_4bit():
    if VLM_4BIT.lower() in ("1", "true", "yes"):
        return True
    if VLM_4BIT.lower() in ("0", "false", "no"):
        return False
    return any(s in VLM_MODEL for s in ("7B", "8B", "13B", "32B", "72B"))  # auto


def vlm_name():
    return VLM_MODEL.split("/")[-1] + (" (4-bit)" if _use_4bit() else "")


def _load():
    global _model, _processor
    if _model is None:
        with _lock:
            if _model is None:
                import torch
                from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
                kw = dict(device_map="cuda", torch_dtype=torch.float16)
                if _use_4bit():
                    from transformers import BitsAndBytesConfig
                    kw["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
                _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(VLM_MODEL, **kw)
                _model.eval()
                _processor = AutoProcessor.from_pretrained(
                    VLM_MODEL, min_pixels=256 * 28 * 28, max_pixels=640 * 28 * 28)
    return _model, _processor


def _subsample(frames):
    if len(frames) <= VLM_MAX_FRAMES:
        fr = list(frames)
    else:
        idxs = [round(i * (len(frames) - 1) / (VLM_MAX_FRAMES - 1)) for i in range(VLM_MAX_FRAMES)]
        fr = [frames[i] for i in idxs]
    # re-index 0..k-1 so the model's best_frame_index maps directly back
    return [{"idx": i, "t": f["t"], "b64": f["b64"]} for i, f in enumerate(fr)]


def _to_pil(b64):
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    w, h = img.size
    if max(w, h) > VLM_FRAME_PX:
        s = VLM_FRAME_PX / max(w, h)
        img = img.resize((int(w * s), int(h * s)))
    return img


def _parse_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    a, b = text.find("{"), text.rfind("}")
    if a >= 0 and b > a:
        try:
            return json.loads(text[a:b + 1])
        except Exception:
            pass
    return None


def align_vlm(steps, frames):
    """Same contract as app.py align(): returns {observed_summary, n_flags, steps:[...with thumb]}."""
    model, processor = _load()
    import torch
    from qwen_vl_utils import process_vision_info

    fr = _subsample(frames)
    # IDENTICAL task spec to Claude (app.align) — only the JSON-format reminder is VLM-specific.
    header = prompts.build_header(steps, len(fr))
    instruction = prompts.INSTRUCTION + prompts.VLM_JSON_SUFFIX

    content = [{"type": "text", "text": header}]
    for f in fr:
        content.append({"type": "text", "text": f"[frame {f['idx']} | t={f['t']}s]"})
        content.append({"type": "image", "image": _to_pil(f["b64"])})
    content.append({"type": "text", "text": instruction})
    messages = [{"role": "user", "content": content}]

    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[chat], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=VLM_MAX_NEW_TOKENS, do_sample=False)
    trimmed = [g[len(i):] for i, g in zip(inputs.input_ids, gen)]
    out = processor.batch_decode(trimmed, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)[0]

    data = _parse_json(out)
    if not data:
        # graceful fallback so the UI still renders
        return {"observed_summary": "(VLM returned unparseable output)\n\n" + out[:1500],
                "n_flags": 0, "raw": out,
                "steps": [{"step_number": i + 1, "step_text": s, "confidence": "low",
                           "flag": False, "warning": "", "start_time_s": 0, "end_time_s": 0,
                           "best_frame_index": -1, "note": "", "thumb": None}
                          for i, s in enumerate(steps)]}

    steps_out = []
    for s in data.get("steps", []):
        bi = s.get("best_frame_index", -1)
        thumb = fr[bi]["b64"] if isinstance(bi, int) and 0 <= bi < len(fr) else None
        steps_out.append({**s, "thumb": thumb})
    n_flags = sum(1 for s in steps_out if s.get("flag"))
    return {"observed_summary": data.get("observed_summary", ""),
            "n_flags": n_flags, "steps": steps_out}


VLM_PF_MAX_FRAMES = int(os.environ.get("VLM_PF_MAX_FRAMES", "32"))


def _classify_frame(model, processor, proto, t, pil_img, n_steps):
    """Ask the VLM which single step ONE frame depicts. Returns step idx 0..n (0 = none)."""
    from qwen_vl_utils import process_vision_info
    import torch
    q = ("You are watching ONE frame from a video of a technician performing this lab protocol:\n"
         + proto + f"\n\nThis is the frame at t={t}s. Which SINGLE protocol step is being performed "
         f"in THIS frame? Reply with ONLY the step number (1-{n_steps}), or 0 if no step is clearly "
         "happening.")
    messages = [{"role": "user", "content": [{"type": "image", "image": pil_img},
                                             {"type": "text", "text": q}]}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[chat], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    out = processor.batch_decode([g[len(i):] for i, g in zip(inputs.input_ids, gen)],
                                 skip_special_tokens=True)[0]
    m = re.search(r"\d+", out)
    v = int(m.group()) if m else 0
    return v if 0 <= v <= n_steps else 0


def align_vlm_perframe(steps, frames):
    """Per-frame classification → aggregate into per-step time-ranges. Same schema as align().
    Plays to a VLM's single-image strength instead of one-shot multi-frame temporal reasoning."""
    model, processor = _load()
    # use as many frames as allowed (finer time resolution; memory is fine — 1 image per call)
    fr = frames if len(frames) <= VLM_PF_MAX_FRAMES else \
        [frames[round(i * (len(frames) - 1) / (VLM_PF_MAX_FRAMES - 1))] for i in range(VLM_PF_MAX_FRAMES)]
    fr = [{"idx": i, "t": f["t"], "b64": f["b64"]} for i, f in enumerate(fr)]
    proto = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))

    labels = []  # (frame_idx, t, step) per frame
    for f in fr:
        lab = _classify_frame(model, processor, proto, f["t"], _to_pil(f["b64"]), len(steps))
        labels.append((f["idx"], f["t"], lab))

    steps_out = []
    for k, s in enumerate(steps, start=1):
        hits = [(fi, t) for (fi, t, lab) in labels if lab == k]
        if hits:
            ts = [t for _, t in hits]
            mid = hits[len(hits) // 2]
            conf = "high" if len(hits) >= 3 else ("medium" if len(hits) == 2 else "low")
            steps_out.append({"step_number": k, "step_text": s, "confidence": conf,
                              "flag": False, "warning": "",
                              "start_time_s": min(ts), "end_time_s": max(ts),
                              "best_frame_index": mid[0], "note": f"{len(hits)} frame(s) match this step.",
                              "thumb": fr[mid[0]]["b64"]})
        else:
            steps_out.append({"step_number": k, "step_text": s, "confidence": "low",
                              "flag": True, "warning": "No frame was classified as this step — "
                              "it may have been skipped or not captured.", "start_time_s": 0,
                              "end_time_s": 0, "best_frame_index": -1,
                              "note": "Not detected in any sampled frame.", "thumb": None})
    n_flags = sum(1 for s in steps_out if s["flag"])
    matched = sum(1 for (_, _, lab) in labels if lab > 0)
    summary = (f"Per-frame classification of {len(fr)} frames: {matched} matched a protocol step, "
               f"{len(fr) - matched} matched none.")
    return {"observed_summary": summary, "n_flags": n_flags, "steps": steps_out}
