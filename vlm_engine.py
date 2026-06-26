"""Open-source VLM (Qwen2.5-VL) engine that mirrors app.py's Claude align() — same schema,
so Claude vs VLM can be shown side-by-side. Lazy-loads the model on first use."""
import base64, io, json, re, threading
from PIL import Image

import prompts

VLM_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
VLM_MAX_FRAMES = 10          # keep vision tokens in T4 memory budget
VLM_FRAME_PX = 640           # resize long side before sending
VLM_MAX_NEW_TOKENS = 2000

_model = None
_processor = None
_lock = threading.Lock()


def vlm_name():
    return VLM_MODEL.split("/")[-1]


def _load():
    global _model, _processor
    if _model is None:
        with _lock:
            if _model is None:
                import torch
                from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
                _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    VLM_MODEL, torch_dtype=torch.float16, device_map="cuda")
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
