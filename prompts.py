"""Shared prompt text so Claude and the VLM get the IDENTICAL task spec — the only
difference is transport (Claude is constrained by a JSON schema; the VLM gets a
JSON-format reminder appended). This keeps the benchmark a fair comparison."""


def build_header(steps, n_frames):
    proto = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    return (
        "You are assisting a human who is reviewing whether a lab protocol was followed in a video. "
        "You are NOT issuing a pass/fail verdict — you give a confidence estimate per step and raise "
        "gentle, suggestive warnings worth a quick human look.\n\n"
        "PROTOCOL:\n" + proto + "\n\n"
        f"Below are {n_frames} frames sampled in chronological order from the video, each "
        "preceded by its index and timestamp. Treat them as one continuous clip."
    )


INSTRUCTION = (
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
    "- start_time_s / end_time_s / best_frame_index: the time-range and the [frame N] index where "
    "this step is visible. ALWAYS fill these using the frame timestamps shown above by picking the "
    "best-matching frame(s); use best_frame_index -1 (and 0/0) ONLY if the step is genuinely not "
    "visible in ANY frame.\n"
    "- note: brief, plain description of what you saw for this step.\n\n"
    "Be helpful, not pedantic: reserve flags for real concerns. Return every step."
)

# Appended only for the VLM (which has no schema-constrained output mode).
# The example uses realistic NON-zero localization values so the model fills the time
# fields from the frame timestamps rather than copying placeholder zeros.
VLM_JSON_SUFFIX = (
    '\n\nReturn ONLY JSON, no prose, of this exact form (fill start_time_s/end_time_s/'
    'best_frame_index with the real frame timestamps where you see each step): '
    '{"observed_summary": "...", "steps": [{"step_number": 1, "step_text": "...", '
    '"confidence": "high", "flag": false, "warning": "", "start_time_s": 12.0, '
    '"end_time_s": 31.5, "best_frame_index": 3, "note": "..."}]}'
)
