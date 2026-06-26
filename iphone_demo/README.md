# ObjectDetectDemo

Lightweight iPhone camera demo that runs on-device Vision detection and draws live bounding boxes with confidence-colored labels.

Detected classes:

- `person` via `VNDetectHumanRectanglesRequest`
- `cat` and `dog` via `VNRecognizeAnimalsRequest`
- `hand` via `VNDetectHumanHandPoseRequest`
- common lab/debug targets via `VNClassifyImageRequest` plus objectness saliency boxes:
  - `beaker`
  - `bottle`
  - `pipette`
  - `microscope`
  - `plate`
  - `rack`
  - `glove`
  - `mask`
  - `laptop`
  - `phone`

Box colors interpolate from red at lower confidence to green at higher confidence. Debug builds show FPS only in the top-right overlay.

The app also runs a lightweight Metal compute benchmark that performs parallel matrix multiplication through Metal Performance Shaders. On an iPhone 13, this uses the phone's Apple GPU through `MTLCreateSystemDefaultDevice()` and MPS' optimized matrix kernels.

Camera/GPU performance notes:

- The camera session prefers a 1280x720 60 FPS format when the device exposes one.
- Video frames are delivered as bi-planar YUV instead of BGRA to reduce capture pipeline bandwidth.
- The FPS overlay measures camera frame delivery, not Vision detection completions.
- Vision detection is decoupled from camera delivery, sampled with frame skipping, and limited to one in-flight frame so preview can stay near 60 FPS.
- Vision requests prefer Apple Neural Engine compute stages when supported by the request, then fall back to GPU.
- Person, animal, classification, and saliency Vision requests are batched through one image handler to reduce repeated image preprocessing.
- The Metal matrix multiply uses `MPSMatrixMultiplication` and schedules follow-up work asynchronously so it competes less with camera preview.

## Protocols

Protocols are **scraped at launch** from `.txt` files bundled in the `Protocols/` folder
(`ProtocolLibrary.loadAll()`), so adding a protocol is just dropping a file into that folder —
no code changes. The drop-down lists every protocol it finds. Currently bundled:

- `Automated protein synthesis` (`PCR Mutagenesis Setup`)
- `Automated liquid handling` (`Liquid Transfer Protocol`)
- `High throughput stem cell culturing` (`Lentiviral infection of iPSC Fibroblast`)

File format (leading metadata lines are optional):

```
Title: Automated protein synthesis   # drop-down name; defaults to the heading line
Order: 1                             # sort position in the drop-down; defaults to the end
PCR Mutagenesis Setup                # heading shown at the top of the protocol panel

1. First step...
2. Second step...
```

Numbered lines become steps and are renumbered sequentially, so duplicate or skipped numbers
in the source file are tolerated. The folder is a copy of the repo-level `protocols/` directory.

The top-left transparent protocol dropdown shows only the selected protocol name.
Swipe from the left edge to reveal the selected protocol. Swipe the protocol panel left to hide it.

Protocol panel behavior:

- The current step is highlighted.
- Tap any step to select it; tapping a step with a memo opens the memo.
- `Done` marks the selected step green and advances the current step.
- `Flag` reveals `Uncertain` and `Appears missed`; those mark the selected step yellow or red.
- Completed or flagged steps are darkened and show their status color marker.
- `Add memo` stores a note on the selected step.
- `Jump to current step` jumps back to the active step.
- `Reset progress` restarts the selected protocol at step one and clears completed steps, flags, and memos.
- `Boxes: On/Off` toggles object detection bounding boxes.

Scan status now appears in a top-right box between FPS and the bounding-box toggle. The bottom bar shows the current protocol step on one line with truncation when needed.

## Recording experiments

A circular record button sits at the bottom-center of the camera screen.

- Tapping it starts recording the camera feed (with audio) via `AVCaptureMovieFileOutput`, which runs alongside the live Vision detection on the same capture session.
- While recording, every panel/overlay control (protocol dropdown, box toggle, step actions, and the `Past` button) is dimmed and disabled so the recording stays clean, and a red `● m:ss` timer appears at the top. The record button morphs into a stop square.
- Tapping again stops recording. The video is saved into the app's `Documents/Experiments` directory, registered in `ExperimentStore`, and (if the user grants access) also copied to the system photo library.

## Past experiments

The `Past` button at the bottom-right opens the **Past Experiments** page (`PastExperimentsViewController`):

- Experiments are **grouped into sections by protocol**; each row is titled by its recording date (with the time as a subtitle) and shows a thumbnail and a duration badge.
- Tap a row to play the video full-screen with `AVPlayerViewController`.
- Swipe a row to delete it.
- The photo-library button in the nav bar imports an existing video (via `PHPickerViewController`); on import you pick which protocol it belongs to so it files into the right section.

Persistence is on-device only for now. `Experiment` carries `inferenceRuns` and `isUploaded` fields so a future uploader can POST each video + metadata to the web server and record inference results against it — that server integration is intentionally not implemented yet.

Open `ObjectDetectDemo.xcodeproj` in Xcode, select a physical iPhone as the run target, and run. The iOS Simulator does not provide a real camera feed for this workflow.

The lab-object and laptop path is a lightweight fallback: Apple's built-in Vision APIs classify the frame and use saliency for approximate boxes. For real per-object lab boxes, replace `detectTargetObjectsWithSaliency(in:)` with a `VNCoreMLRequest` backed by a detector `.mlmodel`.
