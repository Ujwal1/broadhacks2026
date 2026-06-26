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

The top-left transparent protocol dropdown shows only the selected protocol name and supports:

- `Automated protein synthesis`, shown as a swipe-in PCR mutagenesis setup protocol.
- `Automated liquid handling`, shown as a swipe-in `Liquid Transfer Protocol`.

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

Open `ObjectDetectDemo.xcodeproj` in Xcode, select a physical iPhone as the run target, and run. The iOS Simulator does not provide a real camera feed for this workflow.

The lab-object and laptop path is a lightweight fallback: Apple's built-in Vision APIs classify the frame and use saliency for approximate boxes. For real per-object lab boxes, replace `detectTargetObjectsWithSaliency(in:)` with a `VNCoreMLRequest` backed by a detector `.mlmodel`.
