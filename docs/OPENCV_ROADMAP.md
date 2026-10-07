# OpenCV roadmap: Vision-Assisted Multilingual Interpreter

Implementation status: all four requested features are implemented: visual speaker detection,
camera text translation, participant captions, and the debug view. Gesture
controls and a YuNet/SFace upgrade remain optional future work. See the README for setup and limitations.

## What the project currently does with OpenCV

- Captures camera frames and displays a live overlay.
- Detects frontal faces with a Haar cascade.
- Registers and recognizes people using LBPH.
- Attributes speech using LBF mouth landmarks, compensated optical flow and audio VAD.
- Renders face boxes, measured FPS and Unicode captions anchored to participants.

The previous largest-face heuristic can be recovered from earlier Git revisions
as an experimental baseline for labeled-clip comparisons.

## 1. Visual active-speaker detection — highest priority

Detect and track each face, locate mouth landmarks, then measure mouth opening
and local optical flow over a short time window. Subtract motion estimated from
stable cheek/nose points so head movement is not confused with speech. Combine
that score with audio VAD and return "uncertain" when evidence is weak.

OpenCV work: `cv2.face` landmark fitting with a compatible pretrained model,
`goodFeaturesToTrack`, `calcOpticalFlowPyrLK`, image warping, temporal smoothing
and track association. YuNet can improve face detection, but its five landmarks
alone are not a complete mouth contour; a mouth landmark model is still needed.

Demo: two people exchange turns while the larger/closer person stays silent.
Show mouth landmarks, motion arrows and speaker confidence. Compare against the
current largest-face baseline on manually labeled clips. Report speaker accuracy,
false activations, decision latency and preview FPS, including head movement,
occlusion, low light and simultaneous speakers. This is speaker attribution,
not lip reading or audio separation.

## 2. Translate text seen by the camera — strongest second feature

Point the camera at a sign, menu or printed card. Detect the text region, rectify
its perspective, recognize it, translate it, and anchor the translated overlay
to the region as the camera moves. Translate only when recognized text changes.

OpenCV work: DB/EAST text detection through `cv2.dnn`, `getPerspectiveTransform`,
`warpPerspective`, contrast enhancement, `dnn_TextRecognitionModel` and optical
flow tracking. OCR models need appropriate character sets; start with printed
Latin text and explicitly list tested scripts. OCR and translation remain
separate stages. Render non-Latin output using a Unicode-capable text renderer.

Demo: a tilted French/German sign becomes an English/Hindi/Japanese overlay.
Evaluate character error rate, polygon stability, translation latency and FPS.

## 3. Persistent participant IDs and anchored captions

Use periodic face detection plus lightweight tracking between detections.
Associate tracks across frames and attach each caption to the selected speaker.
YuNet/SFace can replace the current Haar/LBPH approach after measuring accuracy.
Add Unicode captions so all spoken languages are visible too.

Evaluate identity switches and caption assignment during crossings/occlusions.
Mark an uncertain identity instead of assigning an arbitrary registered name.
This is valuable infrastructure for feature 1, rather than a separate flashy demo.

## 4. Gesture controls and a vision debug view — optional polish

A deliberately held gesture can pause translation or repeat the last caption.
Use a compatible hand detector/landmark ONNX model through OpenCV DNN, then
geometry and a time-based debounce. Avoid presenting a few gestures as sign
language recognition. Show confirmation before changing state.

Add a toggleable debug view showing face/mouth regions, tracking points, motion
scores, OCR quadrilaterals, stage latency and actual FPS. This makes the use of
computer vision visible and gives the presentation measurable evidence.

## Suggested scope

Build features 1 and 2 first, with stable tracks and a debug view supporting them.
Keep heavy face/OCR inference on workers that consume the newest frame; never
queue every camera frame behind a slow model. Detect on smaller images, track
between detections, and keep display on the main thread. Treat 30 FPS as a
measured hardware target, not a guaranteed outcome.

A strong final presentation shows an ablation: largest-face attribution versus
mouth-motion attribution, and OCR with versus without perspective correction.
Include failure cases, model licenses, and separate results for macOS and Windows.

## Official implementation references

- [Face landmarks](https://docs.opencv.org/4.5.1/d2/d42/tutorial_face_landmark_detection_in_an_image.html)
- [Optical flow](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)
- [Text detection and recognition](https://docs.opencv.org/4.x/d4/d43/tutorial_dnn_text_spotting.html)
- [YuNet/SFace detection and recognition](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html)
