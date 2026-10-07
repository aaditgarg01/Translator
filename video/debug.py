"""Optional OpenCV evidence overlay. Motion scores are not probabilities."""
import cv2
import numpy as np


def draw_debug(frame, faces, state, vision_ms, result_age_ms, scene=None):
    for face in faces:
        landmarks = face.get('landmarks')
        if landmarks is not None:
            for index, point in enumerate(landmarks):
                color = (0, 240, 255) if index >= 48 else (200, 140, 70)
                cv2.circle(frame, tuple(np.rint(point).astype(int)), 2, color, -1)
            cv2.polylines(frame, [np.rint(landmarks[48:60]).astype('int32')], True, (0, 240, 255), 1)
        for before, after in zip(face.get('flow_from', []), face.get('flow_to', [])):
            cv2.arrowedLine(frame, tuple(np.rint(before).astype(int)), tuple(np.rint(after).astype(int)),
                           (80, 240, 90), 1, tipLength=.3)
        x, y, _, _ = map(int, face['bbox'])
        cv2.putText(frame, f"#{face.get('track_id', '?')} motion {face.get('speaker_score', 0):.3f}",
                    (max(0,x), max(15,y-30)), cv2.FONT_HERSHEY_SIMPLEX, .45, (0,240,255), 1)
    lines = ['VISION DEBUG (D to hide, S to save)',
             f'Preview: {state.fps:.1f} FPS | faces: {len(faces)}',
             f'Vision: {vision_ms:.1f} ms | result age: {result_age_ms:.0f} ms',
             f'Audio VAD: {"speech" if state.speech_active else "quiet"}',
             f'Speaker: {state.current_speaker or "uncertain / off camera"}']
    for stage, elapsed in state.get_latencies().items():
        lines.append(f'{stage}: {elapsed:.0f} ms (last completed)')
    if scene:
        lines += [f'OCR: {scene.ocr_ms:.0f} ms | regions: {len(scene.last_regions)}',
                  f'Camera translation: {scene.translation_ms:.0f} ms']
        factor = frame.shape[1] / 640
        for region in scene.last_regions:
            x, y = (region['polygon'][0] * factor).astype(int)
            cv2.putText(frame, f"det {region['confidence']:.2f}", (max(0,x),max(15,y)),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, (255,220,0), 1)
    width = min(440, frame.shape[1]-20)
    top = 135 if frame.shape[0] > 400 else 115
    height = min(len(lines)*20+16, frame.shape[0]-top-8)
    if height < 20 or width < 20:
        return
    region = frame[top:top+height, 10:10+width]
    cv2.addWeighted(region, .2, np.zeros_like(region), .8, 0, region)
    for i, line in enumerate(lines):
        if (i+1)*20 > height:
            break
        cv2.putText(frame, line, (18,top+20+i*20), cv2.FONT_HERSHEY_SIMPLEX, .43, (225,240,240), 1)
