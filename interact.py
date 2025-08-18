import os
import cv2
import numpy as np
import mediapipe as mp
import pyautogui

# Prefer tflite_runtime for speed, fallback to TF
try:
    from tflite_runtime.interpreter import Interpreter
    print("[INFO] Using tflite_runtime.Interpreter")
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter
    print("[INFO] Using tensorflow.lite.Interpreter")

MODEL_PATH = "SFT25/model_multi/gaze_multi.tflite"   # adjust path
LEFT_EYE_SIZE  = (64, 64)
RIGHT_EYE_SIZE = (64, 64)
FACE_SIZE      = (128, 128)

SMOOTHING = 0.25
EDGE_MARGIN = 0.02

# ---------------- Mediapipe ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_IDX  = [33, 133, 159, 145, 468, 469, 470, 471]
RIGHT_IDX = [362, 263, 386, 374, 473, 474, 475, 476]

def crop_rect(frame, pts, out_size):
    h, w = frame.shape[:2]
    if not pts: return None
    x, y, wc, hc = cv2.boundingRect(np.array(pts))
    pad = int(max(wc, hc) * 0.4)
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(w, x + wc + pad); y1 = min(h, y + hc + pad)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0: return None
    crop = cv2.resize(crop, out_size)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return crop

def crop_from_landmarks(frame, landmarks, idxs, out_size):
    h, w = frame.shape[:2]
    pts = [(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in idxs]
    return crop_rect(frame, pts, out_size)

def crop_face(frame, landmarks, out_size):
    h, w = frame.shape[:2]
    xs = [int(l.x*w) for l in landmarks]
    ys = [int(l.y*h) for l in landmarks]
    x0, y0 = max(0, min(xs)), max(0, min(ys))
    x1, y1 = min(w, max(xs)), min(h, max(ys))
    pad = int(max(x1-x0, y1-y0) * 0.15)
    x0, y0 = max(0, x0-pad), max(0, y0-pad)
    x1, y1 = min(w, x1+pad), min(h, y1+pad)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0: return None
    crop = cv2.resize(crop, out_size)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return crop

# ---------------- TFLite setup ----------------
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_crop(crop, info):
    h, w = info["shape"][1:3]
    if crop.shape[:2] != (h, w):
        crop = cv2.resize(crop, (w, h))
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    crop = np.expand_dims(crop, 0).astype(np.float32)
    if info["dtype"] != np.float32:
        scale, zp = info.get("quantization", (1.0, 0))
        crop = np.round(crop / scale + zp).astype(info["dtype"])
    return crop

def predict(left, right, face):
    inputs = [left, right, face]
    for inp, info in zip(inputs, input_details):
        arr = preprocess_crop(inp, info)
        interpreter.set_tensor(info["index"], arr)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]["index"])[0]  # (2,)
    return out

# ---------------- Camera / UI ----------------
cap = cv2.VideoCapture(1)  # try 0 first
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

screen_w, screen_h = pyautogui.size()
ema_x, ema_y = 0.5, 0.5

cv2.namedWindow("interact", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("interact", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("[INFO] Press 'q' to quit. Press 'c' to center cursor.")

while True:
    

    ok, frame = cap.read()
    if not ok:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    debug_panel = np.zeros((200, 600, 3), dtype=np.uint8)  # debug overlay

    if res.multi_face_landmarks:
        lms = res.multi_face_landmarks[0].landmark
        left_crop  = crop_from_landmarks(frame, lms, LEFT_IDX,  LEFT_EYE_SIZE)
        right_crop = crop_from_landmarks(frame, lms, RIGHT_IDX, RIGHT_EYE_SIZE)
        face_crop  = crop_face(frame, lms, FACE_SIZE)

        if left_crop is not None and right_crop is not None and face_crop is not None:
            gx, gy = predict(left_crop, right_crop, face_crop)

            # Clamp + smooth
            m = EDGE_MARGIN
            gx = float(np.clip(gx, m, 1 - m))
            gy = float(np.clip(gy, m, 1 - m))
            ema_x = (1 - SMOOTHING) * ema_x + SMOOTHING * gx
            ema_y = (1 - SMOOTHING) * ema_y + SMOOTHING * gy
            print("Raw model output:", gx, gy)
            cx = int(ema_x * screen_w)
            cy = int(ema_y * screen_h)
            try:
                pyautogui.moveTo(cx, cy)
            except Exception:
                pass

            disp = frame.copy()
            cv2.circle(disp, (int(ema_x * disp.shape[1]), int(ema_y * disp.shape[0])), 6, (0, 255, 0), -1)
            cv2.putText(disp, f"({ema_x:.2f}, {ema_y:.2f})", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("interact", disp)

            # ---- Debug crops ----
            def to_bgr(crop, size):
                crop = (crop * 255).astype(np.uint8)
                return cv2.resize(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), size)

            debug_panel[0:200, 0:200]   = to_bgr(left_crop,  (200, 200))
            debug_panel[0:200, 200:400] = to_bgr(right_crop, (200, 200))
            debug_panel[0:200, 400:600] = to_bgr(face_crop,  (200, 200))

    cv2.imshow("debug", debug_panel)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    if k == ord('c'):
        ema_x, ema_y = 0.5, 0.5
        try:
            pyautogui.moveTo(int(screen_w * 0.5), int(screen_h * 0.5))
        except Exception:
            pass

cap.release()
cv2.destroyAllWindows()
