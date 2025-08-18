import os, cv2, time, numpy as np, mediapipe as mp, pyautogui
from datetime import datetime
import pandas as pd

# ---------------- CONFIG ----------------
MODEL_PATH   = "SFT25/model_multi/gaze_multi.int8.tflite"
DATASET_DIR  = "SFT25/dataset_multi"
CSV_PATH     = os.path.join(DATASET_DIR, "labels.csv")
os.makedirs(DATASET_DIR, exist_ok=True)

LEFT_EYE_SIZE  = (64, 64)
RIGHT_EYE_SIZE = (64, 64)
FACE_SIZE      = (128, 128)

SMOOTHING   = 0.25
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

# ---------------- Helpers ----------------
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

def save_sample(left, right, face, gx, gy):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    lp = os.path.join(DATASET_DIR, f"left_{ts}.jpg")
    rp = os.path.join(DATASET_DIR, f"right_{ts}.jpg")
    fp = os.path.join(DATASET_DIR, f"face_{ts}.jpg")
    for arr, path in [(left, lp), (right, rp), (face, fp)]:
        arr = (arr*255).astype(np.uint8)
        cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    row = {"left_path": lp, "right_path": rp, "face_path": fp, "gx": gx, "gy": gy}
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(CSV_PATH, index=False)

# ---------------- TFLite ----------------
try:
    from tflite_runtime.interpreter import Interpreter
    print("[INFO] Using tflite_runtime.Interpreter")
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter
    print("[INFO] Using tensorflow.lite.Interpreter")

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_crop(crop, info):
    h, w = info["shape"][1:3]
    if crop.shape[:2] != (h, w):
        crop = cv2.resize(crop, (w, h))
    return np.expand_dims(crop, 0).astype(np.float32)

def predict_tflite(left, right, face):
    for inp, info in zip([left, right, face], input_details):
        arr = preprocess_crop(inp, info)
        interpreter.set_tensor(info["index"], arr)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])[0]

# ---------------- Calibration ----------------
def generate_calib_points(sw, sh, nx=4, ny=4):
    return [(int(x*sw/(nx+1)), int(y*sh/(ny+1))) for y in range(1,ny+1) for x in range(1,nx+1)]

# ---------------- Main ----------------
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

screen_w, screen_h = pyautogui.size()
ema_x, ema_y = 0.5, 0.5

cv2.namedWindow("interact", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("interact", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

mode = "tflite"   # or "iris"
calib_mode = False
calib_points = generate_calib_points(screen_w, screen_h, 4, 4)
calib_idx = 0
iris_samples, screen_samples = [], []
H = None
point_timer = None

print("[INFO] Press N = start calibration, T = toggle model, Q = quit.")

while True:
    ok, frame = cap.read()
    if not ok: break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    disp = frame.copy()
    h, w = disp.shape[:2]
    debug_panel = np.zeros((200, 600, 3), dtype=np.uint8)

    if res.multi_face_landmarks:
        lms = res.multi_face_landmarks[0].landmark
        left_crop  = crop_from_landmarks(frame, lms, LEFT_IDX, LEFT_EYE_SIZE)
        right_crop = crop_from_landmarks(frame, lms, RIGHT_IDX, RIGHT_EYE_SIZE)
        face_crop  = crop_face(frame, lms, FACE_SIZE)

        # --- Calibration ---
        if calib_mode and calib_idx < len(calib_points):
            px, py = calib_points[calib_idx]
            px = int(px / screen_w * w)
            py = int(py / screen_h * h)
            cv2.circle(disp, (px, py), 20, (0,0,255), -1)
            cv2.putText(disp, f"Calib {calib_idx+1}/{len(calib_points)}",
                        (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

            if point_timer is None: point_timer = time.time()
            elif time.time() - point_timer > 2.0:
                iris = np.mean([(lms[i].x, lms[i].y) for i in [468, 473]], axis=0)
                iris_samples.append([iris[0]*screen_w, iris[1]*screen_h])
                screen_samples.append([px, py])
                print(f"[CALIB] Collected {calib_idx+1}/{len(calib_points)}")
                gx, gy = px/screen_w, py/screen_h
                save_sample(left_crop, right_crop, face_crop, gx, gy)
                calib_idx += 1
                point_timer = time.time()
                if calib_idx == len(calib_points):
                    H, _ = cv2.findHomography(np.array(iris_samples), np.array(screen_samples))
                    calib_mode = False
                    print("[INFO] Calibration complete!")

                # --- Normal mode ---
        else:
            if left_crop is not None and right_crop is not None and face_crop is not None:
                if mode == "tflite":
                    gx, gy = predict_tflite(left_crop, right_crop, face_crop)
                    gx = float(np.clip(gx, EDGE_MARGIN, 1-EDGE_MARGIN))
                    gy = float(np.clip(gy, EDGE_MARGIN, 1-EDGE_MARGIN))
                    ema_x = (1-SMOOTHING)*ema_x + SMOOTHING*gx
                    ema_y = (1-SMOOTHING)*ema_y + SMOOTHING*gy
                    cx, cy = int(ema_x*screen_w), int(ema_y*screen_h)

                    # --- TFLite Debug window ---
                    tflite_panel = np.zeros((120, 300, 3), dtype=np.uint8)
                    cv2.putText(tflite_panel, f"raw: {gx:.3f}, {gy:.3f}", (10,40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                    cv2.putText(tflite_panel, f"smooth: {ema_x:.3f}, {ema_y:.3f}", (10,80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv2.imshow("tflite_debug", tflite_panel)

                elif mode == "iris" and H is not None:
                    iris = np.mean([(lms[i].x, lms[i].y) for i in [468, 473]], axis=0)
                    iris_pt = np.array([[[iris[0]*screen_w, iris[1]*screen_h]]], dtype=np.float32)
                    mapped = cv2.perspectiveTransform(iris_pt, H)[0][0]
                    cx, cy = int(mapped[0]), int(mapped[1])
                else:
                    cx, cy = int(ema_x*screen_w), int(ema_y*screen_h)

                # --- always draw green dot ---
                cv2.circle(disp, (cx, cy), 12, (0,255,0), -1)

                # Debug crops
                def to_bgr(c, size):
                    arr = (c * 255).astype(np.uint8)
                    return cv2.resize(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), size)
                debug_panel[0:200, 0:200]   = to_bgr(left_crop,  (200,200))
                debug_panel[0:200, 200:400] = to_bgr(right_crop, (200,200))
                debug_panel[0:200, 400:600] = to_bgr(face_crop,  (200,200))

    cv2.imshow("interact", disp)
    cv2.imshow("debug", debug_panel)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'): break
    elif k == ord('t'):
        mode = "iris" if mode=="tflite" else "tflite"
        print("[INFO] Switched mode ->", mode)
    elif k == ord('n'):
        calib_mode = True
        calib_idx, iris_samples, screen_samples, point_timer = 0, [], [], None
        print("[INFO] Calibration started.")

cap.release()
cv2.destroyAllWindows()
