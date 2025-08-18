import os
import cv2
import time
import csv
import uuid
import numpy as np
import mediapipe as mp
import pyautogui

# ============== Config ==============
SAVE_DIR = "SFT25/dataset_multi"
IMG_LEFT  = os.path.join(SAVE_DIR, "left")
IMG_RIGHT = os.path.join(SAVE_DIR, "right")
IMG_FACE  = os.path.join(SAVE_DIR, "face")
CSV_PATH  = os.path.join(SAVE_DIR, "labels.csv")

LEFT_EYE_SIZE  = (64, 64)
RIGHT_EYE_SIZE = (64, 64)
FACE_SIZE      = (128, 128)

GRID_COLS = 5
GRID_ROWS = 3
DWELL_SEC = 0.9   # Thời gian cần nhìn 1 điểm
PAUSE_SEC = 0.25  # Nghỉ giữa các target

# ============== Prepare FS ==============
for d in [SAVE_DIR, IMG_LEFT, IMG_RIGHT, IMG_FACE]:
    os.makedirs(d, exist_ok=True)

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["left_path", "right_path", "face_path", "gx", "gy"])

# ============== FaceMesh ==============
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
    if len(pts) == 0:
        return None
    x, y, wc, hc = cv2.boundingRect(np.array(pts))
    pad = int(max(wc, hc) * 0.4)
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(w, x + wc + pad); y1 = min(h, y + hc + pad)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return cv2.resize(crop, out_size)

def crop_from_landmarks(frame, landmarks, idxs, out_size):
    h, w = frame.shape[:2]
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in idxs]
    return crop_rect(frame, pts, out_size)

# ============== Targets (grid points) ==============
screen_w, screen_h = pyautogui.size()
xs = np.linspace(int(screen_w*0.1), int(screen_w*0.9), GRID_COLS)
ys = np.linspace(int(screen_h*0.15), int(screen_h*0.85), GRID_ROWS)
targets = [(int(x), int(y)) for y in ys for x in xs]
rng = np.random.default_rng(2025); rng.shuffle(targets)

# ============== Capture ==============
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

cv2.namedWindow("collector", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("collector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty("collector", cv2.WND_PROP_TOPMOST, 1)

print("[INFO] Press 'q' to quit.")

with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    for (tx, ty) in targets:
        t0 = time.time()
        while time.time() - t0 < DWELL_SEC + 2.0:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            # Hiển thị dot di chuyển tại target (tx, ty)
            disp = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            cv2.circle(disp, (tx, ty), 20, (0,255,0), -1)
            cv2.putText(disp, f"Look at the green dot", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
            cv2.imshow("collector", disp)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release(); cv2.destroyAllWindows(); raise SystemExit

            if not res.multi_face_landmarks:
                continue

            lms = res.multi_face_landmarks[0].landmark
            left  = crop_from_landmarks(frame, lms, LEFT_IDX,  LEFT_EYE_SIZE)
            right = crop_from_landmarks(frame, lms, RIGHT_IDX, RIGHT_EYE_SIZE)

            # Full face
            h, w = frame.shape[:2]
            xs_all = [int(l.x*w) for l in lms]
            ys_all = [int(l.y*h) for l in lms]
            x0, y0 = max(0, min(xs_all)), max(0, min(ys_all))
            x1, y1 = min(w, max(xs_all)), min(h, max(ys_all))
            pad = int(max(x1-x0, y1-y0) * 0.15)
            x0, y0 = max(0, x0-pad), max(0, y0-pad)
            x1, y1 = min(w, x1+pad), min(h, y1+pad)
            face = frame[y0:y1, x0:x1]
            if face.size:
                face = cv2.resize(face, FACE_SIZE)
            else:
                face = None

            if left is not None and right is not None and face is not None:
                if time.time() - t0 >= DWELL_SEC:
                    gx = tx / screen_w
                    gy = ty / screen_h
                    uid = uuid.uuid4().hex
                    lp = os.path.join(IMG_LEFT,  f"{uid}.jpg")
                    rp = os.path.join(IMG_RIGHT, f"{uid}.jpg")
                    fp = os.path.join(IMG_FACE,  f"{uid}.jpg")
                    cv2.imwrite(lp, left)
                    cv2.imwrite(rp, right)
                    cv2.imwrite(fp, face)
                    writer.writerow([lp, rp, fp, f"{gx:.6f}", f"{gy:.6f}"])
                    f.flush()
                    print("[CAPTURE]", uid, gx, gy)
                    break
        time.sleep(PAUSE_SEC)

cap.release()
cv2.destroyAllWindows()
print("[DONE] Data appended to", CSV_PATH)
