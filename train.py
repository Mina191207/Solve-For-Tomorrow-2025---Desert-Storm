import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ======================
# CONFIG
# ======================
SAVE_DIR   = "SFT25/dataset_multi"
CSV_PATH   = os.path.join(SAVE_DIR, "labels.csv")
MODEL_DIR  = "SFT25/model_multi"
TFLITE_PATH = os.path.join(MODEL_DIR, "gaze_multi.tflite")
os.makedirs(MODEL_DIR, exist_ok=True)

LEFT_EYE_SIZE  = (64, 64)
RIGHT_EYE_SIZE = (64, 64)
FACE_SIZE      = (128, 128)

BATCH_SIZE = 32
EPOCHS = 20
VAL_RATIO = 0.15

# ======================
# LOAD CSV
# ======================
print("[INFO] Loading:", CSV_PATH)
df = pd.read_csv(CSV_PATH)
print("[DEBUG] Raw Columns:", df.columns.tolist())

# Normalize column names
rename_map = {
    "fname_left": "left_path",
    "fname_right": "right_path",
    "fname_face": "face_path",
    "gx": "gx",
    "gy": "gy"
}
df = df.rename(columns=rename_map)

print("[DEBUG] Renamed Columns:", df.columns.tolist())
assert {"left_path","right_path","face_path","gx","gy"}.issubset(df.columns)


# Clean missing files
def file_exists(p):
    return os.path.exists(p) and os.path.getsize(p) > 0

mask = df[["left_path", "right_path", "face_path"]].applymap(file_exists).all(axis=1)
missing = len(df) - mask.sum()
if missing > 0:
    print(f"[WARN] Dropping {missing} rows with missing files")
df = df[mask].reset_index(drop=True)

# Shuffle + split
df = df.sample(frac=1.0, random_state=2025).reset_index(drop=True)
n_val = max(1, int(len(df) * VAL_RATIO))
val_df = df.iloc[:n_val]
train_df = df.iloc[n_val:]
print(f"[INFO] Train={len(train_df)}, Val={len(val_df)}")

# ======================
# TF.DATA PIPELINES
# ======================
def parse_row(left, right, face, gx, gy):
    def _load(path, size):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, size)
        img = tf.cast(img, tf.float32) / 255.0
        return img
    l = _load(left, LEFT_EYE_SIZE)
    r = _load(right, RIGHT_EYE_SIZE)
    f = _load(face, FACE_SIZE)
    return (l, r, f), tf.stack([gx, gy])

def make_ds(dataframe, batch_size, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((
        dataframe.left_path.values,
        dataframe.right_path.values,
        dataframe.face_path.values,
        dataframe.gx.astype(np.float32).values,
        dataframe.gy.astype(np.float32).values,
    ))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe), reshuffle_each_iteration=True)
    ds = ds.map(parse_row, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_ds(train_df, BATCH_SIZE, shuffle=True)
val_ds   = make_ds(val_df, BATCH_SIZE, shuffle=False)

# ======================
# MODEL
# ======================
def branch(input_shape):
    x_in = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x_in)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(96, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    return x_in, x

l_in, l_feat = branch((*LEFT_EYE_SIZE, 3))
r_in, r_feat = branch((*RIGHT_EYE_SIZE, 3))
f_in, f_feat = branch((*FACE_SIZE, 3))

x = layers.Concatenate()([l_feat, r_feat, f_feat])
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
out = layers.Dense(2, activation="sigmoid")(x)  # predict normalized gx, gy

model = keras.Model([l_in, r_in, f_in], out)
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
model.summary()

# ======================
# TRAINING
# ======================
steps_per_epoch = max(1, len(train_df)//BATCH_SIZE)
val_steps = max(1, len(val_df)//BATCH_SIZE)

checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "best_model.keras"),
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    callbacks=[checkpoint_cb]
)

# Save final keras model
model.save(os.path.join(MODEL_DIR, "final_model.keras"))
print("[SAVE] Keras model saved to", MODEL_DIR)

# ======================
# TFLITE EXPORT
# ======================
print("[INFO] Converting to TFLite…")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tfl = converter.convert()
with open(TFLITE_PATH, "wb") as f:
    f.write(tfl)
print("[SAVE]", TFLITE_PATH)

# Quantized version
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    q = converter.convert()
    qpath = TFLITE_PATH.replace(".tflite", ".int8.tflite")
    with open(qpath, "wb") as f:
        f.write(q)
    print("[SAVE]", qpath)
except Exception as e:
    print("[WARN] Quantization failed:", e)
