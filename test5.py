import cv2
import mediapipe as mp
import numpy as np
import time
import os
import random

# --- KONFIGURASI ---
POSE_FOLDER = "poses"      # Folder berisi gambar pose target
THRESHOLD_SHOW = 5         # Jeda detik antar pose
NUM_POSES = 10             # Jumlah pose target
SIMILARITY_SCALE = 1000    # Skala perhitungan skor

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ==============================
# 🔹 FUNGSI BANTUAN
# ==============================

def extract_pose_landmarks(image_path):
    """Ambil dan normalisasi landmark pose dari gambar target."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        result = pose.process(image_rgb)
        if not result.pose_landmarks:
            return None
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark])
        return normalize_pose(landmarks)

def normalize_pose(landmarks):
    """Normalisasi pose agar tidak terpengaruh posisi kamera dan ukuran tubuh."""
    if landmarks is None:
        return None
    # Gunakan pinggul tengah sebagai pusat (mid-hip)
    left_hip = landmarks[23]
    right_hip = landmarks[24]
    center = (left_hip + right_hip) / 2
    landmarks -= center  # translasi ke pusat

    # Skala berdasarkan jarak antara bahu dan pinggul
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    shoulder_center = (left_shoulder + right_shoulder) / 2
    body_length = np.linalg.norm(shoulder_center - center)
    if body_length > 0:
        landmarks /= body_length
    return landmarks

def calculate_similarity(target, current):
    """Hitung skor kemiripan antar dua pose (0–100%)."""
    if target is None or current is None:
        return 0.0
    min_len = min(len(target), len(current))
    target, current = target[:min_len], current[:min_len]
    diff = np.linalg.norm(target - current)
    score = max(0, 100 - diff * SIMILARITY_SCALE)
    return round(score, 2)

# ==============================
# 🔹 SIAPKAN GAMBAR POSE TARGET
# ==============================
pose_files = sorted([os.path.join(POSE_FOLDER, f) for f in os.listdir(POSE_FOLDER)
                     if f.lower().endswith(('.jpg', '.png'))])

if len(pose_files) == 0:
    print("❌ Tidak ada gambar di folder poses/. Pastikan ada pose1.jpg ... pose10.jpg")
    exit()

random.shuffle(pose_files)
pose_files = pose_files[:NUM_POSES]

# ==============================
# 🔹 GAME LOOP
# ==============================
cap = cv2.VideoCapture(0)
pose_index = 0
pose_target_path = pose_files[pose_index]
pose_target = extract_pose_landmarks(pose_target_path)

if pose_target is None:
    print(f"⚠️ Gagal mendeteksi pose dari {pose_target_path}. Pastikan gambar berisi tubuh utuh.")
    exit()

score_display = 0
start_time = time.time()
total_score = []

print(f"▶️ Mulai game dengan {len(pose_files)} pose!")

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(image_rgb)

        if result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Tampilkan foto target di kiri atas
        target_img = cv2.imread(pose_target_path)
        if target_img is not None:
            target_img = cv2.resize(target_img, (200, 200))
            frame[10:210, 10:210] = target_img

        # Hitung skor setiap beberapa detik
        elapsed = time.time() - start_time
        if elapsed > THRESHOLD_SHOW and result.pose_landmarks:
            pose_player = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark])
            pose_player = normalize_pose(pose_player)
            score_display = calculate_similarity(pose_target, pose_player)
            total_score.append(score_display)

            # Pindah ke pose berikutnya
            pose_index += 1
            if pose_index >= len(pose_files):
                print("✅ Semua pose selesai!")
                print(f"🎯 Rata-rata kemiripan: {np.mean(total_score):.2f}%")
                break

            pose_target_path = pose_files[pose_index]
            pose_target = extract_pose_landmarks(pose_target_path)
            start_time = time.time()

        # Tampilkan info di layar
        cv2.putText(frame, f"Pose {pose_index+1}/{len(pose_files)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Kemiripan: {score_display:.2f}%", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Pose Imitation Game", frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
