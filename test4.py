"""
pose_mimic_game.py
Prototype game: dua pemain meniru pose target.
Dependencies: mediapipe, opencv-python, numpy

Kontrol keyboard:
 - r : rekam pose target (akan minta nama)
 - l : list pose target yang terekam (ditampilkan di console)
 - d : hapus semua target (konfirmasi)
 - SPACE : mulai ronde (jika ada target; akan pilih target acak)
 - q or ESC : keluar
 - s : simpan targets ke file (targets.npy)
 - o : load targets dari file (targets.npy)
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import random
import json
import os

mp_pose = mp.solutions.pose

# pilih indeks landmark yang akan dipakai (pose)
LANDMARK_IDS = [
    0,   # nose
    11, 12,  # shoulders L R
    13, 14,  # elbows L R
    15, 16,  # wrists L R
    23, 24,  # hips L R
    25, 26,  # knees L R
    27, 28   # ankles L R
]

TARGETS_FILE = "pose_targets.npy"

def extract_pose_vector(landmarks, image_w, image_h):
    """
    Return a normalized vector representing pose.
    Steps:
      - Collect selected landmarks as (x,y)
      - Translate so pelvis center at origin
      - Scale by torso length (distance shoulder_center - hip_center)
      - Flatten to 1D vector
    If landmarks missing -> return None
    """
    if landmarks is None:
        return None
    pts = []
    for idx in LANDMARK_IDS:
        lm = landmarks[idx]
        # if visibility attribute present and low, allow but still include
        pts.append([lm.x * image_w, lm.y * image_h])
    pts = np.array(pts, dtype=np.float32)

    # compute centers
    left_sh = pts[LANDMARK_IDS.index(11)]
    right_sh = pts[LANDMARK_IDS.index(12)]
    sh_center = (left_sh + right_sh) / 2.0

    left_hip = pts[LANDMARK_IDS.index(23)]
    right_hip = pts[LANDMARK_IDS.index(24)]
    hip_center = (left_hip + right_hip) / 2.0

    torso_len = np.linalg.norm(sh_center - hip_center)
    if torso_len < 1e-3:
        return None

    norm_pts = (pts - hip_center) / torso_len
    return norm_pts.flatten()

def similarity_score(vec_target, vec_player):
    """
    Compute similarity between two pose vectors.
    We use cosine similarity on the flattened normalized vectors.
    Return score 0..100
    """
    if vec_target is None or vec_player is None:
        return 0.0
    # ensure same size
    if vec_target.shape != vec_player.shape:
        return 0.0
    # cosine similarity
    a = vec_target.astype(np.float32)
    b = vec_player.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-6:
        return 0.0
    cos = np.dot(a, b) / denom
    # map cos (-1..1) to 0..100
    score = (cos + 1) / 2 * 100
    return float(np.clip(score, 0.0, 100.0))

def draw_landmarks_on_frame(frame, results, offset_x=0):
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style()
        )

def choose_roi_by_persons(landmarks_list, image_w):
    """
    Determine which detected pose corresponds to left player and right player.
    Here, landmarks_list: list of pose landmarks objects for each detected person.
    But MediaPipe Pose gives only single person per frame. We'll instead
    use nose x coordinate to split into left/right by their x position.
    For single-person camera (most common), we attempt to detect two people by
    scanning for multiple poses is not available in MediaPipe Pose (single person).
    So strategy:
      - if only one person detected: decide left or right by x position of nose
      - we'll return two pose vectors: left_player_vec, right_player_vec (either may be None)
    """
    # With default MediaPipe Pose we only have one set of landmarks -> either left or right
    # We'll just rely on x position of nose to decide whether the person is left or right.
    return None  # handled differently in main loop

def save_targets(targets):
    np.save(TARGETS_FILE, np.array(targets, dtype=object))
    print(f"[Saved] {len(targets)} target(s) to {TARGETS_FILE}")

def load_targets():
    if os.path.exists(TARGETS_FILE):
        arr = np.load(TARGETS_FILE, allow_pickle=True)
        return list(arr.tolist())
    return []

def main():
    print("Starting Pose Mimic Game prototype...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera. Exiting.")
        return

    image_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    targets = load_targets()
    print(f"Loaded {len(targets)} targets.")

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    font = cv2.FONT_HERSHEY_SIMPLEX
    round_active = False

    round_duration = 4.0  # deteksi selama 4 detik
    countdown = 3  # sebelum mulai ronde
    last_round_start = None
    chosen_target = None
    scores = {"left": 0.0, "right": 0.0}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame_flipped = cv2.flip(frame, 1)  # mirror view
        rgb = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        # draw landmarks
        if results.pose_landmarks:
            draw_landmarks_on_frame(frame_flipped, results)

        # Get normalized vector for current pose (if present)
        vec = None
        if results.pose_landmarks:
            vec = extract_pose_vector(results.pose_landmarks.landmark, image_w, image_h)

        # Determine left/right player based on nose x coordinate (mirror frame)
        left_vec = None
        right_vec = None
        if results.pose_landmarks:
            nose = results.pose_landmarks.landmark[0]
            nose_x = nose.x * image_w
            # If nose on left half -> assign to left player
            if nose_x < image_w / 2:
                left_vec = vec
            else:
                right_vec = vec

        # Display instruction overlay
        cv2.putText(frame_flipped, "Press 'r' to RECORD target | SPACE to start round | 'l' list | 's' save | 'o' load", (10, 20), font, 0.5, (255,255,255), 1)

        # If recording mode triggered previously, we handle key below
        # If round active, manage countdown and scoring
        if round_active:
            now = time.time()
            elapsed = now - last_round_start
            # phase: countdown (negative elapsed), active (0..round_duration), ended (>round_duration)
            if elapsed < 0:
                # still countdown
                cv2.putText(frame_flipped, f"Get ready: {int(abs(elapsed))}", (10, 60), font, 1.0, (0,255,255), 2)
            elif elapsed <= round_duration:
                # active scoring window
                remaining = round_duration - elapsed
                cv2.putText(frame_flipped, f"Do the pose! Time left: {remaining:.1f}s", (10, 60), font, 1.0, (0,255,0), 2)
                # compute instantaneous score(s)
                if chosen_target is not None:
                    # compute score for left and right
                    left_score = similarity_score(chosen_target['vec'], left_vec) if left_vec is not None else 0.0
                    right_score = similarity_score(chosen_target['vec'], right_vec) if right_vec is not None else 0.0
                    # accumulate (simple average over frames) -> we just average by counting frames via timestamp
                    # We'll implement simple exponential moving average to smooth:
                    alpha = 0.2
                    scores['left'] = alpha * left_score + (1-alpha) * scores['left']
                    scores['right'] = alpha * right_score + (1-alpha) * scores['right']
                    cv2.putText(frame_flipped, f"Left score: {scores['left']:.1f}", (10, 100), font, 0.8, (200,200,255), 2)
                    cv2.putText(frame_flipped, f"Right score: {scores['right']:.1f}", (10, 130), font, 0.8, (200,200,255), 2)

                    # show chosen target name
                    cv2.putText(frame_flipped, f"Target: {chosen_target['name']}", (10, 160), font, 0.8, (255,255,0), 2)
            else:
                # round ended -> show winner
                round_active = False
                if scores['left'] > scores['right']:
                    result_text = f"Left wins ({scores['left']:.1f} vs {scores['right']:.1f})"
                elif scores['right'] > scores['left']:
                    result_text = f"Right wins ({scores['right']:.1f} vs {scores['left']:.1f})"
                else:
                    result_text = f"Tie ({scores['left']:.1f} vs {scores['right']:.1f})"
                cv2.putText(frame_flipped, "Round ended! " + result_text, (10, 200), font, 0.9, (0,255,255), 2)
                print("[Round finished]", result_text)

        # Show small preview of chosen target skeleton (if exists)
        if chosen_target is not None:
            # draw skeleton preview at top-right
            # reconstruct keypoint positions from normalized vector
            vec_t = chosen_target['vec']
            if vec_t is not None:
                pts = vec_t.reshape((-1,2))
                # map back to small box
                box_w, box_h = 200, 200
                base_x = image_w - box_w - 10
                base_y = 10
                # draw points
                for (x_norm, y_norm) in pts:
                    # undo normalization roughly: multiply by scale 50 and shift
                    x = int(base_x + (x_norm * 50) + box_w/2)
                    y = int(base_y + (y_norm * 50) + 30)
                    cv2.circle(frame_flipped, (x,y), 3, (0,255,255), -1)
                cv2.rectangle(frame_flipped, (base_x-2, base_y-2), (base_x+box_w+2, base_y+box_h+2), (50,50,50), 1)
                cv2.putText(frame_flipped, "Target preview", (base_x, base_y+box_h+18), font, 0.5, (255,255,255), 1)

        # Show instruction for recorded targets
        cv2.putText(frame_flipped, f"Recorded targets: {len(targets)}", (10, image_h - 10), font, 0.7, (180,180,180), 1)

        cv2.imshow('Pose Mimic Game (mirror)', frame_flipped)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            # Record current pose as new target
            if vec is None:
                print("No pose detected: pose not recorded.")
            else:
                name = input("Enter name for this target pose (single word): ").strip()
                if name == "":
                    name = f"pose_{len(targets)+1}"
                targets.append({'name': name, 'vec': vec})
                print(f"[Recorded] target '{name}' (total {len(targets)})")
        elif key == ord('l'):
            print("Targets:")
            for i,t in enumerate(targets):
                print(f"  {i+1}. {t['name']}")
        elif key == ord('d'):
            confirm = input("Delete all targets? Type 'yes' to confirm: ")
            if confirm.lower() == 'yes':
                targets = []
                print("All targets deleted.")
        elif key == ord('s'):
            save_targets(targets)
        elif key == ord('o'):
            targets = load_targets()
            print(f"Loaded {len(targets)} targets.")
        elif key == 32:  # SPACE -> start round
            if len(targets) == 0:
                print("No targets recorded. Press 'r' to record a target first.")
            else:
                # choose random target
                chosen_target = random.choice(targets)
                print(f"[Round start] Target: {chosen_target['name']}")
                # reset scores
                scores = {'left': 0.0, 'right': 0.0}
                # start with negative time for countdown
                last_round_start = time.time() - countdown
                round_active = True

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
