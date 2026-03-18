"""
Phase 1b: Extended Feature Extraction for V11 Round 3
Extracts V10's 85 features PLUS new features for Round 3 experiments:
  - 17 keypoint confidence scores [Exp 3.1]
  -  2 soft cross-body distances   [Exp 3.2]
  - 36 acceleration features       [Exp 3.3]
  - 17 motion magnitude features   [Exp 3.4]
  -  1 hand velocity correlation   [Exp 3.5]
Total: 85 + 17 + 2 + 36 + 17 + 1 = 158 features per frame

Feature layout:
  [ 0: 36] Group A - base coords + elbow angles
  [36: 72] Group B - velocity
  [72: 77] Group C - spatial (wrist-hip, hand height, cross-body binary)
  [77: 85] Group D - advanced (trunk, shoulder-hip, head, knees, wrist sym, bbox)
  [85:102] NEW: keypoint confidence scores (17)
  [102:104] NEW: soft cross-body distances (2)
  [104:140] NEW: acceleration = 2nd-order velocity (36)
  [140:157] NEW: motion magnitude per keypoint (17)
  [157:158] NEW: hand velocity correlation (1)
"""

import os
import shutil
import numpy as np
import cv2
import yaml
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# -- Paths -------------------------------------------------------------------
BASE_DIR = Path(r"C:\Users\malho\Desktop\claudeagent\RETAILPROJECT")
CLASS_DIRS = {
    0: BASE_DIR / "data" / "Class_0_Normal",
    1: BASE_DIR / "data" / "Class_1_Shoplifting",
}
FEATURE_DIR = BASE_DIR / "v11_Features"
MAX_FRAMES = 30
MIN_FRAMES = 15
N_FEATURES_V11 = 158

# -- Clean output directory --------------------------------------------------
if FEATURE_DIR.exists():
    shutil.rmtree(FEATURE_DIR)
for label in [0, 1]:
    (FEATURE_DIR / str(label)).mkdir(parents=True, exist_ok=True)

# -- Write ByteTrack config --------------------------------------------------
tracker_cfg = {
    "tracker_type": "bytetrack",
    "track_high_thresh": 0.5,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.6,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "fuse_score": True,
    "gmc_method": "none",
}
tracker_path = BASE_DIR / "custom_tracker.yaml"
with open(tracker_path, "w") as f:
    yaml.dump(tracker_cfg, f)

# -- Load YOLO model ---------------------------------------------------------
model = YOLO("yolo26s-pose.pt")


# -- Helper functions ---------------------------------------------------------
def cosine_angle(a, b, c):
    v1 = a - b
    v2 = c - b
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
    cos_val = np.clip(dot / norm, -1.0, 1.0)
    return float(np.arccos(cos_val) / np.pi)


def extract_frame_features(kpts_xyn, bbox_xyxy, frame_w, frame_h, kpts_conf):
    """
    Extract per-frame features (no velocity/acceleration yet).
    Returns (49-dim V10 features, bbox_area, 17-dim conf, 2-dim soft cross-body).
    """
    kpts = kpts_xyn  # (17, 2) normalized to frame
    x1, y1, x2, y2 = bbox_xyxy
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    bbox_area = bbox_w * bbox_h

    # -- GROUP A: BASE (36 features) --
    norm_kpts = np.zeros((17, 2), dtype=np.float32)
    for i in range(17):
        norm_x = (kpts[i, 0] - (x1 / frame_w)) / (bbox_w / frame_w + 1e-8)
        norm_y = (kpts[i, 1] - (y1 / frame_h)) / (bbox_h / frame_h + 1e-8)
        norm_kpts[i] = [norm_x, norm_y]

    base_coords = norm_kpts.flatten()  # 34
    left_elbow = cosine_angle(norm_kpts[5], norm_kpts[7], norm_kpts[9])
    right_elbow = cosine_angle(norm_kpts[6], norm_kpts[8], norm_kpts[10])
    group_a = np.concatenate([base_coords, [left_elbow, right_elbow]])  # 36

    # -- GROUP C: SPATIAL (5 features) --
    left_wrist_hip = np.linalg.norm(norm_kpts[9] - norm_kpts[11])
    right_wrist_hip = np.linalg.norm(norm_kpts[10] - norm_kpts[12])
    hand_height = (
        np.mean([norm_kpts[9, 1], norm_kpts[10, 1]])
        - np.mean([norm_kpts[5, 1], norm_kpts[6, 1]])
    )
    left_cross = 1.0 if norm_kpts[9, 0] > norm_kpts[6, 0] else 0.0
    right_cross = 1.0 if norm_kpts[10, 0] < norm_kpts[5, 0] else 0.0
    group_c = np.array([left_wrist_hip, right_wrist_hip, hand_height, left_cross, right_cross])

    # -- GROUP D: ADVANCED (8 features) --
    shoulder_mid = (norm_kpts[5] + norm_kpts[6]) / 2
    hip_mid = (norm_kpts[11] + norm_kpts[12]) / 2
    trunk_vec = shoulder_mid - hip_mid
    vertical = np.array([0.0, -1.0])
    dot_t = np.dot(trunk_vec, vertical)
    norm_t = np.linalg.norm(trunk_vec) * np.linalg.norm(vertical) + 1e-8
    trunk_angle = float(np.arccos(np.clip(dot_t / norm_t, -1.0, 1.0)) / np.pi)

    shoulder_dist = np.linalg.norm(norm_kpts[5] - norm_kpts[6])
    hip_dist = np.linalg.norm(norm_kpts[11] - norm_kpts[12])
    sh_ratio = shoulder_dist / (hip_dist + 1e-6)

    head_x = norm_kpts[0, 0] - np.mean([norm_kpts[1, 0], norm_kpts[2, 0]])
    head_y = norm_kpts[0, 1] - np.mean([norm_kpts[1, 1], norm_kpts[2, 1]])

    left_knee = cosine_angle(norm_kpts[11], norm_kpts[13], norm_kpts[15])
    right_knee = cosine_angle(norm_kpts[12], norm_kpts[14], norm_kpts[16])

    wrist_sym = 0.0    # placeholder, computed later
    bbox_change = 0.0  # placeholder, computed later

    group_d = np.array([
        trunk_angle, sh_ratio, head_x, head_y,
        left_knee, right_knee, wrist_sym, bbox_change,
    ])

    # V10 features (49 per frame before velocity)
    frame_feats_v10 = np.concatenate([group_a, group_c, group_d])  # 49

    # -- NEW: Keypoint confidences (17) --
    conf = np.array(kpts_conf, dtype=np.float32)  # 17

    # -- NEW: Soft cross-body distances (2) --
    # (wrist_x - opposite_shoulder_x) / bbox_w -- continuous version
    bbox_w_norm = bbox_w / frame_w + 1e-8
    left_soft = (norm_kpts[9, 0] - norm_kpts[6, 0])   # left wrist vs right shoulder
    right_soft = (norm_kpts[5, 0] - norm_kpts[10, 0])  # right wrist vs left shoulder (negated so positive = crossing)
    soft_cross = np.array([left_soft, right_soft], dtype=np.float32)

    return frame_feats_v10, bbox_area, conf, soft_cross


def build_full_features(frame_list, bbox_areas, conf_list, soft_cross_list):
    """
    Assemble 158-dim feature vectors.
    V10 (85) + conf (17) + soft_cross (2) + accel (36) + magnitude (17) + hand_corr (1)
    """
    n_frames = len(frame_list)
    frames = np.array(frame_list)    # (N, 49)
    confs = np.array(conf_list)      # (N, 17)
    soft_crosses = np.array(soft_cross_list)  # (N, 2)

    base = frames[:, :36]
    spatial = frames[:, 36:41]
    advanced = frames[:, 41:49].copy()

    # -- GROUP B: VELOCITY (36) --
    velocity = np.zeros_like(base)
    velocity[1:] = base[1:] - base[:-1]

    # Fix wrist symmetry (advanced index 6)
    for t in range(n_frames):
        left_wrist_vel = np.linalg.norm(velocity[t, 18:20])
        right_wrist_vel = np.linalg.norm(velocity[t, 20:22])
        advanced[t, 6] = abs(left_wrist_vel - right_wrist_vel)

    # Fix bbox area change (advanced index 7)
    areas = np.array(bbox_areas)
    for t in range(1, n_frames):
        advanced[t, 7] = (areas[t] - areas[t - 1]) / (areas[t - 1] + 1e-6)

    # -- V10 features complete: base(36) + velocity(36) + spatial(5) + advanced(8) = 85 --
    v10_feats = np.concatenate([base, velocity, spatial, advanced], axis=1)  # (N, 85)

    # -- NEW: Acceleration (36) = 2nd-order velocity --
    acceleration = np.zeros_like(velocity)
    acceleration[1:] = velocity[1:] - velocity[:-1]

    # -- NEW: Motion magnitude per keypoint (17) --
    # sqrt(vel_x^2 + vel_y^2) for each of 17 keypoints
    magnitude = np.zeros((n_frames, 17), dtype=np.float32)
    for k in range(17):
        vx = velocity[:, 2 * k]
        vy = velocity[:, 2 * k + 1] if 2 * k + 1 < 34 else np.zeros(n_frames)
        magnitude[:, k] = np.sqrt(vx ** 2 + vy ** 2)

    # -- NEW: Hand velocity correlation (1) --
    # dot(left_hand_vel, right_hand_vel) / (norms + eps)
    hand_corr = np.zeros((n_frames, 1), dtype=np.float32)
    for t in range(n_frames):
        left_vel = velocity[t, 18:20]   # kpt9 (left wrist)
        right_vel = velocity[t, 20:22]  # kpt10 (right wrist)
        l_norm = np.linalg.norm(left_vel)
        r_norm = np.linalg.norm(right_vel)
        if l_norm > 1e-6 and r_norm > 1e-6:
            hand_corr[t, 0] = np.dot(left_vel, right_vel) / (l_norm * r_norm)

    # -- Assemble all 158 features --
    # [0:85]    V10 features
    # [85:102]  keypoint confidences (17)
    # [102:104] soft cross-body (2)
    # [104:140] acceleration (36)
    # [140:157] motion magnitude (17)
    # [157:158] hand velocity correlation (1)
    full = np.concatenate([
        v10_feats,       # 85
        confs,           # 17
        soft_crosses,    # 2
        acceleration,    # 36
        magnitude,       # 17
        hand_corr,       # 1
    ], axis=1)

    assert full.shape[1] == N_FEATURES_V11, f"Expected {N_FEATURES_V11}, got {full.shape[1]}"
    return full


# -- Main extraction loop ----------------------------------------------------
stats = {0: {"tracks": 0, "lengths": [], "zero_clips": []},
         1: {"tracks": 0, "lengths": [], "zero_clips": []}}

for label, class_dir in CLASS_DIRS.items():
    clips = sorted([f for f in class_dir.iterdir() if f.suffix == ".mp4"])
    class_name = "Normal" if label == 0 else "Shoplifting"
    print(f"\nProcessing {class_name}: {len(clips)} clips")

    for clip_path in tqdm(clips, desc=class_name):
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            print(f"  WARNING: Could not open {clip_path.name}")
            continue

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # {track_id: [(v10_feats_49, bbox_area, conf_17, soft_cross_2), ...]}
        track_data = {}

        frame_count = 0
        while frame_count < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            results = model.track(
                frame,
                persist=True,
                verbose=False,
                conf=0.3,
                tracker=str(tracker_path),
                classes=[0],
            )

            r = results[0]
            if r.boxes is None or r.boxes.id is None or r.keypoints is None:
                continue

            boxes = r.boxes
            keypoints = r.keypoints

            for i in range(len(boxes)):
                track_id = int(boxes.id[i].item())
                bbox = boxes.xyxy[i].cpu().numpy()
                kpts = keypoints.xyn[i].cpu().numpy()   # (17, 2)

                # Get confidence scores for each keypoint
                kpts_conf = keypoints.conf[i].cpu().numpy()  # (17,)

                feats, area, conf, soft_cross = extract_frame_features(
                    kpts, bbox, frame_w, frame_h, kpts_conf
                )

                if track_id not in track_data:
                    track_data[track_id] = []
                track_data[track_id].append((feats, area, conf, soft_cross))

        cap.release()

        # Process each track
        clip_name = clip_path.stem
        valid_tracks = 0
        for tid, data_list in track_data.items():
            if len(data_list) < MIN_FRAMES:
                continue

            data_list = data_list[:MAX_FRAMES]
            frame_feats = [d[0] for d in data_list]
            bbox_areas = [d[1] for d in data_list]
            conf_list = [d[2] for d in data_list]
            soft_cross_list = [d[3] for d in data_list]

            full_features = build_full_features(
                frame_feats, bbox_areas, conf_list, soft_cross_list
            )

            out_path = FEATURE_DIR / str(label) / f"{clip_name}_id{tid}.npy"
            np.save(out_path, full_features)
            valid_tracks += 1
            stats[label]["lengths"].append(len(data_list))

        stats[label]["tracks"] += valid_tracks
        if valid_tracks == 0:
            stats[label]["zero_clips"].append(clip_name)

        # Reset tracker between clips
        model.predictor = None

# -- Summary ------------------------------------------------------------------
print("\n" + "=" * 60)
print("V11 FEATURE EXTRACTION SUMMARY")
print("=" * 60)
for label in [0, 1]:
    name = "Normal" if label == 0 else "Shoplifting"
    s = stats[label]
    lengths = s["lengths"]
    print(f"\n{name} (class {label}):")
    print(f"  Total tracks saved: {s['tracks']}")
    if lengths:
        print(f"  Sequence lengths -- avg: {np.mean(lengths):.1f}, "
              f"min: {np.min(lengths)}, max: {np.max(lengths)}")
    print(f"  Clips with zero valid tracks ({len(s['zero_clips'])}): "
          f"{s['zero_clips'][:10]}{'...' if len(s['zero_clips']) > 10 else ''}")

for label in [0, 1]:
    saved = list((FEATURE_DIR / str(label)).glob("*.npy"))
    if saved:
        sample = np.load(saved[0])
        print(f"\nClass {label}: {len(saved)} .npy files, sample shape: {sample.shape}")

print(f"\nFeature layout ({N_FEATURES_V11} total):")
print("  [ 0: 85] V10 base+velocity+spatial+advanced")
print("  [85:102] Keypoint confidences (17)")
print("  [102:104] Soft cross-body distances (2)")
print("  [104:140] Acceleration (36)")
print("  [140:157] Motion magnitude per keypoint (17)")
print("  [157:158] Hand velocity correlation (1)")
print("\nDone.")
