"""
Phase 1: Feature Extraction
YOLO26s-pose + ByteTrack -> 85 skeleton features per frame per tracked person
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
FEATURE_DIR = BASE_DIR / "v10_Features"
MAX_FRAMES = 30
MIN_FRAMES = 15

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

def cosine_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Cosine angle at point b between points a-b-c, returned in [0,1]."""
    v1 = a - b
    v2 = c - b
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
    cos_val = np.clip(dot / norm, -1.0, 1.0)
    return float(np.arccos(cos_val) / np.pi)


def extract_frame_features(
    kpts_xyn: np.ndarray,
    bbox_xyxy: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> tuple:
    """
    Extract per-frame features (no velocity yet).
    Returns (49-dim feature array, bbox_area).
    Layout: base(36) + spatial(5) + advanced(8) = 49
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

    base_coords = norm_kpts.flatten()  # 34 values
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
    # Trunk angle
    shoulder_mid = (norm_kpts[5] + norm_kpts[6]) / 2
    hip_mid = (norm_kpts[11] + norm_kpts[12]) / 2
    trunk_vec = shoulder_mid - hip_mid
    vertical = np.array([0.0, -1.0])
    dot_t = np.dot(trunk_vec, vertical)
    norm_t = np.linalg.norm(trunk_vec) * np.linalg.norm(vertical) + 1e-8
    trunk_angle = float(np.arccos(np.clip(dot_t / norm_t, -1.0, 1.0)) / np.pi)

    # Shoulder-hip ratio
    shoulder_dist = np.linalg.norm(norm_kpts[5] - norm_kpts[6])
    hip_dist = np.linalg.norm(norm_kpts[11] - norm_kpts[12])
    sh_ratio = shoulder_dist / (hip_dist + 1e-6)

    # Head orientation
    head_x = norm_kpts[0, 0] - np.mean([norm_kpts[1, 0], norm_kpts[2, 0]])
    head_y = norm_kpts[0, 1] - np.mean([norm_kpts[1, 1], norm_kpts[2, 1]])

    # Knee angles
    left_knee = cosine_angle(norm_kpts[11], norm_kpts[13], norm_kpts[15])
    right_knee = cosine_angle(norm_kpts[12], norm_kpts[14], norm_kpts[16])

    # Placeholders for velocity-dependent features
    wrist_sym = 0.0
    bbox_change = 0.0

    group_d = np.array([
        trunk_angle, sh_ratio, head_x, head_y,
        left_knee, right_knee, wrist_sym, bbox_change,
    ])

    frame_feats = np.concatenate([group_a, group_c, group_d])  # 49
    return frame_feats, bbox_area


def build_full_features(frame_list: list, bbox_areas: list) -> np.ndarray:
    """
    Assemble 85-dim vectors: base(36) + velocity(36) + spatial(5) + advanced(8).
    Computes velocity, wrist symmetry, and bbox area change.
    """
    n_frames = len(frame_list)
    frames = np.array(frame_list)  # (N, 49)

    base = frames[:, :36]
    spatial = frames[:, 36:41]
    advanced = frames[:, 41:49].copy()

    # GROUP B: VELOCITY
    velocity = np.zeros_like(base)
    velocity[1:] = base[1:] - base[:-1]

    # Fix wrist symmetry (advanced index 6)
    # Left wrist = kpt9 -> base indices 18,19; right wrist = kpt10 -> 20,21
    for t in range(n_frames):
        left_wrist_vel = np.linalg.norm(velocity[t, 18:20])
        right_wrist_vel = np.linalg.norm(velocity[t, 20:22])
        advanced[t, 6] = abs(left_wrist_vel - right_wrist_vel)

    # Fix bbox area change (advanced index 7)
    areas = np.array(bbox_areas)
    for t in range(1, n_frames):
        advanced[t, 7] = (areas[t] - areas[t - 1]) / (areas[t - 1] + 1e-6)

    # Assemble: base(36) + velocity(36) + spatial(5) + advanced(8) = 85
    full = np.concatenate([base, velocity, spatial, advanced], axis=1)
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

        # {track_id: [(features_49, bbox_area), ...]}
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
                kpts = keypoints.xyn[i].cpu().numpy()

                feats, area = extract_frame_features(kpts, bbox, frame_w, frame_h)

                if track_id not in track_data:
                    track_data[track_id] = []
                track_data[track_id].append((feats, area))

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

            full_features = build_full_features(frame_feats, bbox_areas)

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
print("FEATURE EXTRACTION SUMMARY")
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

print("\nDone.")
