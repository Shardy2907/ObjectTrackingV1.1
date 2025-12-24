import cv2
import pickle
import numpy as np
import os
import random

# -----------------------------
# --- Helper functions --------
# -----------------------------
def box_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2)/2, (y1 + y2)/2])

def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def extract_histogram(image, box):
    x1, y1, x2, y2 = [int(i) for i in box]
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        return np.zeros((512,), dtype=np.float32)
    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_patch], [0,1,2], None, [8,8,8], [0,180,0,256,0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def appearance_distance(hist1, hist2):
    return cv2.compareHist(hist1.astype('float32'), hist2.astype('float32'), cv2.HISTCMP_BHATTACHARYYA)

def draw_tracks(image, tracks):
    img = image.copy()
    for track in tracks:
        color = [random.randint(0,255) for _ in range(3)]
        for i in range(1, len(track["centers"])):
            p1 = tuple(track["centers"][i-1].astype(int))
            p2 = tuple(track["centers"][i].astype(int))
            cv2.line(img, p1, p2, color, 2)
    return img

# -----------------------------
# --- Appearance-based tracking
# -----------------------------
def track_by_appearance(detections, images, tau, alpha=0.5, max_distance=None):
    """
    max_distance: maximum allowed spatial distance (pixels) to match a detection
                  to an existing track. If None, no gating is applied.
    """
    tracks = {}
    finished_tracks = []
    track_id = 0

    for frame_id in range(len(detections)):
        current_detections = detections[frame_id]
        current_image = images[frame_id]
        centers = [box_center(b) for b in current_detections]
        hists = [extract_histogram(current_image, b) for b in current_detections]
        assigned_tracks = set()

        for center, hist in zip(centers, hists):
            best_score = float("inf")
            best_track = None
            best_spatial = float("inf")

            for tid, track in tracks.items():
                if tid in assigned_tracks:
                    continue

                dist_spatial = euclidean_distance(center, track["centers"][-1])

                # ---- Spatial gating (skip impossible matches) ----
                if max_distance is not None and dist_spatial > max_distance:
                    continue

                dist_appearance = appearance_distance(hist, track["histograms"][-1])
                score = alpha * dist_spatial + (1 - alpha) * dist_appearance

                if score < best_score:
                    best_score = score
                    best_track = tid
                    best_spatial = dist_spatial

            # Accept match only if within max_distance (or if max_distance is disabled)
            if best_track is not None and (max_distance is None or best_spatial <= max_distance):
                tracks[best_track]["centers"].append(center)
                tracks[best_track]["histograms"].append(hist)
                tracks[best_track]["missed"] = 0
                assigned_tracks.add(best_track)
            else:
                tracks[track_id] = {"centers": [center], "histograms": [hist], "missed": 0}
                track_id += 1

        # Update missed tracks
        to_delete = []
        for tid in list(tracks.keys()):
            if tid not in assigned_tracks:
                tracks[tid]["missed"] += 1
                if tracks[tid]["missed"] > tau:
                    finished_tracks.append(tracks[tid])
                    to_delete.append(tid)
        for tid in to_delete:
            del tracks[tid]

    finished_tracks.extend(tracks.values())
    return finished_tracks

# -----------------------------
# --- Main Script -------------
# -----------------------------
if __name__ == "__main__":

    # Paths (adjust if necessary)
    image_folder = "sequence"  # folder containing images
    pickle_file = "predictions.pickle"  # Faster R-CNN detections

    # Load images
    image_files = sorted([
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.endswith(".png") or f.endswith(".jpeg")
    ])
    images = [cv2.imread(f) for f in image_files]

    # Load detections
    with open(pickle_file, "rb") as f:
        detections = pickle.load(f)

    # ---- Choose a max association distance (pixels) ----
    # Try 50–120 depending on resolution and how fast people move.
    MAX_DIST = 80

    # Test appearance-based tracking for τ = 2, 5, 10
    taus = [2, 5, 10]
    for tau in taus:
        tracks_app = track_by_appearance(detections, images, tau, alpha=0.5, max_distance=MAX_DIST)
        print(f"Appearance-based tracking: τ={tau}, total tracks={len(tracks_app)}")
        img_tracks = draw_tracks(images[0], tracks_app)
        cv2.imwrite(f"appearance_tracks_tau{tau}.png", img_tracks)

    print("All appearance-based tracks saved for τ=2,5,10.")
