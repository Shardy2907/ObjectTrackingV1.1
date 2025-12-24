import cv2
import pickle
import numpy as np
import os
import random

def box_center(box):
    """Compute center point of a bounding box"""
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2)/2, (y1 + y2)/2])

def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def extract_histogram(image, box):
    """Extract HSV color histogram for a bounding box"""
    x1, y1, x2, y2 = [int(i) for i in box]
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        return np.zeros((512,), dtype=np.float32)  # handle empty boxes
    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_patch], [0,1,2], None, [8,8,8], [0,180,0,256,0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def appearance_distance(hist1, hist2):
    """Bhattacharyya distance between histograms"""
    return cv2.compareHist(hist1.astype('float32'), hist2.astype('float32'), cv2.HISTCMP_BHATTACHARYYA)

def draw_tracks(image, tracks):
    """Draw tracks as colored lines connecting centers"""
    img = image.copy()
    for track in tracks:
        color = [random.randint(0,255) for _ in range(3)]
        for i in range(1, len(track["centers"])):
            p1 = tuple(track["centers"][i-1].astype(int))
            p2 = tuple(track["centers"][i].astype(int))
            cv2.line(img, p1, p2, color, 2)
    return img

# -----------------------------
# --- Tracking functions -------
# -----------------------------

def track_by_distance(detections, tau, max_distance=None):
    """Task 2: Simple distance-based tracking with optional distance gating"""
    tracks = {}
    finished_tracks = []
    track_id = 0

    for frame_id in range(len(detections)):
        current_detections = detections[frame_id]
        centers = [box_center(b) for b in current_detections]
        assigned_tracks = set()

        for center in centers:
            min_dist = float("inf")
            best_track = None

            for tid, track in tracks.items():
                if tid in assigned_tracks:
                    continue
                dist = euclidean_distance(center, track["centers"][-1])
                if dist < min_dist:
                    min_dist = dist
                    best_track = tid

            # Apply gating: only match if close enough
            if best_track is not None and (max_distance is None or min_dist <= max_distance):
                tracks[best_track]["centers"].append(center)
                tracks[best_track]["missed"] = 0
                assigned_tracks.add(best_track)
            else:
                tracks[track_id] = {"centers": [center], "missed": 0}
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

def track_by_appearance(detections, images, tau, alpha=0.5, max_distance=None):
    """Task 3: Appearance-based tracking with optional spatial gating"""
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
            best_spatial = float("inf")  # track spatial distance of the best match

            for tid, track in tracks.items():
                if tid in assigned_tracks:
                    continue

                dist_spatial = euclidean_distance(center, track["centers"][-1])

                # Apply spatial gating early (skip impossible matches)
                if max_distance is not None and dist_spatial > max_distance:
                    continue

                dist_appearance = appearance_distance(hist, track["histograms"][-1])
                score = alpha * dist_spatial + (1 - alpha) * dist_appearance

                if score < best_score:
                    best_score = score
                    best_track = tid
                    best_spatial = dist_spatial

            # If no eligible track (or none within max_distance), start a new one
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

if __name__ == "__main__":

    # Paths (change if needed)
    image_folder = "sequence"  # folder containing image sequence
    pickle_file = "predictions.pickle"  # detections

    # Load image sequence
    image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder)
                          if f.endswith(".png") or f.endswith(".jpeg")])
    images = [cv2.imread(f) for f in image_files]

    # Load detections
    with open(pickle_file, "rb") as f:
        detections = pickle.load(f)

    # Choose a max association distance (pixels)
    MAX_DIST = 80  # try 50-120 depending on resolution and motion

    # ------------------------
    # Task 2: Distance-based tracking
    # ------------------------
    taus = [2, 5, 10]
    for tau in taus:
        tracks = track_by_distance(detections, tau, max_distance=MAX_DIST)
        print(f"Distance-based tracking: τ={tau}, total tracks={len(tracks)}")
        img_tracks = draw_tracks(images[0], tracks)
        cv2.imwrite(f"tracks_distance_tau{tau}.png", img_tracks)

    # ------------------------
    # Task 3: Appearance-based tracking
    # ------------------------
    tau = 5  # you can test other τ values
    tracks_app = track_by_appearance(detections, images, tau, alpha=0.5, max_distance=MAX_DIST)
    print(f"Appearance-based tracking: τ={tau}, total tracks={len(tracks_app)}")
    img_app = draw_tracks(images[0], tracks_app)
    cv2.imwrite("tracks_appearance.png", img_app)

    print("Tracking finished. Output images saved.")
