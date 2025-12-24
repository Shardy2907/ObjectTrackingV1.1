import pickle
import numpy as np
import cv2
import random
import math

# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def box_center(box):
    """
    Compute center of a bounding box (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])


def euclidean_distance(p1, p2):
    """
    Euclidean distance between two points
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


# --------------------------------------------------
# Tracking by detection (with distance threshold)
# --------------------------------------------------

def track_by_detection(detections, tau, max_distance=None):
    """
    detections: list of detections per frame
    tau: maximum allowed missed frames
    max_distance: maximum allowed distance (in pixels) to associate a detection
                  to an existing track. If None, no threshold is used.
    """

    tracks = {}
    finished_tracks = []
    next_track_id = 0

    for frame_id in range(len(detections)):
        current_detections = detections[frame_id]
        centers = [box_center(b) for b in current_detections]

        assigned_tracks = set()

        # Assign detections to existing tracks
        for center in centers:
            min_dist = float("inf")
            best_track = None

            for tid, track in tracks.items():
                if tid in assigned_tracks:
                    continue

                last_center = track["centers"][-1]
                dist = euclidean_distance(center, last_center)

                if dist < min_dist:
                    min_dist = dist
                    best_track = tid

            # Apply gating (distance threshold)
            if best_track is not None and (max_distance is None or min_dist <= max_distance):
                tracks[best_track]["centers"].append(center)
                tracks[best_track]["missed"] = 0
                assigned_tracks.add(best_track)
            else:
                # Create new track
                tracks[next_track_id] = {
                    "centers": [center],
                    "missed": 0
                }
                next_track_id += 1

        # Update missed counters
        to_remove = []
        for tid in list(tracks.keys()):
            if tid not in assigned_tracks:
                tracks[tid]["missed"] += 1
                if tracks[tid]["missed"] > tau:
                    finished_tracks.append(tracks[tid])
                    to_remove.append(tid)

        for tid in to_remove:
            del tracks[tid]

    finished_tracks.extend(tracks.values())
    return finished_tracks


# --------------------------------------------------
# Visualisation
# --------------------------------------------------

def draw_tracks(image, tracks):
    """
    Draw colored track lines on the image
    """
    img = image.copy()

    for track in tracks:
        color = [random.randint(0, 255) for _ in range(3)]
        for i in range(1, len(track["centers"])):
            p1 = tuple(track["centers"][i - 1].astype(int))
            p2 = tuple(track["centers"][i].astype(int))
            cv2.line(img, p1, p2, color, 2)

    return img


# --------------------------------------------------
# Main execution
# --------------------------------------------------

if __name__ == "__main__":

    # Load detections from pickle file
    with open("predictions.pickle", "rb") as f:
        detections = pickle.load(f)

    # Load first frame (empty scene)
    first_frame = cv2.imread("sequence/S1-T1-C.00000.jpeg")

    if first_frame is None:
        raise FileNotFoundError("Could not load first frame. Check path: sequence/S1-T1-C.00000.jpeg")

    # Choose a max association distance (pixels)
    # Try values like 30, 50, 80, 120 depending on your resolution and walking speed.
    MAX_DIST = 80

    # Run tracking for different tau values
    tracks_tau2 = track_by_detection(detections, tau=2, max_distance=MAX_DIST)
    tracks_tau5 = track_by_detection(detections, tau=5, max_distance=MAX_DIST)
    tracks_tau10 = track_by_detection(detections, tau=10, max_distance=MAX_DIST)

    # Print total number of tracks
    print("Total tracks (τ = 2):", len(tracks_tau2))
    print("Total tracks (τ = 5):", len(tracks_tau5))
    print("Total tracks (τ = 10):", len(tracks_tau10))

    # Draw and save results
    img_tau2 = draw_tracks(first_frame, tracks_tau2)
    img_tau5 = draw_tracks(first_frame, tracks_tau5)
    img_tau10 = draw_tracks(first_frame, tracks_tau10)

    cv2.imwrite("tracks_tau2.png", img_tau2)
    cv2.imwrite("tracks_tau5.png", img_tau5)
    cv2.imwrite("tracks_tau10.png", img_tau10)

    print("Tracking visualisations saved successfully.")
