
import cv2
import os
import numpy as np

# --- Configuration ---
video_path = "C:/Users/HP/Desktop/3DReconstruction/Dataset/SET_2/vids/near_horiz_vid.mp4"
output_dir = "C:/Users/HP/Desktop/3DReconstruction/Dataset/SET_2/vids/frames"
os.makedirs(output_dir, exist_ok=True)

# --- Reference resolution and threshold ---
ref_width, ref_height = 1280, 720
base_threshold = 100.0  # At 720p

# --- Open video and extract FPS & resolution ---
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
N = max(1, round(fps * (5 / 30)))  # Spacing group size

# --- Compute blur threshold scaled to resolution ---
reference_pixels = ref_width * ref_height
current_pixels = width * height
scale = current_pixels / reference_pixels
adaptive_threshold = base_threshold * scale

print(f"Video FPS: {fps:.2f}, Resolution: {width}x{height}")
print(f"Using N = {N}, Adaptive blur threshold = {adaptive_threshold:.2f}")

# --- Helper: Laplacian sharpness score ---
def laplacian_variance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# --- Read and process in chunks of N frames ---
frame_group = []
frame_index = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_group.append((frame, frame_index))

    if len(frame_group) == N:
        # Pick the sharpest frame in this group
        scores = [laplacian_variance(f[0]) for f in frame_group]
        best_idx = int(np.argmax(scores))
        best_frame, best_frame_index = frame_group[best_idx]
        best_score = scores[best_idx]

        # Keep only if it's sharp enough
        if best_score >= adaptive_threshold :
            filename = f"{output_dir}/frame_{saved_count:04d}_idx{best_frame_index}_score{int(best_score)}.jpg"
            cv2.imwrite(filename, best_frame)
            saved_count += 1

        # Clear group for next chunk
        frame_group = []

    frame_index += 1

cap.release()
print(f"Saved {saved_count} sharp frames to {output_dir}")


