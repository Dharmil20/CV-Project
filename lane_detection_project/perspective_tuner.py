
import cv2
import numpy as np
from config import Config

def draw_points(img, points):
    pts = np.array(points, np.int32).reshape((-1, 1, 2))
    img_copy = img.copy()
    cv2.polylines(img_copy, [pts], isClosed=True, color=(0,255,255), thickness=2)
    for idx, pt in enumerate(points):
        cv2.circle(img_copy, tuple(pt), 8, (0,0,255), -1)
        cv2.putText(img_copy, str(idx), tuple(pt), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    return img_copy

def main():
    config = Config()
    video_path = 'test_videos/input.mp4'  # Change as needed
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame from {video_path}")
        return
    src_points = [list(pt) for pt in config.SRC_POINTS]
    selected_idx = 0

    print("Perspective Tuner Controls:")
    print("  Arrow keys: Move selected point")
    print("  Tab: Next point")
    print("  S: Print SRC_POINTS to console")
    print("  Q: Quit")

    while True:
        vis = draw_points(frame, src_points)
        cv2.imshow('Perspective Tuner', vis)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print("SRC_POINTS =", [tuple(pt) for pt in src_points])
        elif key == 9:  # Tab
            selected_idx = (selected_idx + 1) % len(src_points)
        elif key == 81:  # Left arrow
            src_points[selected_idx][0] -= 5
        elif key == 83:  # Right arrow
            src_points[selected_idx][0] += 5
        elif key == 82:  # Up arrow
            src_points[selected_idx][1] -= 5
        elif key == 84:  # Down arrow
            src_points[selected_idx][1] += 5

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
