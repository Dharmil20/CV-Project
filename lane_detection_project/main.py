
import cv2
import numpy as np
from collections import deque
from config import Config
from utils.thresholding import Thresholder
from utils.perspective import PerspectiveTransformer
from utils.lane_detector import LaneDetector
from utils.metrics import MetricsCalculator
from utils.visualizer import Visualizer

class LaneDetectionPipeline:
    """
    Main pipeline for lane detection.
    """
    def __init__(self, config: Config):
        self.config = config
        self.thresholder = Thresholder(config)
        self.perspective = PerspectiveTransformer(config)
        self.lane_detector = LaneDetector(config)
        self.metrics = MetricsCalculator(config)
        self.visualizer = Visualizer(config)
        self.left_fit_buffer = deque(maxlen=config.SMOOTHING_FRAMES)
        self.right_fit_buffer = deque(maxlen=config.SMOOTHING_FRAMES)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        binary = self.thresholder.apply_thresholds(frame)
        binary_warped = self.perspective.warp(binary)
        left_fit, right_fit, out_img, lane_pixels = self.lane_detector.detect_lanes(binary_warped, visualize=self.config.SHOW_WINDOWS)

        if left_fit is not None and right_fit is not None:
            self.left_fit_buffer.append(left_fit)
            self.right_fit_buffer.append(right_fit)

        if len(self.left_fit_buffer) > 0 and len(self.right_fit_buffer) > 0:
            left_fit_avg = np.mean(self.left_fit_buffer, axis=0)
            right_fit_avg = np.mean(self.right_fit_buffer, axis=0)
        else:
            left_fit_avg = left_fit
            right_fit_avg = right_fit

        # Draw lane overlay
        result = frame.copy()
        if left_fit_avg is not None and right_fit_avg is not None:
            result = self.visualizer.draw_lane_overlay(result, binary_warped, left_fit_avg, right_fit_avg, self.perspective.Minv, alpha=self.config.LANE_ALPHA)

            # Calculate metrics
            y_eval = frame.shape[0] - 1
            curvature_left, curvature_right = self.metrics.calculate_curvature(left_fit_avg, right_fit_avg, y_eval)
            offset = self.metrics.calculate_offset(left_fit_avg, right_fit_avg, y_eval, frame.shape[1])
            if self.config.SHOW_METRICS:
                result = self.visualizer.draw_metrics(result, curvature_left, curvature_right, offset)

        # Debug visualizations
        if self.config.SHOW_BINARY:
            cv2.imshow('Binary Threshold', binary * 255)
        if self.config.SHOW_WARPED:
            cv2.imshow('Bird Eye View', binary_warped * 255)
        if self.config.SHOW_WINDOWS:
            cv2.imshow('Sliding Windows', out_img)

        return result

def main():
    config = Config()
    pipeline = LaneDetectionPipeline(config)
    video_path = 'test_videos/input.mp4'  # Change as needed
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter('output_videos/result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    paused = False
    frame_idx = 0
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            processed = pipeline.process_frame(frame)
            cv2.imshow('Lane Detection', processed)
            out.write(processed)
            frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('b'):
            config.SHOW_BINARY = not config.SHOW_BINARY
        elif key == ord('w'):
            config.SHOW_WARPED = not config.SHOW_WARPED
        elif key == ord('s'):
            config.SHOW_WINDOWS = not config.SHOW_WINDOWS
        elif key == ord('m'):
            config.SHOW_METRICS = not config.SHOW_METRICS
        elif key == ord('f') and not paused:
            cv2.imwrite(f'debug_frames/frame_{frame_idx}.png', processed)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print('Processing complete. Output saved to output_videos/result.mp4')

if __name__ == "__main__":
    main()
