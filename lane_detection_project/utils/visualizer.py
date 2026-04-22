
import cv2
import numpy as np
from typing import Any

class Visualizer:
	"""
	Handles drawing lane overlays and metrics on images.
	"""
	def __init__(self, config: Any):
		"""
		Initialize Visualizer with configuration.
		Args:
			config: Configuration object.
		"""
		self.config = config

	def draw_lane_overlay(self, original_img: np.ndarray, binary_warped: np.ndarray, left_fit: np.ndarray, right_fit: np.ndarray, Minv: np.ndarray, alpha: float = 0.3) -> np.ndarray:
		"""
		Draw lane overlay and boundaries on the original image.
		Args:
			original_img: Original BGR image.
			binary_warped: Warped binary image.
			left_fit: Polynomial coefficients for left lane.
			right_fit: Polynomial coefficients for right lane.
			Minv: Inverse perspective matrix.
			alpha: Overlay transparency.
		Returns:
			Image with lane overlay.
		"""
		# Create blank image for overlay
		warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
		color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
		h, w = binary_warped.shape
		ploty = np.linspace(0, h-1, h)
		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

		# Create points for polygon
		pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
		pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
		pts = np.hstack((pts_left, pts_right)).astype(np.int32)

		# Draw lane area
		cv2.fillPoly(color_warp, [pts], (0, 255, 0))
		# Draw lane boundaries
		cv2.polylines(color_warp, [pts_left.astype(np.int32)], isClosed=False, color=(0,255,255), thickness=20)
		cv2.polylines(color_warp, [pts_right.astype(np.int32)], isClosed=False, color=(0,255,255), thickness=20)

		# Warp overlay back to original perspective
		newwarp = cv2.warpPerspective(color_warp, Minv, (w, h))
		# Blend with original image
		result = cv2.addWeighted(original_img, 1, newwarp, alpha, 0)
		return result

	def draw_metrics(self, image: np.ndarray, curvature_left: float, curvature_right: float, offset: float) -> np.ndarray:
		"""
		Draw curvature and vehicle offset metrics on the image.
		Args:
			image: Input image.
			curvature_left: Left lane curvature radius (meters).
			curvature_right: Right lane curvature radius (meters).
			offset: Vehicle offset from lane center (meters).
		Returns:
			Annotated image.
		"""
		avg_curvature = (curvature_left + curvature_right) / 2.0
		direction = "Right" if offset > 0 else "Left"
		curvature_text = f"Radius of Curvature: {avg_curvature:.1f} m"
		offset_text = f"Vehicle Position: {abs(offset):.2f} m {direction} of center"
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(image, curvature_text, (50, 50), font, 1, (255,255,255), 2, cv2.LINE_AA)
		cv2.putText(image, offset_text, (50, 100), font, 1, (255,255,255), 2, cv2.LINE_AA)
		return image
