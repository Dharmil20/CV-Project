
import numpy as np
from typing import Any, Tuple, Optional

class LaneDetector:
	"""
	Detects lane lines using sliding window and look-ahead search with polynomial fitting.
	"""
	def __init__(self, config: Any):
		"""
		Initialize LaneDetector.
		Args:
			config: Configuration object.
		"""
		self.config = config
		self.left_fit: Optional[np.ndarray] = None
		self.right_fit: Optional[np.ndarray] = None
		self.detected: bool = False

	def detect_lanes(self, binary_warped: np.ndarray, visualize: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, dict]:
		"""
		Main detection method. Chooses between sliding window and look-ahead search.
		Args:
			binary_warped: Warped binary image.
			visualize: Whether to return visualization image.
		Returns:
			left_fit, right_fit, out_img, lane_pixels (dict with leftx, lefty, rightx, righty)
		"""
		out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
		if self.detected and self.left_fit is not None and self.right_fit is not None and self.config.USE_LOOK_AHEAD:
			lane_pixels, out_img = self._look_ahead_search(binary_warped, out_img, visualize)
		else:
			lane_pixels, out_img = self._sliding_window_search(binary_warped, out_img, visualize)

		# Fit polynomials if pixels found
		if len(lane_pixels['leftx']) > 0 and len(lane_pixels['rightx']) > 0:
			self.left_fit = np.polyfit(lane_pixels['lefty'], lane_pixels['leftx'], self.config.POLY_ORDER)
			self.right_fit = np.polyfit(lane_pixels['righty'], lane_pixels['rightx'], self.config.POLY_ORDER)
			self.detected = True
		else:
			self.detected = False
		return self.left_fit, self.right_fit, out_img, lane_pixels

	def _sliding_window_search(self, binary_warped: np.ndarray, out_img: np.ndarray, visualize: bool) -> Tuple[dict, np.ndarray]:
		"""
		Sliding window search for lane pixels.
		Returns lane pixel coordinates and visualization image.
		"""
		histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
		midpoint = np.int32(histogram.shape[0] // 2)
		leftx_base = np.argmax(histogram[:midpoint])
		rightx_base = np.argmax(histogram[midpoint:]) + midpoint

		nwindows = self.config.NWINDOWS
		window_height = np.int32(binary_warped.shape[0] // nwindows)
		nonzero = binary_warped.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		leftx_current = leftx_base
		rightx_current = rightx_base
		margin = self.config.MARGIN
		minpix = self.config.MINPIX

		left_lane_inds = []
		right_lane_inds = []

		for window in range(nwindows):
			win_y_low = binary_warped.shape[0] - (window + 1) * window_height
			win_y_high = binary_warped.shape[0] - window * window_height
			win_xleft_low = leftx_current - margin
			win_xleft_high = leftx_current + margin
			win_xright_low = rightx_current - margin
			win_xright_high = rightx_current + margin

			good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
							  (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
			good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
							   (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

			left_lane_inds.append(good_left_inds)
			right_lane_inds.append(good_right_inds)

			if len(good_left_inds) > minpix:
				leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
			if len(good_right_inds) > minpix:
				rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

			if visualize:
				cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0,255,0), 2)
				cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0,255,0), 2)

		left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else np.array([])
		right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([])

		leftx = nonzerox[left_lane_inds] if left_lane_inds.size > 0 else np.array([])
		lefty = nonzeroy[left_lane_inds] if left_lane_inds.size > 0 else np.array([])
		rightx = nonzerox[right_lane_inds] if right_lane_inds.size > 0 else np.array([])
		righty = nonzeroy[right_lane_inds] if right_lane_inds.size > 0 else np.array([])

		if visualize:
			out_img[lefty, leftx] = [255, 0, 0]
			out_img[righty, rightx] = [0, 0, 255]

		lane_pixels = {'leftx': leftx, 'lefty': lefty, 'rightx': rightx, 'righty': righty}
		return lane_pixels, out_img

	def _look_ahead_search(self, binary_warped: np.ndarray, out_img: np.ndarray, visualize: bool) -> Tuple[dict, np.ndarray]:
		"""
		Search for lane pixels within a margin of the previous polynomial fit.
		"""
		nonzero = binary_warped.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		margin = self.config.SEARCH_MARGIN

		left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] - margin)) &
						  (nonzerox < (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] + margin)))
		right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2] - margin)) &
						   (nonzerox < (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2] + margin)))

		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds]
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds]

		if visualize:
			out_img[lefty, leftx] = [255, 0, 0]
			out_img[righty, rightx] = [0, 0, 255]

		lane_pixels = {'leftx': leftx, 'lefty': lefty, 'rightx': rightx, 'righty': righty}
		return lane_pixels, out_img

	def get_lane_points(self, y_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Generate x coordinates for given y values using fitted polynomials.
		Returns left_fitx, right_fitx.
		"""
		left_fitx = self.left_fit[0]*y_values**2 + self.left_fit[1]*y_values + self.left_fit[2] if self.left_fit is not None else np.zeros_like(y_values)
		right_fitx = self.right_fit[0]*y_values**2 + self.right_fit[1]*y_values + self.right_fit[2] if self.right_fit is not None else np.zeros_like(y_values)
		return left_fitx, right_fitx
