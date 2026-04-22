
import cv2
import numpy as np
from typing import Any

class Thresholder:
	"""
	Applies color and gradient thresholding to input images for lane detection.
	"""
	def __init__(self, config: Any):
		"""
		Initialize Thresholder with configuration.
		Args:
			config: Configuration object with threshold parameters.
		"""
		self.config = config

	def apply_thresholds(self, image: np.ndarray) -> np.ndarray:
		"""
		Apply color and gradient thresholds to the input image.
		Args:
			image: Input BGR image (as read by cv2).
		Returns:
			Binary image (np.ndarray) with 0 or 1 values.
		"""
		# Convert to HLS color space
		hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
		l_channel = hls[:, :, 1]
		s_channel = hls[:, :, 2]

		# Color thresholding
		s_binary = np.zeros_like(s_channel)
		s_binary[(s_channel >= self.config.S_THRESH[0]) & (s_channel <= self.config.S_THRESH[1])] = 1

		l_binary = np.zeros_like(l_channel)
		l_binary[(l_channel >= self.config.L_THRESH[0]) & (l_channel <= self.config.L_THRESH[1])] = 1

		color_binary = np.zeros_like(s_channel)
		color_binary[(s_binary == 1) | (l_binary == 1)] = 1

		# Gradient thresholding (Sobel X)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.config.SOBEL_KERNEL)
		abs_sobelx = np.absolute(sobelx)
		scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx)) if np.max(abs_sobelx) > 0 else abs_sobelx
		grad_binary = np.zeros_like(scaled_sobel)
		grad_binary[(scaled_sobel >= self.config.SOBELX_THRESH[0]) & (scaled_sobel <= self.config.SOBELX_THRESH[1])] = 1

		# Combine color and gradient
		combined = np.zeros_like(grad_binary)
		combined[(color_binary == 1) | (grad_binary == 1)] = 1
		return combined.astype(np.uint8)
