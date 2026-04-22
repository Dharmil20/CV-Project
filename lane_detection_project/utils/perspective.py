
import cv2
import numpy as np
from typing import Any

class PerspectiveTransformer:
	"""
	Handles perspective transform and its inverse for lane detection.
	"""
	def __init__(self, config: Any):
		"""
		Initialize PerspectiveTransformer and compute matrices.
		Args:
			config: Configuration object with SRC_POINTS and DST_POINTS.
		"""
		self.config = config
		self._compute_matrices()

	def _compute_matrices(self):
		src = np.float32(self.config.SRC_POINTS)
		dst = np.float32(self.config.DST_POINTS)
		self.M = cv2.getPerspectiveTransform(src, dst)
		self.Minv = cv2.getPerspectiveTransform(dst, src)

	def warp(self, image: np.ndarray) -> np.ndarray:
		"""
		Apply perspective transform to get bird's eye view.
		Args:
			image: Input image.
		Returns:
			Warped image.
		"""
		img_size = (image.shape[1], image.shape[0])
		return cv2.warpPerspective(image, self.M, img_size, flags=cv2.INTER_LINEAR)

	def unwarp(self, image: np.ndarray) -> np.ndarray:
		"""
		Apply inverse perspective transform to return to original view.
		Args:
			image: Warped image.
		Returns:
			Unwarped image.
		"""
		img_size = (image.shape[1], image.shape[0])
		return cv2.warpPerspective(image, self.Minv, img_size, flags=cv2.INTER_LINEAR)
