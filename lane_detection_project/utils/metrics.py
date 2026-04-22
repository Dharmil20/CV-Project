
import numpy as np
from typing import Any, Tuple

class MetricsCalculator:
	"""
	Calculates lane curvature and vehicle offset metrics.
	"""
	def __init__(self, config: Any):
		"""
		Initialize MetricsCalculator with configuration.
		Args:
			config: Configuration object with metric parameters.
		"""
		self.config = config

	def calculate_curvature(self, left_fit: np.ndarray, right_fit: np.ndarray, y_eval: float) -> Tuple[float, float]:
		"""
		Calculate the radius of curvature for left and right lanes in meters.
		Args:
			left_fit: Polynomial coefficients for left lane (pixel space).
			right_fit: Polynomial coefficients for right lane (pixel space).
			y_eval: y-value at which to evaluate curvature (usually bottom of image).
		Returns:
			(left_curverad, right_curverad): Tuple of curvature radii in meters.
		"""
		ym_per_pix = self.config.YM_PER_PIX
		xm_per_pix = self.config.XM_PER_PIX

		# Generate y values
		ploty = np.linspace(0, y_eval, num=int(y_eval+1))

		# Calculate x values for each lane line
		leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

		# Fit new polynomials in world space
		left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
		right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

		# Calculate curvature radius
		left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.abs(2*left_fit_cr[0])
		right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.abs(2*right_fit_cr[0])
		return left_curverad, right_curverad

	def calculate_offset(self, left_fit: np.ndarray, right_fit: np.ndarray, y_eval: float, image_width: int) -> float:
		"""
		Calculate the vehicle's offset from the lane center in meters.
		Args:
			left_fit: Polynomial coefficients for left lane (pixel space).
			right_fit: Polynomial coefficients for right lane (pixel space).
			y_eval: y-value at which to evaluate offset (usually bottom of image).
			image_width: Width of the image in pixels.
		Returns:
			offset_meters: Offset from lane center (positive = right of center).
		"""
		xm_per_pix = self.config.XM_PER_PIX
		# Lane line positions at the bottom of the image
		left_x = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
		right_x = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
		lane_center = (left_x + right_x) / 2.0
		vehicle_center = image_width / 2.0
		offset_pixels = vehicle_center - lane_center
		offset_meters = offset_pixels * xm_per_pix
		return offset_meters
