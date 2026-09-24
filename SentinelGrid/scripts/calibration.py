"""Create an image-to-ground homography from a calibrated pinhole camera.

Use this with K (intrinsics), R and t (world-to-camera extrinsics) from a
dataset. The world ground plane is Z=0 and units are the dataset's world units.
"""
import numpy as np


def image_to_ground(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    world_ground_to_image = K @ np.column_stack((R[:, 0], R[:, 1], t.reshape(3)))
    return np.linalg.inv(world_ground_to_image)
