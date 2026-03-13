"""Geometry and coordinate transformation utilities for VL53L5CX."""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.spatial.transform import Rotation

from . import config


class CoordinateMethod(Enum):
    """Coordinate transformation method."""

    UNIFORM = "Uniform Grid"  # Assumes uniform angular spacing (ideal pinhole model)
    ST_LOOKUP = "ST Lookup Table"  # Uses ST-calibrated pitch/yaw tables


@dataclass
class ZoneAngles:
    """Pre-computed zone angle data for coordinate transforms.

    Contains precomputed values for both uniform grid and ST lookup table methods.
    """

    # Uniform grid method values
    tan_x: np.ndarray  # Tangent of X angles for each zone
    tan_y: np.ndarray  # Tangent of Y angles for each zone

    # ST lookup table method values (precomputed sin/cos for performance)
    st_sin_pitch: np.ndarray  # sin(pitch) for each zone
    st_cos_pitch: np.ndarray  # cos(pitch) for each zone
    st_sin_yaw: np.ndarray  # sin(yaw) for each zone
    st_cos_yaw: np.ndarray  # cos(yaw) for each zone

    # Ray directions for visualization - uniform method
    ray_dir_x: np.ndarray  # Normalized ray direction X component
    ray_dir_y: np.ndarray  # Normalized ray direction Y component
    ray_dir_z: np.ndarray  # Normalized ray direction Z component

    # Ray directions for visualization - ST lookup method
    st_ray_dir_x: np.ndarray
    st_ray_dir_y: np.ndarray
    st_ray_dir_z: np.ndarray


def compute_zone_angles() -> ZoneAngles:
    """Pre-compute the angle for each zone center.

    The sensor lens flips the image, so zone 0 corresponds to top-right.
    Computes values for both uniform grid and ST lookup table methods.
    """
    # === Uniform Grid Method ===
    # Convert diagonal FoV to per-axis FoV (assuming square sensor)
    # For a square, diagonal = side * sqrt(2), so side = diagonal / sqrt(2)
    fov_per_axis_deg = config.FOV_DIAGONAL_DEG / np.sqrt(2)
    fov_per_axis_rad = np.deg2rad(fov_per_axis_deg)

    # Angle step per zone
    angle_step = fov_per_axis_rad / config.RESOLUTION

    # Zone center offsets from optical axis
    # Zones are numbered row-major: 0-7 = row 0, 8-15 = row 1, etc.
    # Due to lens flip, we invert the mapping
    zone_angles_x = np.zeros(config.NUM_ZONES)
    zone_angles_y = np.zeros(config.NUM_ZONES)

    for i in range(config.NUM_ZONES):
        row = i // config.RESOLUTION
        col = i % config.RESOLUTION

        # Center of zone relative to center of grid (0-7 -> -3.5 to 3.5)
        # Flip due to lens inversion
        col_offset = (config.RESOLUTION - 1) / 2 - col  # Flip X
        row_offset = (config.RESOLUTION - 1) / 2 - row  # Flip Y

        zone_angles_x[i] = col_offset * angle_step
        zone_angles_y[i] = row_offset * angle_step

    # Precompute tan of zone angles for XY calculation
    # The sensor reports perpendicular (z-axis) distance, not radial
    tan_x = np.tan(zone_angles_x)
    tan_y = np.tan(zone_angles_y)

    # Also precompute normalized ray directions for visualization
    norm = np.sqrt(tan_x**2 + tan_y**2 + 1)
    ray_dir_x = tan_x / norm
    ray_dir_y = tan_y / norm
    ray_dir_z = 1.0 / norm

    # === ST Lookup Table Method ===
    # Precompute sin/cos for ST-calibrated pitch/yaw angles
    pitch_rad = np.deg2rad(config.ST_PITCH_ANGLES_DEG)
    yaw_rad = np.deg2rad(config.ST_YAW_ANGLES_DEG)

    st_sin_pitch = np.sin(pitch_rad)
    st_cos_pitch = np.cos(pitch_rad)
    st_sin_yaw = np.sin(yaw_rad)
    st_cos_yaw = np.cos(yaw_rad)

    # Compute ST ray directions (normalized)
    # Ray direction = (cos_yaw * cos_pitch, sin_yaw * cos_pitch, sin_pitch)
    # Negate X to match our lens-flip convention
    st_ray_dir_x = -st_cos_yaw * st_cos_pitch
    st_ray_dir_y = st_sin_yaw * st_cos_pitch
    st_ray_dir_z = st_sin_pitch
    # Normalize (should already be unit length, but ensure)
    st_norm = np.sqrt(st_ray_dir_x**2 + st_ray_dir_y**2 + st_ray_dir_z**2)
    st_ray_dir_x = st_ray_dir_x / st_norm
    st_ray_dir_y = st_ray_dir_y / st_norm
    st_ray_dir_z = st_ray_dir_z / st_norm

    return ZoneAngles(
        tan_x=tan_x,
        tan_y=tan_y,
        st_sin_pitch=st_sin_pitch,
        st_cos_pitch=st_cos_pitch,
        st_sin_yaw=st_sin_yaw,
        st_cos_yaw=st_cos_yaw,
        ray_dir_x=ray_dir_x,
        ray_dir_y=ray_dir_y,
        ray_dir_z=ray_dir_z,
        st_ray_dir_x=st_ray_dir_x,
        st_ray_dir_y=st_ray_dir_y,
        st_ray_dir_z=st_ray_dir_z,
    )


def distances_to_points(
    distances: np.ndarray,
    zone_angles: ZoneAngles,
    method: CoordinateMethod = CoordinateMethod.UNIFORM,
) -> np.ndarray:
    """Convert distance measurements to 3D point coordinates.

    The sensor is assumed to be pointing UP (+Z direction),
    lying flat on a horizontal surface. The VL53L5CX reports
    perpendicular (z-axis) distance, not radial distance - the
    chip performs this conversion internally.

    Args:
        distances: Array of 64 distance values in mm (perpendicular)
        zone_angles: Pre-computed zone angle data
        method: Coordinate transformation method to use

    Returns:
        Nx3 array of (x, y, z) coordinates in meters
    """
    # Convert to meters
    z_m = distances / 1000.0

    if method == CoordinateMethod.UNIFORM:
        # Uniform grid method: assumes uniform angular spacing
        # z IS the perpendicular distance, use tangent for lateral offset
        x = z_m * zone_angles.tan_x
        y = z_m * zone_angles.tan_y
        z = z_m
    else:
        # ST lookup table method: uses calibrated pitch/yaw angles
        # Hypotenuse = z_perpendicular / sin(pitch)
        # Then project using pitch/yaw to get XYZ
        hyp_m = z_m / zone_angles.st_sin_pitch

        # Negate X to match our lens-flip convention (ST tables use different X direction)
        x = -zone_angles.st_cos_yaw * zone_angles.st_cos_pitch * hyp_m
        y = zone_angles.st_sin_yaw * zone_angles.st_cos_pitch * hyp_m
        z = z_m  # Z is still the perpendicular distance

    return np.column_stack([x, y, z])


def get_colors(distances: np.ndarray, status: np.ndarray) -> np.ndarray:
    """Generate colors based on distance and validity.

    Valid points: Blue (close) to Red (far)
    Invalid points: Gray

    Args:
        distances: Array of distance values in mm
        status: Array of status values (5 = valid)

    Returns:
        Nx3 array of RGB colors (uint8)
    """
    # Normalize distances for color mapping
    d_norm = np.clip(
        (distances - config.MIN_RANGE_MM) / (config.MAX_RANGE_MM - config.MIN_RANGE_MM),
        0,
        1,
    )

    # Status 5 = valid measurement
    valid = (status == 5) & (distances >= config.MIN_RANGE_MM)

    # Vectorized color calculation for all points
    # Blue to Red gradient: R increases, G peaks in middle, B decreases
    r = (d_norm * 255).astype(np.uint8)
    g = ((1 - np.abs(2 * d_norm - 1)) * 200).astype(np.uint8)
    b = ((1 - d_norm) * 255).astype(np.uint8)

    # Stack into Nx3 array
    colors = np.column_stack([r, g, b])

    # Set invalid points to gray
    colors[~valid] = [128, 128, 128]

    return colors


def correct_imu_to_tof_frame(quaternion: np.ndarray) -> np.ndarray:
    """Apply frame correction for IMU-to-ToF sensor alignment.

    The BNO08X IMU is mounted 90° counterclockwise (around Z) relative to
    the VL53L5CX ToF sensor. This function applies a 90° clockwise correction.

    Args:
        quaternion: [w, x, y, z] quaternion from IMU (wxyz format)

    Returns:
        Corrected quaternion in wxyz format
    """
    # Convert IMU quaternion from wxyz to xyzw for scipy
    imu_xyzw = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    imu_rot = Rotation.from_quat(imu_xyzw)

    # 90° clockwise around Z = -90° around Z
    correction = Rotation.from_euler('z', -90, degrees=True)

    # Apply correction: corrected = imu * correction (correction in sensor's local frame)
    corrected_rot = imu_rot * correction

    # Convert back to wxyz format
    corrected_xyzw = corrected_rot.as_quat()
    return np.array([corrected_xyzw[3], corrected_xyzw[0], corrected_xyzw[1], corrected_xyzw[2]])


def rotate_points_by_quaternion(points: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """Rotate points using a quaternion.

    Args:
        points: Nx3 array of 3D points
        quaternion: [w, x, y, z] quaternion (wxyz format from BNO08X)

    Returns:
        Rotated Nx3 array of 3D points
    """
    # scipy uses xyzw format, convert from wxyz
    quat_xyzw = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    rotation = Rotation.from_quat(quat_xyzw)

    return rotation.apply(points)


def rotation_matrix_from_vectors(vec_from: np.ndarray, vec_to: np.ndarray) -> np.ndarray:
    """Compute rotation matrix that rotates vec_from to vec_to.

    Uses Rodrigues' rotation formula.

    Args:
        vec_from: Source vector (will be normalized)
        vec_to: Target vector (will be normalized)

    Returns:
        3x3 rotation matrix
    """
    # Normalize inputs
    a = vec_from / np.linalg.norm(vec_from)
    b = vec_to / np.linalg.norm(vec_to)

    # Handle parallel vectors
    dot = np.dot(a, b)
    if dot > 0.9999:
        return np.eye(3)
    if dot < -0.9999:
        # 180 degree rotation - find perpendicular axis
        perp = np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])
        axis = np.cross(a, perp)
        axis = axis / np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)

    # Rodrigues' formula
    v = np.cross(a, b)
    s = np.linalg.norm(v)  # sin(angle)
    c = dot  # cos(angle)

    # Skew-symmetric cross-product matrix
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])

    # Rotation matrix: R = I + vx + vx^2 * (1-c)/s^2
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
