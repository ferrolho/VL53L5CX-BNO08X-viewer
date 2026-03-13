#!/usr/bin/env python3
"""VL53L5CX Point Cloud Viewer - Main application."""

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
import time

import numpy as np
import viser
from scipy.spatial.transform import Rotation

from . import config
from .filters import TemporalFilter, fit_plane, fit_plane_ransac
from .geometry import (
    CoordinateMethod,
    compute_zone_angles,
    correct_imu_to_tof_frame,
    distances_to_points,
    get_colors,
)
from .logging_config import setup_logging
from .scene import create_grid, create_scene_hierarchy, update_zone_rays
from .serial_reader import SerialReader

logger = logging.getLogger("vl53l5cx_viewer.main")


@dataclass
class MappingState:
    """State for mapping mode point accumulation (world coordinates)."""

    accumulated_points: list[np.ndarray] = field(default_factory=list)
    accumulated_colors: list[np.ndarray] = field(default_factory=list)
    _clear_requested: bool = False

    def request_clear(self):
        """Request a clear - will be processed in the main loop."""
        self._clear_requested = True

    def process_clear_if_requested(self) -> bool:
        """Process pending clear request. Returns True if cleared."""
        if self._clear_requested:
            self._clear_requested = False
            self.accumulated_points.clear()
            self.accumulated_colors.clear()
            return True
        return False

    def add(self, points: np.ndarray, colors: np.ndarray):
        self.accumulated_points.append(points)
        self.accumulated_colors.append(colors)

    def get_display_data(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.accumulated_points:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
        if len(self.accumulated_points) == 1:
            return self.accumulated_points[0], self.accumulated_colors[0]
        return np.vstack(self.accumulated_points), np.vstack(self.accumulated_colors)

    def total_points(self) -> int:
        return sum(len(p) for p in self.accumulated_points)

    def downsample(self, voxel_size: float, max_points: int):
        if not self.accumulated_points:
            return
        all_points = np.vstack(self.accumulated_points)
        all_colors = np.vstack(self.accumulated_colors)
        all_points, all_colors = voxel_downsample(all_points, all_colors, voxel_size)
        if len(all_points) > max_points:
            all_points = all_points[-max_points:]
            all_colors = all_colors[-max_points:]
        self.accumulated_points.clear()
        self.accumulated_points.append(all_points)
        self.accumulated_colors.clear()
        self.accumulated_colors.append(all_colors)


def voxel_downsample(
    points: np.ndarray, colors: np.ndarray, voxel_size: float
) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return points, colors
    voxel_indices = np.ascontiguousarray(np.floor(points / voxel_size).astype(np.int64))
    keys = voxel_indices.view(dtype=[("x", np.int64), ("y", np.int64), ("z", np.int64)]).ravel()
    _, unique_idx = np.unique(keys, return_index=True)
    return points[unique_idx], colors[unique_idx]


class VL53L5CXViewer:
    """Real-time point cloud viewer for VL53L5CX ToF sensor."""

    def __init__(self, port: str, baud: int = 115200):
        self.serial_reader = SerialReader(port, baud)
        self.zone_angles = compute_zone_angles()
        self.temporal_filter = TemporalFilter()

        # Compute board positions and offsets
        self.imu_board_center = (
            np.array(config.IMU_BOARD.world_position)
            - np.array(config.IMU_BOARD.sensor_offset)
        )
        self.tof_board_center = (
            np.array(config.TOF_BOARD.world_position)
            - np.array(config.TOF_BOARD.sensor_offset)
        )
        # Offset from IMU sensor to ToF sensor (for rotating ToF position when IMU active)
        self.imu_to_tof_offset = (
            np.array(config.TOF_BOARD.world_position)
            - np.array(config.IMU_BOARD.world_position)
        )

    def _setup_scene(self, server: viser.ViserServer):
        """Initialize the 3D scene."""
        server.scene.add_frame("/origin", axes_length=0.002, axes_radius=0.0001)
        create_grid(server)
        assets_dir = Path(__file__).parent.parent / "assets"
        self.scene = create_scene_hierarchy(server, assets_dir, self.zone_angles)

    def _setup_gui(self, server: viser.ViserServer, mapping_state: MappingState):
        """Initialize GUI controls."""
        with server.gui.add_folder("Sensor Info"):
            self.distance_text = server.gui.add_text("Status", initial_value="Waiting...")
            self.freq_text = server.gui.add_text("Frequency (Hz)", initial_value="0")
            self.imu_status_text = server.gui.add_text("IMU", initial_value="Not detected")

        with server.gui.add_folder("Settings"):
            self.point_size_slider = server.gui.add_slider(
                "Point Size", min=0.001, max=0.020, step=0.001, initial_value=0.005
            )
            self.show_rays_checkbox = server.gui.add_checkbox("Show Zone Rays", initial_value=True)
            self.clip_rays_checkbox = server.gui.add_checkbox(
                "Clip to Measurement", initial_value=False
            )

            @self.show_rays_checkbox.on_update
            def _on_show_rays_toggle(event: viser.GuiEvent) -> None:
                self.clip_rays_checkbox.disabled = not self.show_rays_checkbox.value

            @self.clip_rays_checkbox.on_update
            def _on_clip_rays_toggle(event: viser.GuiEvent) -> None:
                if not self.clip_rays_checkbox.value:
                    # Recreate full-length rays when clip mode is disabled
                    method = next(
                        m for m in CoordinateMethod if m.value == self.coord_method_dropdown.value
                    )
                    self.scene.zone_rays = update_zone_rays(
                        server, self.zone_angles, method, visible=self.show_rays_checkbox.value
                    )

            server.gui.add_markdown("---")
            self.coord_method_dropdown = server.gui.add_dropdown(
                "Coordinate Method",
                options=[m.value for m in CoordinateMethod],
                initial_value=CoordinateMethod.UNIFORM.value,
            )

            @self.coord_method_dropdown.on_update
            def _on_coord_method_change(event: viser.GuiEvent) -> None:
                method = next(
                    m for m in CoordinateMethod if m.value == self.coord_method_dropdown.value
                )
                # Update rays and replace stale handles
                self.scene.zone_rays = update_zone_rays(
                    server, self.zone_angles, method, visible=self.show_rays_checkbox.value
                )

            server.gui.add_markdown("---")
            self.imu_rotation_checkbox = server.gui.add_checkbox(
                "Apply IMU Rotation", initial_value=True
            )
            server.gui.add_markdown("---")
            self.filter_checkbox = server.gui.add_checkbox("Enable Filtering", initial_value=False)
            self.filter_strength_slider = server.gui.add_slider(
                "Filter Strength", min=0.0, max=1.0, step=0.05, initial_value=0.5, disabled=True
            )

            @self.filter_checkbox.on_update
            def _on_filter_toggle(event: viser.GuiEvent) -> None:
                self.filter_strength_slider.disabled = not self.filter_checkbox.value
                if not self.filter_checkbox.value:
                    self.temporal_filter.reset()

            server.gui.add_markdown("---")
            self.fit_plane_checkbox = server.gui.add_checkbox("Fit Plane", initial_value=False)
            self.plane_method_dropdown = server.gui.add_dropdown(
                "Method",
                options=["Least Squares", "RANSAC"],
                initial_value="Least Squares",
                disabled=True,
            )
            self.ransac_threshold_slider = server.gui.add_slider(
                "RANSAC Threshold (mm)", min=1, max=50, step=1, initial_value=10, visible=False
            )
            self.plane_error_text = server.gui.add_text(
                "Plane RMSE (mm)", initial_value="--"
            )

            @self.fit_plane_checkbox.on_update
            def _on_fit_plane_toggle(event: viser.GuiEvent) -> None:
                self.plane_method_dropdown.disabled = not self.fit_plane_checkbox.value
                self.ransac_threshold_slider.visible = (
                    self.fit_plane_checkbox.value
                    and self.plane_method_dropdown.value == "RANSAC"
                )
                if not self.fit_plane_checkbox.value:
                    self.plane_error_text.value = "--"

            @self.plane_method_dropdown.on_update
            def _on_plane_method_change(event: viser.GuiEvent) -> None:
                self.ransac_threshold_slider.visible = (
                    self.plane_method_dropdown.value == "RANSAC"
                )

        with server.gui.add_folder("Mapping"):
            self.mapping_checkbox = server.gui.add_checkbox("Mapping Mode", initial_value=False)
            self.voxel_size_slider = server.gui.add_slider(
                "Voxel Size (mm)", min=5, max=50, step=5, initial_value=10
            )
            self.max_points_slider = server.gui.add_slider(
                "Max Points (k)", min=10, max=500, step=10, initial_value=100
            )
            self.point_count_text = server.gui.add_text("Points", initial_value="0")
            clear_button = server.gui.add_button("Clear Map")

            @clear_button.on_click
            def _on_clear_click(event: viser.GuiEvent) -> None:
                mapping_state.request_clear()

            @self.mapping_checkbox.on_update
            def _on_mapping_toggle(event: viser.GuiEvent) -> None:
                if self.mapping_checkbox.value:
                    # Entering mapping mode: remove live points
                    server.scene.remove_by_name("/breadboard/tof/sensor/points")
                else:
                    # Exiting mapping mode: clear accumulated map
                    mapping_state.request_clear()

    def _update_scene_transforms(
        self, corrected_quat: np.ndarray, imu_connected: bool, apply_rotation: bool
    ) -> tuple[np.ndarray, Rotation] | None:
        """Update board frame transforms based on IMU orientation.

        Returns (tof_sensor_world_pos, tof_world_rot) if IMU active, else None.
        """
        imu_sensor_pos = np.array(config.IMU_BOARD.world_position)

        if apply_rotation and imu_connected:
            # Convert corrected quaternion to Rotation
            imu_rot = Rotation.from_quat(
                [corrected_quat[1], corrected_quat[2], corrected_quat[3], corrected_quat[0]]
            )

            # IMU board rotates around IMU sensor position
            self.scene.imu_board.wxyz = tuple(corrected_quat)
            # Board center position when rotated (sensor stays at world_position)
            imu_board_offset = -np.array(config.IMU_BOARD.sensor_offset)
            rotated_imu_board_offset = imu_rot.apply(imu_board_offset)
            self.scene.imu_board.position = tuple(imu_sensor_pos + rotated_imu_board_offset)

            # ToF sensor position: IMU sensor + rotated offset
            tof_sensor_pos = imu_sensor_pos + imu_rot.apply(self.imu_to_tof_offset)
            # ToF board rotates with IMU
            self.scene.tof_board.wxyz = tuple(corrected_quat)
            tof_board_offset = -np.array(config.TOF_BOARD.sensor_offset)
            rotated_tof_board_offset = imu_rot.apply(tof_board_offset)
            self.scene.tof_board.position = tuple(tof_sensor_pos + rotated_tof_board_offset)

            return tof_sensor_pos, imu_rot
        else:
            # Reset to configured positions
            self.scene.imu_board.wxyz = (1.0, 0.0, 0.0, 0.0)
            self.scene.imu_board.position = tuple(self.imu_board_center)
            self.scene.tof_board.wxyz = (1.0, 0.0, 0.0, 0.0)
            self.scene.tof_board.position = tuple(self.tof_board_center)
            return None

    def _process_frame(
        self, server: viser.ViserServer, mapping_state: MappingState, plane_handle
    ):
        """Process a single frame of sensor data."""
        distances, status, quaternion = self.serial_reader.get_data()

        if self.filter_checkbox.value:
            distances = self.temporal_filter.apply(distances, self.filter_strength_slider.value)

        imu_connected = self.serial_reader.imu_connected
        self.imu_status_text.value = "Connected" if imu_connected else "Not detected"

        corrected_quat = correct_imu_to_tof_frame(quaternion) if imu_connected else quaternion

        # Handle clear request atomically in main loop (must be outside sensor data check)
        if mapping_state.process_clear_if_requested():
            self.point_count_text.value = "0"
            server.scene.remove_by_name("/map/points")

        if np.any(distances > 0):
            # Get selected coordinate method
            coord_method = next(
                m for m in CoordinateMethod if m.value == self.coord_method_dropdown.value
            )

            # Points in sensor-local coordinates (z forward from sensor)
            points_local = distances_to_points(distances, self.zone_angles, coord_method)
            colors = get_colors(distances, status)
            valid_mask = (status == 5) & (distances >= config.MIN_RANGE_MM)

            # Update scene transforms and get world transform info
            transform_result = self._update_scene_transforms(
                corrected_quat, imu_connected, self.imu_rotation_checkbox.value
            )

            if np.any(valid_mask):
                valid_local = points_local[valid_mask].astype(np.float32)
                valid_colors = colors[valid_mask]

                # For mapping mode, we need points in world coordinates
                if self.mapping_checkbox.value:
                    # Transform local points to world
                    if transform_result is not None:
                        tof_sensor_pos, imu_rot = transform_result
                        # Apply sensor yaw, then IMU rotation, then translate
                        sensor_yaw = Rotation.from_euler(
                            "z", config.TOF_BOARD.sensor_yaw_deg, degrees=True
                        )
                        world_rot = imu_rot * sensor_yaw
                        valid_world = world_rot.apply(valid_local) + tof_sensor_pos
                    else:
                        # Just sensor yaw + offset
                        sensor_yaw = Rotation.from_euler(
                            "z", config.TOF_BOARD.sensor_yaw_deg, degrees=True
                        )
                        valid_world = (
                            sensor_yaw.apply(valid_local)
                            + np.array(config.TOF_BOARD.world_position)
                        )

                    mapping_state.add(valid_world, valid_colors)

                    if (
                        mapping_state.total_points() > config.DOWNSAMPLE_POINT_THRESHOLD
                        or len(mapping_state.accumulated_points)
                        > config.DOWNSAMPLE_BUFFER_THRESHOLD
                    ):
                        voxel_size_m = self.voxel_size_slider.value / 1000.0
                        max_pts = self.max_points_slider.value * 1000
                        mapping_state.downsample(voxel_size_m, max_pts)

                    display_points, display_colors = mapping_state.get_display_data()
                    self.point_count_text.value = f"{len(display_points):,}"

                    # Mapping points in world space
                    server.scene.add_point_cloud(
                        "/map/points",
                        points=display_points,
                        colors=display_colors,
                        point_size=self.point_size_slider.value,
                        point_shape="circle",
                    )
                else:
                    # Live points in sensor-local coordinates (frame handles transform)
                    server.scene.add_point_cloud(
                        "/breadboard/tof/sensor/points",
                        points=valid_local,
                        colors=valid_colors,
                        point_size=self.point_size_slider.value,
                        point_shape="circle",
                    )

                # Plane fitting (in sensor-local for consistency with live view)
                if self.fit_plane_checkbox.value and len(valid_local) >= 3:
                    if self.plane_method_dropdown.value == "RANSAC":
                        threshold_m = self.ransac_threshold_slider.value / 1000.0
                        plane_fit = fit_plane_ransac(valid_local, threshold=threshold_m)
                    else:
                        plane_fit = fit_plane(valid_local)

                    if plane_fit is not None:
                        pos, wxyz, size, rmse_mm = plane_fit
                        self.plane_error_text.value = f"{rmse_mm:.2f}"
                        plane_handle = server.scene.add_box(
                            "/breadboard/tof/sensor/plane",
                            dimensions=(size, size, 0.0001),
                            position=pos,
                            wxyz=wxyz,
                            color=(255, 255, 0),
                            opacity=0.5,
                        )

                valid_distances = distances[valid_mask]
                self.distance_text.value = (
                    f"Range: {valid_distances.min():.0f}-{valid_distances.max():.0f}mm"
                )
            else:
                self.distance_text.value = "No valid data"

        if plane_handle is not None:
            plane_handle.visible = self.fit_plane_checkbox.value

        self.freq_text.value = f"{self.serial_reader.data_fps:.1f}"

        # Update zone rays visibility and clipping
        if self.show_rays_checkbox.value and self.clip_rays_checkbox.value:
            # Recreate rays clipped to measured distances
            coord_method = next(
                m for m in CoordinateMethod if m.value == self.coord_method_dropdown.value
            )
            self.scene.zone_rays = update_zone_rays(
                server, self.zone_angles, coord_method,
                visible=True, distances=distances
            )
        else:
            for ray in self.scene.zone_rays:
                ray.visible = self.show_rays_checkbox.value

        return plane_handle

    def run(self, host: str = "0.0.0.0", port: int = 8080):
        """Start the viewer."""
        self.serial_reader.connect()
        self.serial_reader.start()

        server = viser.ViserServer(host=host, port=port)
        logger.info("Viser server started at http://localhost:%d", port)

        @server.on_client_connect
        def on_client_connect(client: viser.ClientHandle) -> None:
            client.camera.position = (0.0, -0.50, 0.50)
            client.camera.look_at = (0.0, 0.0, 0.0)
            client.camera.up = (0.0, 0.0, 1.0)
            client.camera.near = 0.001
            client.camera.fov = 0.35

        mapping_state = MappingState()
        self._setup_scene(server)
        self._setup_gui(server, mapping_state)

        plane_handle = None

        try:
            while True:
                frame_start = time.time()
                plane_handle = self._process_frame(server, mapping_state, plane_handle)

                elapsed = time.time() - frame_start
                if elapsed < config.FRAME_TIME:
                    time.sleep(config.FRAME_TIME - elapsed)

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.serial_reader.stop()


def main():
    parser = argparse.ArgumentParser(description="VL53L5CX Point Cloud Viewer")
    parser.add_argument(
        "--port",
        "-p",
        default="/dev/cu.usbserial-0001",
        help="Serial port (default: /dev/cu.usbserial-0001)",
    )
    parser.add_argument(
        "--baud", "-b", type=int, default=115200, help="Baud rate (default: 115200)"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Viser server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--viser-port", type=int, default=8080, help="Viser server port (default: 8080)"
    )
    parser.add_argument(
        "--debug", "-d", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)

    viewer = VL53L5CXViewer(port=args.port, baud=args.baud)
    viewer.run(host=args.host, port=args.viser_port)


if __name__ == "__main__":
    main()
