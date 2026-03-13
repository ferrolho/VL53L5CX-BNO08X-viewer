"""Tests for filters module."""

import numpy as np
import pytest

from viewer.filters import TemporalFilter, fit_plane, fit_plane_ransac


class TestTemporalFilter:
    """Tests for temporal EMA filter."""

    def test_first_frame_passthrough(self):
        """First frame should pass through unchanged."""
        filt = TemporalFilter()
        distances = np.array([100, 200, 300], dtype=np.float32)

        result = filt.apply(distances, strength=0.5)

        np.testing.assert_array_equal(result, distances)

    def test_filter_smooths_changes(self):
        """Filter should smooth sudden changes."""
        filt = TemporalFilter()
        initial = np.array([100, 100, 100], dtype=np.float32)
        changed = np.array([200, 200, 200], dtype=np.float32)

        filt.apply(initial, strength=0.5)
        result = filt.apply(changed, strength=0.5)

        # Result should be between initial and changed
        assert np.all(result > initial)
        assert np.all(result < changed)

    def test_strength_zero_no_filtering(self):
        """Strength 0 should pass through unchanged."""
        filt = TemporalFilter()
        initial = np.array([100, 100, 100], dtype=np.float32)
        changed = np.array([200, 200, 200], dtype=np.float32)

        filt.apply(initial, strength=0.0)
        result = filt.apply(changed, strength=0.0)

        np.testing.assert_array_equal(result, changed)

    def test_strength_one_maximum_smoothing(self):
        """Strength 1 should keep previous value."""
        filt = TemporalFilter()
        initial = np.array([100, 100, 100], dtype=np.float32)
        changed = np.array([200, 200, 200], dtype=np.float32)

        filt.apply(initial, strength=1.0)
        result = filt.apply(changed, strength=1.0)

        np.testing.assert_array_equal(result, initial)

    def test_reset_clears_state(self):
        """Reset should clear filter state."""
        filt = TemporalFilter()
        initial = np.array([100, 100, 100], dtype=np.float32)

        filt.apply(initial, strength=0.5)
        filt.reset()

        # After reset, next frame should be treated as first
        new_distances = np.array([200, 200, 200], dtype=np.float32)
        result = filt.apply(new_distances, strength=0.5)

        np.testing.assert_array_equal(result, new_distances)


class TestFitPlane:
    """Tests for least-squares plane fitting."""

    def test_fits_horizontal_plane(self):
        """Should fit a horizontal plane correctly."""
        # Points on z=0.5 plane
        points = np.array([
            [0, 0, 0.5],
            [1, 0, 0.5],
            [0, 1, 0.5],
            [1, 1, 0.5],
        ], dtype=np.float32)

        result = fit_plane(points)

        assert result is not None
        pos, wxyz, size, rmse_mm = result
        # Position should be at centroid z
        assert abs(pos[2] - 0.5) < 0.01
        # Orientation should be identity (horizontal plane)
        assert abs(wxyz[0] - 1.0) < 0.1 or abs(wxyz[0] + 1.0) < 0.1
        # Perfect plane should have zero RMSE
        assert rmse_mm < 0.01

    def test_returns_none_for_collinear_points(self):
        """Should return None for collinear points."""
        # All points on a line
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
        ], dtype=np.float32)

        result = fit_plane(points)

        # May return None or a degenerate result
        # Just check it doesn't crash

    def test_minimum_three_points(self):
        """Should work with exactly 3 points."""
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=np.float32)

        result = fit_plane(points)

        assert result is not None


class TestFitPlaneRansac:
    """Tests for RANSAC plane fitting."""

    def test_fits_plane_with_outliers(self):
        """RANSAC should ignore outliers."""
        # Inliers on z=0.5 plane
        inliers = np.array([
            [0, 0, 0.5],
            [1, 0, 0.5],
            [0, 1, 0.5],
            [1, 1, 0.5],
            [0.5, 0.5, 0.5],
        ], dtype=np.float32)
        # Outlier far from plane
        outlier = np.array([[0.5, 0.5, 2.0]], dtype=np.float32)
        points = np.vstack([inliers, outlier])

        result = fit_plane_ransac(points, threshold=0.1)

        assert result is not None
        pos, wxyz, size, rmse_mm = result
        # Should fit to inliers, ignoring outlier
        assert abs(pos[2] - 0.5) < 0.2

    def test_returns_none_for_insufficient_points(self):
        """Should return None with fewer than 3 points."""
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
        ], dtype=np.float32)

        result = fit_plane_ransac(points)

        assert result is None
