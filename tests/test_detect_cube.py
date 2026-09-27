"""立方体贴图检测的反投影与去重测试。

不需要 YOLO —— 反投影是纯几何，可以独立验证。
这很重要：如果反投影错了，检测再多也全错，而且很难从结果上看出来。
"""

from __future__ import annotations

import numpy as np
import pytest

from pano_agent.geometry.projection import erp_to_cubemap
from pano_agent.geometry.sphere import great_circle_distance, sph_to_vec, vec_to_sph
from pano_agent.perception.detect_cube import (
    _FACE_ORIENTATION,
    CubeDetector,
    CubeDetectorConfig,
)
from pano_agent.perception.types import SphericalDetection


def make_detector(face_size: int = 512) -> CubeDetector:
    return CubeDetector(CubeDetectorConfig(face_size=face_size))


# ============================================================== 面的朝向


class TestFaceOrientation:
    def test_six_faces_defined(self):
        assert set(_FACE_ORIENTATION) == {"front", "back", "left", "right", "up", "down"}

    def test_axes_are_orthonormal(self):
        for name, (fwd, right, up) in _FACE_ORIENTATION.items():
            for v in (fwd, right, up):
                assert np.linalg.norm(v) == pytest.approx(1.0), f"{name} 轴非单位长度"
            assert np.dot(fwd, right) == pytest.approx(0.0), f"{name} forward⊥right"
            assert np.dot(fwd, up) == pytest.approx(0.0), f"{name} forward⊥up"
            assert np.dot(right, up) == pytest.approx(0.0), f"{name} right⊥up"

    def test_axes_match_geometry_module_convention(self):
        """立方体面的朝向必须与 geometry.erp_to_cubemap 一致。

        两边不一致的话，检测框会落在错误的球面位置 —— 而且因为
        "看起来还算合理" 而极难发现。
        """
        from pano_agent.geometry.projection import _CUBE_FACE_UV

        assert set(_FACE_ORIENTATION) == set(_CUBE_FACE_UV)

        # 用同一个方向在两个模块里分别定位，结果应当一致：
        # 对 front 面，其 forward 轴是 +Z，故 sph_to_vec(0, 0) 应等于它
        np.testing.assert_allclose(
            _FACE_ORIENTATION["front"][0], sph_to_vec(0.0, 0.0), atol=1e-12
        )
        # right 面 forward 是 +X → yaw=90°
        np.testing.assert_allclose(
            _FACE_ORIENTATION["right"][0], sph_to_vec(90.0, 0.0), atol=1e-12
        )
        # up 面 forward 是 +Y → pitch=90°
        np.testing.assert_allclose(
            _FACE_ORIENTATION["up"][0], sph_to_vec(0.0, 90.0), atol=1e-12
        )


# ============================================================== 反投影


class TestFaceBoxToSpherical:
    def test_face_center_maps_to_face_axis(self):
        """★ 面正中央的小框，其球面中心应等于该面的朝向轴。"""
        det = make_detector(512)
        expected = {
            "front": (0.0, 0.0),
            "right": (90.0, 0.0),
            "left": (-90.0, 0.0),
            "up": (0.0, 90.0),
            "down": (0.0, -90.0),
            "back": (180.0, 0.0),
        }

        for face, (exp_yaw, exp_pitch) in expected.items():
            center = 256.0
            half = 20.0
            out = det._face_box_to_spherical(
                face, center - half, center - half, center + half, center + half,
                512, 0.9, 0,
            )
            assert out is not None, f"{face} 反投影失败"

            d = great_circle_distance(out.yaw_deg, out.pitch_deg, exp_yaw, exp_pitch)
            assert d < 1.0, (
                f"{face} 面中心应映射到 ({exp_yaw}, {exp_pitch})，"
                f"实际 ({out.yaw_deg:.2f}, {out.pitch_deg:.2f})，差 {d:.2f}°"
            )

    def test_box_size_scales_with_extent(self):
        """框越大，反投影出的角尺寸越大。"""
        det = make_detector(512)
        small = det._face_box_to_spherical("front", 246, 246, 266, 266, 512, 0.9, 0)
        large = det._face_box_to_spherical("front", 156, 156, 356, 356, 512, 0.9, 0)

        assert small is not None and large is not None
        assert large.width_deg > small.width_deg
        assert large.height_deg > small.height_deg

    def test_quadrant_positions_are_correct(self):
        """面内四个象限的位置关系必须正确（左右上下不能颠倒）。"""
        det = make_detector(512)
        c, h = 256.0, 30.0

        center = det._face_box_to_spherical("front", c - h, c - h, c + h, c + h, 512, 0.9, 0)
        # front 面 forward=+Z，right=+X，up=+Y
        # 面内左上角（x<c, y<c）→ 应偏向 -X（yaw<0）与 +Y（pitch>0）
        left_up = det._face_box_to_spherical(
            "front", c - 80 - h, c - 80 - h, c - 80 + h, c - 80 + h, 512, 0.9, 0
        )
        right_down = det._face_box_to_spherical(
            "front", c + 80 - h, c + 80 - h, c + 80 + h, c + 80 + h, 512, 0.9, 0
        )

        assert center is not None and left_up is not None and right_down is not None
        assert left_up.yaw_deg < center.yaw_deg, "面内左侧应对应更小的 yaw"
        assert left_up.pitch_deg > center.pitch_deg, "面内上方应对应更大的 pitch"
        assert right_down.yaw_deg > center.yaw_deg
        assert right_down.pitch_deg < center.pitch_deg

    def test_cross_seam_face_center_uses_circular_mean(self):
        """★ back 面（yaw≈180）的中心不能因端点平均而落到 0°。

        这与 region.from_erp_pixels 踩过的是同一个坑：
        对接近 ±180° 的角取算术平均会得到 0（正好是反面）。
        """
        det = make_detector(512)
        c, h = 256.0, 60.0
        out = det._face_box_to_spherical("back", c - h, c - h, c + h, c + h, 512, 0.9, 0)

        assert out is not None
        assert abs(abs(out.yaw_deg) - 180.0) < 5.0, (
            f"back 面中心应在 ±180 附近，实际 {out.yaw_deg:.2f}°"
        )

    def test_unknown_face_returns_none(self):
        det = make_detector()
        assert det._face_box_to_spherical("nonexistent", 0, 0, 10, 10, 512, 0.9, 0) is None


class TestProjectionRoundTrip:
    """用 geometry 的立方体投影做闭环验证。"""

    def test_known_direction_lands_in_expected_face(self):
        """已知球面方向应出现在预期的立方体面上。

        做法：在 front 面取一个框，反投影到球面，再断言该方向确实
        朝向 +Z 方向 —— 若不成立说明两面约定不一致。
        """
        det = make_detector(512)
        out = det._face_box_to_spherical("front", 236, 236, 276, 276, 512, 0.9, 0)

        assert out is not None
        v = sph_to_vec(out.yaw_deg, out.pitch_deg)
        # front 面的 forward 是 +Z
        assert v[2] > 0.9, f"front 面检出的方向应主要指向 +Z，实际 {np.round(v, 3)}"

    def test_each_face_produces_distinct_region(self):
        """六个面的中心区域应互不重叠。"""
        det = make_detector(512)
        centers = []
        for face in _FACE_ORIENTATION:
            out = det._face_box_to_spherical(face, 236, 236, 276, 276, 512, 0.9, 0)
            assert out is not None
            centers.append(out.to_box())

        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                iou = centers[i].iou(centers[j], n_samples=3000)
                assert iou < 0.2, f"面 {i} 与面 {j} 的中心区域重叠过多 (IoU={iou:.3f})"


# ============================================================== 跨面去重


class TestCrossFaceMerge:
    def test_identical_detections_merge(self):
        """同一目标的重复检出应合并为一个。"""
        det = make_detector()
        dets = [
            SphericalDetection(30.0, 10.0, 20.0, 15.0, confidence=0.9, category="0"),
            SphericalDetection(30.2, 10.1, 19.8, 15.2, confidence=0.8, category="0"),
        ]
        merged = det._merge_across_faces(dets)
        assert len(merged) == 1
        assert merged[0].confidence == pytest.approx(0.9), "应保留置信度更高的那个"

    def test_distant_detections_kept_separate(self):
        det = make_detector()
        dets = [
            SphericalDetection(30.0, 10.0, 20.0, 15.0, confidence=0.9, category="0"),
            SphericalDetection(120.0, 10.0, 20.0, 15.0, confidence=0.9, category="0"),
        ]
        assert len(det._merge_across_faces(dets)) == 2

    def test_different_categories_not_merged(self):
        """即使位置重叠，类别不同也不应合并。"""
        det = make_detector()
        dets = [
            SphericalDetection(30.0, 10.0, 20.0, 15.0, confidence=0.9, category="0"),
            SphericalDetection(30.0, 10.0, 20.0, 15.0, confidence=0.9, category="2"),
        ]
        assert len(det._merge_across_faces(dets)) == 2

    def test_cross_seam_duplicates_merge(self):
        """★ 跨越 ERP 接缝的重复检出必须能合并。

        这是用**球面 IoU** 而非像素 IoU 的价值所在：
        同一目标在接缝两侧的像素框位于图像两端，像素 IoU 为 0，
        而球面 IoU 接近 1。
        """
        det = make_detector()
        dets = [
            SphericalDetection(179.5, 5.0, 12.0, 10.0, confidence=0.85, category="0"),
            SphericalDetection(-179.6, 5.1, 12.1, 10.1, confidence=0.80, category="0"),
        ]
        merged = det._merge_across_faces(dets)
        assert len(merged) == 1, f"跨接缝的重复检出应合并，实际 {len(merged)} 个"

    def test_empty_and_single_inputs(self):
        det = make_detector()
        assert det._merge_across_faces([]) == []
        one = [SphericalDetection(0.0, 0.0, 10.0, 10.0)]
        assert len(det._merge_across_faces(one)) == 1


class TestConfigValidation:
    def test_default_conf_is_coco_standard(self):
        """立方体域应当能用 COCO 的标准阈值 —— 这正是换投影的价值。"""
        cfg = CubeDetectorConfig()
        assert cfg.conf >= 0.1, (
            "若立方体域也需要把阈值降到 0.01 才能检出，说明投影没有解决问题"
        )

    def test_face_size_reasonable(self):
        cfg = CubeDetectorConfig()
        assert 256 <= cfg.face_size <= 2048
