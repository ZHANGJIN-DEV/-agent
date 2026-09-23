"""ERP 检测的接缝碎片合并测试。

这部分是**纯几何逻辑**，不依赖 torch / ultralytics，因此可以独立验证。
它解决的是：一个位于 yaw≈±180° 的目标在 ERP 上被切成左右两半，
检测器会输出两个残缺的框，下游跟踪器会误认为出现了两个目标。
"""

from __future__ import annotations

import numpy as np
import pytest

from pano_agent.perception.detect import _merge_seam_fragments

W, H = 2048, 1024


def box(x1: float, y1: float, x2: float, y2: float, conf: float = 0.9, cls: float = 0.0):
    return np.array([x1, y1, x2, y2, conf, cls], dtype=np.float64)


class TestNoMergeCases:
    def test_ordinary_boxes_unchanged(self):
        boxes = np.array([box(100, 100, 200, 200), box(500, 300, 700, 500)])
        out = _merge_seam_fragments(boxes, W, H)
        assert len(out) == 2
        np.testing.assert_allclose(np.sort(out, axis=0), np.sort(boxes, axis=0))

    def test_empty_input(self):
        out = _merge_seam_fragments(np.empty((0, 6)), W, H)
        assert out.size == 0

    def test_vertically_disjoint_fragments_do_not_merge(self):
        """贴边但垂直方向完全不重叠 —— 是两个不同的目标。"""
        boxes = np.array([box(0, 0, 30, 50), box(2020, 500, 2047, 600)])
        out = _merge_seam_fragments(boxes, W, H)
        assert len(out) == 2

    def test_horizontally_far_fragments_do_not_merge(self):
        """都贴边但水平间距过大（合并后会是个超宽的离谱框）。"""
        boxes = np.array([box(0, 100, 10, 200), box(1200, 100, 1220, 200)])
        # 右框不贴右边界，因此连候选都算不上
        out = _merge_seam_fragments(boxes, W, H)
        assert len(out) == 2

    def test_only_left_touching(self):
        boxes = np.array([box(0, 100, 30, 200)])
        out = _merge_seam_fragments(boxes, W, H)
        assert len(out) == 1

    def test_only_right_touching(self):
        boxes = np.array([box(2020, 100, 2047, 200)])
        out = _merge_seam_fragments(boxes, W, H)
        assert len(out) == 1


class TestMergeCases:
    def test_left_and_right_fragments_merge(self):
        """★ 核心：贴左与贴右的两个碎片应合并为一个跨接缝的框。"""
        boxes = np.array(
            [
                box(0, 100, 30, 200, conf=0.8),      # 贴在左边界
                box(2020, 105, 2047, 195, conf=0.9),  # 贴在右边界
            ]
        )
        out = _merge_seam_fragments(boxes, W, H)

        assert len(out) == 1, f"应合并为一个框，实际 {len(out)} 个"
        merged = out[0]

        # 接缝连续坐标：右碎片在接缝左侧保持不动，
        # 左碎片整体右移一个周期接到它后面
        assert merged[0] == pytest.approx(2020.0)
        assert merged[2] == pytest.approx(30.0 + W)  # 2078
        # 合并框的宽度应接近两段之和，而非整个图像宽度
        assert (merged[2] - merged[0]) < 0.2 * W

        # 垂直范围取并集
        assert merged[1] == pytest.approx(100.0)
        assert merged[3] == pytest.approx(200.0)
        # 置信度取较高者
        assert merged[4] == pytest.approx(0.9)

    def test_vertical_bounds_take_union(self):
        boxes = np.array([box(0, 150, 40, 260), box(2010, 100, 2047, 220)])
        out = _merge_seam_fragments(boxes, W, H)
        assert len(out) == 1
        assert out[0][1] == pytest.approx(100.0)  # min
        assert out[0][3] == pytest.approx(260.0)  # max

    def test_multiple_pairs_merge_independently(self):
        """接缝处同时有两个目标时，应各自配对合并。"""
        boxes = np.array(
            [
                box(0, 100, 30, 160),        # A 左
                box(2020, 102, 2047, 158),   # A 右
                box(0, 500, 25, 560),        # B 左
                box(2025, 505, 2047, 555),   # B 右
            ]
        )
        out = _merge_seam_fragments(boxes, W, H)
        assert len(out) == 2

    def test_nearest_match_wins(self):
        """一个左碎片面对多个右碎片时，应合并水平间距最小的那个。"""
        boxes = np.array(
            [
                box(0, 100, 20, 200),         # 左碎片
                box(2040, 100, 2047, 200),    # 右碎片 A：gap = |2048-2047| = 1
                box(1990, 105, 2047, 195),    # 右碎片 B：gap = |2048-2047| = 1
            ]
        )
        out = _merge_seam_fragments(boxes, W, H)
        # 只应产生一次合并，剩余一个未配对的框
        assert len(out) == 2

    def test_requires_both_touching_edges(self):
        """右框不贴右边界（距边界超过容差）时不算碎片，不应参与合并。"""
        # x2 = 1990，距右边界 58 像素 > 容差 40，不算贴边
        boxes = np.array([box(0, 100, 30, 200), box(1900, 100, 1990, 200)])
        out = _merge_seam_fragments(boxes, W, H)
        assert len(out) == 2

    def test_gap_tolerance_uses_left_edge(self):
        """左碎片未贴到 x=0 时，间距容差才真正起作用。

        左框 x1=38（< 容差 40，算贴边），右框 x2=2047，
        则 gap = |(38 + 2048) - 2047| = 39 ≤ 40 → 合并。
        """
        boxes = np.array([box(38, 100, 60, 200), box(2010, 100, 2047, 200)])
        out = _merge_seam_fragments(boxes, W, H)
        assert len(out) == 1

    def test_merged_coords_are_continuous_across_seam(self):
        """★ 合并结果必须是接缝连续坐标，而非 [0, ..., width-1]。"""
        boxes = np.array([box(0, 100, 20, 200), box(2018, 100, 2047, 200)])
        out = _merge_seam_fragments(boxes, W, H)
        assert len(out) == 1

        x1, x2 = out[0][0], out[0][2]
        # 右碎片在接缝左侧、左碎片（+width）在其后
        assert x1 == pytest.approx(2018.0)
        assert x2 == pytest.approx(20.0 + W)  # 2068
        # 宽度应接近两段之和（约 50 像素），而不是整个图像宽度
        assert (x2 - x1) < 0.2 * W, "合并框不应横跨整个图像宽度"


class TestDownstreamIntegration:
    """合并后的框应能被正确地转换为跨接缝的球面区域。"""

    def test_merged_box_maps_to_seam_centered_region(self):
        from pano_agent.geometry.region import SphericalBox

        boxes = np.array([box(0, 400, 40, 600), box(2010, 402, 2047, 598)])
        out = _merge_seam_fragments(boxes, W, H)
        assert len(out) == 1

        x1, y1, x2, y2 = out[0][:4]
        sphere = SphericalBox.from_erp_pixels(x1, y1, x2, y2, W, H)

        # 合并框横跨整个宽度，其球面中心应位于接缝附近（yaw ≈ ±180）
        assert abs(abs(sphere.yaw_deg) - 180.0) < 15.0, (
            f"跨接缝目标的球面中心应接近 ±180°，实际 {sphere.yaw_deg:.1f}°"
        )
        # 且它应当同时"包含"接缝两侧的点
        assert bool(sphere.contains(179.0, 0.0)) or bool(sphere.contains(-179.0, 0.0))
