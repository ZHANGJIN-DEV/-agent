"""实例提取器的测试。

这些测试用**合成实例图**（而不是真实数据），因为要精确控制
实例的边界、类别、跨接缝位置 —— 真实数据的正确性无法逐像素断言。
"""

from __future__ import annotations

import numpy as np
import pytest

from pano_agent.perception.instance import (
    BACKGROUND_CLASSES,
    CLASS_NAMES,
    InstanceExtractor,
    InstanceExtractorConfig,
)

W, H = 512, 256


def make_instance_map() -> np.ndarray:
    """构造一张合成的实例图（RGBA）。

    格式与真实数据一致：
        A 通道 = 类别 ID
        RGB    = (类别, 实例) 的联合彩色编码
    """
    arr = np.zeros((H, W, 4), dtype=np.uint8)

    # 类别 24 (building)，两个实例，不同颜色
    arr[20:80, 50:150, :3] = [200, 100, 50]     # 实例 A
    arr[20:80, 50:150, 3] = 24
    arr[20:80, 300:400, :3] = [50, 200, 100]    # 实例 B（同类不同色）
    arr[20:80, 300:400, 3] = 24

    # 类别 10 (lamp)，一个实例
    arr[120:160, 200:240, :3] = [10, 20, 30]
    arr[120:160, 200:240, 3] = 10

    # 背景类：类别 3 (sky)
    #
    # 注意尺寸：**不能**写成 `arr[0:20, :]` —— 那是 20×512 的横条，
    # 长宽比 25.6 会被 max_aspect 过滤掉（这是**正确**行为：
    # 全景图里横贯整个宽度的细长条多半是高纬畸变产物）。
    # 这里给一个尺寸合理的背景区域，以便测试"允许包含背景"的开关。
    arr[0:60, 100:400, :3] = [0, 0, 255]
    arr[0:60, 100:400, 3] = 3

    return arr


class TestBasicExtraction:
    def test_extracts_instances(self):
        ext = InstanceExtractor()
        dets = ext.extract(make_instance_map())
        assert len(dets) == 3, f"应有 3 个前景实例，实际 {len(dets)}"

    def test_background_excluded_by_default(self):
        ext = InstanceExtractor()
        dets = ext.extract(make_instance_map())
        assert all(d.category != "sky" for d in dets)

    def test_background_included_when_asked(self):
        ext = InstanceExtractor(InstanceExtractorConfig(include_background=True))
        dets = ext.extract(make_instance_map())
        assert any(d.category == "sky" for d in dets)

    def test_same_class_instances_are_separate(self):
        """★ 同类不同实例必须被分开 —— 这正是实例分割相对语义分割的价值。"""
        ext = InstanceExtractor()
        dets = ext.extract(make_instance_map())
        buildings = [d for d in dets if d.category == "building"]
        assert len(buildings) == 2, "两个 building 应被识别为两个独立实例"

        # 两者的球面位置应明显不同
        import math

        d = math.hypot(buildings[0].yaw_deg - buildings[1].yaw_deg,
                       buildings[0].pitch_deg - buildings[1].pitch_deg)
        assert d > 10.0, "两个实例应在不同位置"

    def test_sorted_by_confidence(self):
        ext = InstanceExtractor()
        dets = ext.extract(make_instance_map())
        confs = [d.confidence for d in dets]
        assert confs == sorted(confs, reverse=True)


class TestSphericalConversion:
    def test_outputs_are_spherical(self):
        """输出必须是球面角度，不是像素。"""
        ext = InstanceExtractor()
        for d in ext.extract(make_instance_map()):
            assert -180.0 <= d.yaw_deg < 180.0
            assert -90.0 <= d.pitch_deg <= 90.0
            assert d.width_deg > 0 and d.height_deg > 0

    def test_bigger_instance_larger_angular_size(self):
        arr = np.zeros((H, W, 4), dtype=np.uint8)
        arr[20:60, 20:60, :3] = [10, 20, 30]      # 小
        arr[20:60, 20:60, 3] = 24
        arr[100:200, 100:300, :3] = [40, 50, 60]  # 大
        arr[100:200, 100:300, 3] = 24

        dets = InstanceExtractor().extract(arr)
        assert len(dets) == 2
        small, big = min(dets, key=lambda d: d.width_deg), max(dets, key=lambda d: d.width_deg)
        assert big.width_deg > small.width_deg

    def test_no_pixel_units_leak(self):
        """角度值不应出现像素级的小数目（那说明单位搞错了）。"""
        ext = InstanceExtractor()
        for d in ext.extract(make_instance_map()):
            # 512 宽的图，20 像素宽的实例约 14°；若输出 20 说明用了像素
            assert d.width_deg > 1.0, f"角宽 {d.width_deg} 过小，疑似用了像素单位"


class TestConfigFiltering:
    def test_min_area_filters_small_regions(self):
        arr = np.zeros((H, W, 4), dtype=np.uint8)
        arr[10:14, 10:14, :3] = [1, 2, 3]   # 16 像素
        arr[10:14, 10:14, 3] = 24
        arr[100:140, 100:140, :3] = [4, 5, 6]
        arr[100:140, 100:140, 3] = 24

        ext = InstanceExtractor(InstanceExtractorConfig(min_area_px=300))
        dets = ext.extract(arr)
        assert len(dets) == 1, "小区域应被面积阈值过滤"

    def test_max_boxes_per_class_limits_output(self):
        arr = np.zeros((H, W, 4), dtype=np.uint8)
        for i in range(6):
            x = 20 + i * 70
            arr[50:120, x : x + 50, :3] = [i * 20 + 10, 100, 50]
            arr[50:120, x : x + 60, 3] = 24

        ext = InstanceExtractor(InstanceExtractorConfig(max_boxes_per_class=2))
        dets = ext.extract(arr)
        buildings = [d for d in dets if d.category == "building"]
        assert len(buildings) <= 2

    def test_max_aspect_filters_slivers(self):
        """过度细长的区域（高纬畸变条带）应被过滤。"""
        arr = np.zeros((H, W, 4), dtype=np.uint8)
        arr[10:14, 10:500, :3] = [1, 2, 3]  # 长宽比 ~120
        arr[10:14, 10:500, 3] = 24
        arr[100:160, 100:160, :3] = [4, 5, 6]
        arr[100:160, 100:160, 3] = 24

        ext = InstanceExtractor(InstanceExtractorConfig(max_aspect=20.0))
        dets = ext.extract(arr)
        assert all(
            max(d.width_deg, d.height_deg) / max(min(d.width_deg, d.height_deg), 1e-6) < 100
            for d in dets
        ), "细长条带应被过滤"


class TestSeamHandling:
    def test_cross_seam_instance_is_one_detection(self):
        """★ 跨越 ERP 接缝的实例必须被识别为**一个**目标。

        朴素做法会把它当成两个（图像最左和最右各一块），
        这正是 360VOT 定义的 CROSS_BOUNDARY 挑战。
        """
        arr = np.zeros((H, W, 4), dtype=np.uint8)
        # 左边缘一部分
        arr[80:140, 0:40, :3] = [200, 50, 50]
        arr[80:140, 0:40, 3] = 24
        # 右边缘一部分（同一实例的另一半）
        arr[80:140, W - 40 : W, :3] = [200, 50, 50]
        arr[80:140, W - 40 : W, 3] = 24

        dets = InstanceExtractor().extract(arr)
        assert len(dets) == 1, (
            f"跨接缝的实例应合并为 1 个，实际 {len(dets)} 个 —— "
            f"环形 padding 或接缝连续坐标失效"
        )

        # 其球面中心应接近 ±180°
        assert abs(abs(dets[0].yaw_deg) - 180.0) < 30.0, (
            f"跨接缝实例的中心应接近 ±180°，实际 {dets[0].yaw_deg:.1f}°"
        )


class TestRobustness:
    def test_empty_input(self):
        arr = np.zeros((H, W, 4), dtype=np.uint8)
        arr[..., :3] = [1, 2, 3]
        arr[..., 3] = 3  # 全是 sky
        assert InstanceExtractor().extract(arr) == []

    def test_unknown_class_ignored(self):
        arr = np.zeros((H, W, 4), dtype=np.uint8)
        arr[50:150, 50:150, :3] = [1, 2, 3]
        arr[50:150, 50:150, 3] = 200  # 不在类别表里
        assert InstanceExtractor().extract(arr) == []

    def test_rejects_bad_shape(self):
        with pytest.raises(ValueError, match="RGBA"):
            InstanceExtractor().extract(np.zeros((10, 10, 3), dtype=np.uint8))

    def test_black_rgb_is_a_valid_instance(self):
        """纯黑 (0,0,0) 是合法的实例颜色，不应被误当作"空"。"""
        arr = np.zeros((H, W, 4), dtype=np.uint8)
        arr[50:150, 50:150, :3] = [0, 0, 0]
        arr[50:150, 50:150, 3] = 24
        assert len(InstanceExtractor().extract(arr)) == 1


class TestClassTables:
    def test_background_classes_are_known(self):
        for cid in BACKGROUND_CLASSES:
            assert cid in CLASS_NAMES, f"背景类 {cid} 应在类别表中"

    def test_nyc_classes_present(self):
        """Omni360-X NYC 的类别应完整登记。"""
        for name in ("building", "road", "sky", "pillar", "lamp", "trashcan", "plant"):
            assert name in CLASS_NAMES.values(), f"缺少类别 {name}"
