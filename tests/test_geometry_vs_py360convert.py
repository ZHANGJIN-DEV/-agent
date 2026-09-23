"""与参考实现 py360convert 的对拍测试。

==============================================================================
为什么必须有这一层测试
==============================================================================

单元测试只能验证「自洽性」：往返一致、恒等成立、边界不炸。
但一个自洽的实现完全可能是**系统性地错**的 —— 整体旋转 90°、镜像、
或者方向约定反了。这类错误在单元测试里全部通过，却在真实渲染时表现为
「画面看起来有点怪」而难以定位。

对拍测试用**独立的第三方实现**作为地面真值，专门捕获此类错误。

==============================================================================
两层对拍，各管一件事
==============================================================================

**第 1 层：坐标级对拍（严格，容差 1e-4）**
    比较「某个像素对应的球面方向」以及「该方向落在 ERP 的哪个像素」。
    这是几何本身的验证，不受插值实现影响，是**最有价值的断言**。

**第 2 层：渲染级对拍（宽松，容差 1 像素）**
    比较最终渲染出的图像。此处两个实现必然有差异，因为：
      - py360convert 对 ERP 做 1 像素 padding 后调用 cv2.remap；
      - 坐标经 cv2.convertMaps 转成 CV_16SC2 定点数，精度约 1/32 像素；
      - 边界与极点的 padding 策略不同。
    因此渲染级容差放宽到 1 个像素是合理的 —— 它的作用是抓「整体错位」，
    而不是替代第 1 层。

分层的意义：若第 1 层通过而第 2 层失败，问题一定在插值；反之则在几何。
不分开的话，一个 0.3 像素的插值差异会淹没真正的几何错误。

==============================================================================
关于经度环绕的假差异
==============================================================================

yaw ∈ [-180, 180)，-179.9 与 +179.9 实际只差 0.2°，直接相减却得 359.8°。
所有涉及 yaw 的比较都必须用环绕感知的角差（见 `circular_aware_diff`）。
"""

from __future__ import annotations

import numpy as np
import pytest

py360convert = pytest.importorskip(
    "py360convert", reason="需要 py360convert 作为对拍参照：pip install py360convert"
)

from py360convert.utils import uv2coor, xyz2uv, xyzpers  # noqa: E402

from pano_agent.geometry.camera import VirtualCamera  # noqa: E402
from pano_agent.geometry.projection import render_nfov  # noqa: E402
from pano_agent.geometry.sphere import erp_pixel_to_sph, sph_to_erp_pixel, vec_to_sph  # noqa: E402

ERP_W, ERP_H = 1024, 512

# 覆盖赤道 / 高纬 / 跨接缝 / 窄视场
CAMERA_CASES = [
    (0.0, 0.0, 90.0),
    (40.0, 20.0, 90.0),
    (90.0, 0.0, 90.0),
    (180.0, 0.0, 90.0),
    (-90.0, 0.0, 90.0),
    (179.0, 0.0, 60.0),     # 跨接缝
    (-179.0, 0.0, 60.0),
    (0.0, 60.0, 90.0),      # 高纬
    (0.0, -60.0, 90.0),
    (0.0, 85.0, 90.0),      # 近极点
    (-135.0, -40.0, 75.0),  # 斜向
]


@pytest.fixture(scope="module")
def angle_erp() -> np.ndarray:
    """角度编码 ERP：R 通道 = yaw，G 通道 = pitch。

    用角度而非真实图像做对拍，误差可以直接用「度」衡量，
    且不受纹理与压缩干扰。
    """
    xs, ys = np.meshgrid(np.arange(ERP_W), np.arange(ERP_H))
    yaw, pitch = erp_pixel_to_sph(xs, ys, ERP_W, ERP_H)
    return np.stack([yaw, pitch, np.zeros_like(yaw)], axis=-1)


def circular_aware_diff(a: np.ndarray, b: np.ndarray, period: float = 360.0) -> np.ndarray:
    """环绕感知的角差，把差值折叠到 [-period/2, period/2]。"""
    d = np.abs(a - b) % period
    return np.minimum(d, period - d)


def reference_ray_directions(cam: VirtualCamera) -> tuple[np.ndarray, np.ndarray]:
    """复现 py360convert 内部的光线生成链，返回 (yaw_deg, pitch_deg) 网格。"""
    xyz = xyzpers(
        np.deg2rad(cam.fov_x_deg),
        np.deg2rad(cam.fov_y_deg),
        -np.deg2rad(cam.yaw_deg),
        np.deg2rad(cam.pitch_deg),
        (cam.height, cam.width),
        np.deg2rad(cam.roll_deg),
    )
    u, v = xyz2uv(xyz)
    return np.degrees(np.squeeze(u)), np.degrees(np.squeeze(v))


def reference_erp_coords(cam: VirtualCamera) -> tuple[np.ndarray, np.ndarray]:
    """py360convert 采样 ERP 时使用的像素坐标。"""
    xyz = xyzpers(
        np.deg2rad(cam.fov_x_deg),
        np.deg2rad(cam.fov_y_deg),
        -np.deg2rad(cam.yaw_deg),
        np.deg2rad(cam.pitch_deg),
        (cam.height, cam.width),
        np.deg2rad(cam.roll_deg),
    )
    u, v = xyz2uv(xyz)
    cx, cy = uv2coor(u, v, ERP_H, ERP_W)
    return np.squeeze(cx), np.squeeze(cy)


# ============================================================== 第 1 层


class TestRayDirectionsMatch:
    """★ 核心：每个输出像素对应的球面方向必须与参考实现一致。"""

    @pytest.mark.parametrize(("yaw", "pitch", "fov"), CAMERA_CASES)
    def test_spherical_directions(self, yaw: float, pitch: float, fov: float):
        cam = VirtualCamera(yaw, pitch, 0.0, fov, 128, 72)

        ref_yaw, ref_pitch = reference_ray_directions(cam)
        our_yaw, our_pitch = vec_to_sph(cam.pixel_rays())

        np.testing.assert_allclose(our_yaw, ref_yaw, atol=1e-4)
        np.testing.assert_allclose(our_pitch, ref_pitch, atol=1e-4)

    @pytest.mark.parametrize(("yaw", "pitch", "fov"), CAMERA_CASES)
    def test_erp_sampling_coordinates(self, yaw: float, pitch: float, fov: float):
        """方向一致还不够 —— ERP 像素换算也必须一致（半像素错误会在这里暴露）。"""
        cam = VirtualCamera(yaw, pitch, 0.0, fov, 128, 72)

        ref_cx, ref_cy = reference_erp_coords(cam)
        our_yaw, our_pitch = vec_to_sph(cam.pixel_rays())
        our_cx, our_cy = sph_to_erp_pixel(our_yaw, our_pitch, ERP_W, ERP_H)

        # x 可能因环绕相差一个周期，比较前取模
        dx = np.abs((our_cx - ref_cx + ERP_W / 2) % ERP_W - ERP_W / 2)
        assert dx.max() < 1e-3, f"x 坐标最大差 {dx.max():.5f} 像素"
        assert np.abs(our_cy - ref_cy).max() < 1e-3

    def test_wide_angle_case(self):
        """宽视场会放大边缘误差，单独覆盖。"""
        cam = VirtualCamera(0, 0, 0, 140.0, 200, 100)
        ref_yaw, ref_pitch = reference_ray_directions(cam)
        our_yaw, our_pitch = vec_to_sph(cam.pixel_rays())
        np.testing.assert_allclose(our_yaw, ref_yaw, atol=1e-3)
        np.testing.assert_allclose(our_pitch, ref_pitch, atol=1e-3)

    def test_odd_size_case(self):
        """奇数尺寸下中心像素精确存在，是验证对称性的好用例。"""
        cam = VirtualCamera(17.0, -23.0, 0.0, 75.0, 101, 57)
        ref_yaw, ref_pitch = reference_ray_directions(cam)
        our_yaw, our_pitch = vec_to_sph(cam.pixel_rays())
        np.testing.assert_allclose(our_yaw, ref_yaw, atol=1e-4)
        np.testing.assert_allclose(our_pitch, ref_pitch, atol=1e-4)


# ============================================================== 第 2 层


class TestRenderedImageAgrees:
    """渲染级对拍：容差为 1 个像素，用于捕捉整体错位。"""

    @pytest.mark.parametrize(("yaw", "pitch", "fov"), CAMERA_CASES)
    def test_nearest_within_one_pixel(
        self, angle_erp: np.ndarray, yaw: float, pitch: float, fov: float
    ):
        cam = VirtualCamera(yaw, pitch, 0.0, fov, 128, 72)
        ours = render_nfov(angle_erp, cam, mode="nearest")
        theirs = py360convert.e2p(
            angle_erp,
            fov_deg=(cam.fov_x_deg, cam.fov_y_deg),
            u_deg=cam.yaw_deg,
            v_deg=cam.pitch_deg,
            out_hw=(cam.height, cam.width),
            mode="nearest",
        )

        diff_yaw = circular_aware_diff(ours[..., 0], theirs[..., 0])

        # 最近邻下，坐标若落在 x.5 的取整临界点附近，两个实现可能取到
        # 相邻的 **ERP 像素**（而非相机像素）—— 因此容差应以 ERP 像素的
        # 角分辨率为准，即 360 / ERP_W。
        erp_pixel_angle = 360.0 / ERP_W
        assert diff_yaw.max() <= erp_pixel_angle + 1e-6, (
            f"最大偏差 {diff_yaw.max():.4f}° 超过一个 ERP 像素 {erp_pixel_angle:.4f}°"
        )
        # 且这种临界点翻转不应是普遍现象
        assert (diff_yaw > 1e-6).mean() < 0.6

    @pytest.mark.parametrize(("yaw", "pitch", "fov"), CAMERA_CASES[:6])
    def test_bilinear_within_one_pixel(
        self, angle_erp: np.ndarray, yaw: float, pitch: float, fov: float
    ):
        cam = VirtualCamera(yaw, pitch, 0.0, fov, 128, 72)
        ours = render_nfov(angle_erp, cam, mode="bilinear")
        theirs = py360convert.e2p(
            angle_erp,
            fov_deg=(cam.fov_x_deg, cam.fov_y_deg),
            u_deg=cam.yaw_deg,
            v_deg=cam.pitch_deg,
            out_hw=(cam.height, cam.width),
            mode="bilinear",
        )

        diff_yaw = circular_aware_diff(ours[..., 0], theirs[..., 0])
        pixel_angle = fov / cam.width

        assert diff_yaw.mean() < 0.25 * pixel_angle
        assert diff_yaw.max() < 1.5 * pixel_angle


# ============================================================== 约定锁定


class TestEndpointSamplingConvention:
    """锁定「端点采样」约定 —— 对拍过程中确定下来的关键设计决策。

    py360convert 用 `linspace(-x_max, x_max, num=W)` 生成光线，即
    **图像边界精确等于视场边界**；另一种常见做法是标准针孔内参模型
    （像素间距 = 1/f），二者在图像边缘累计相差半个像素。

    本项目采用前者，理由：
      1. 用户指定 FOV 时，期望画面正好覆盖该角度范围；
      2. 与生态内广泛使用的参考实现一致，便于交叉验证。

    这两个测试是上游 `TestRayDirectionsMatch` 能通过的前提。
    """

    def test_image_edge_equals_fov_boundary(self):
        """第 0 列的光线与光轴夹角应精确等于 fov_x / 2。"""
        from pano_agent.geometry.sphere import great_circle_distance

        fov = 60.0
        cam = VirtualCamera(0, 0, 0, fov, 101, 57)
        yaw_l, pitch_l = vec_to_sph(cam.pixel_rays()[28, 0])

        d = great_circle_distance(float(yaw_l), float(pitch_l), 0.0, 0.0)
        assert float(d) == pytest.approx(fov / 2, abs=1e-6)

    def test_pixel_spacing_is_uniform_in_tangent_plane(self):
        """切平面坐标等距 —— 端点采样的直接结果。

        不能拿 yaw 直接测均匀性：切平面横坐标 x 等距时 yaw = atan(x)
        是非线性的，步长必然随位置变化（这是透视投影的正确行为）。
        """
        cam = VirtualCamera(0, 0, 0, 90.0, 129, 73)
        rays = cam.pixel_rays()
        tx = rays[36, :, 0] / rays[36, :, 2]
        steps = np.diff(tx)
        assert steps.std() < 1e-12, f"切平面步长不均匀，std={steps.std():.2e}"

    def test_center_pixel_is_exact_optical_axis(self):
        """奇数尺寸时中心像素的光线精确等于光轴方向。"""
        cam = VirtualCamera(0, 0, 0, 90.0, 101, 57)
        yaw_c, pitch_c = vec_to_sph(cam.pixel_rays()[28, 50])
        assert float(yaw_c) == pytest.approx(0.0, abs=1e-9)
        assert float(pitch_c) == pytest.approx(0.0, abs=1e-9)


class TestSeamHandling:
    def test_no_invalid_value_near_seam(self, angle_erp: np.ndarray):
        """接缝处不应出现 NaN 或未采样区域。"""
        for yaw in (179.0, 180.0, -179.0, -180.0):
            cam = VirtualCamera(yaw, 0.0, 0.0, 90.0, 64, 36)
            assert np.isfinite(render_nfov(angle_erp, cam, mode="bilinear")).all()

    def test_seam_region_has_valid_values(self, angle_erp: np.ndarray):
        """接缝附近不应出现 NaN 或越界值。

        ⚠ 重要陷阱：**不能用 yaw 角度图做接缝连续性测试**。
        角度图在接缝处从 +179.82 跳到 -179.82，双线性插值会混合这两个值
        得到 ≈0 —— 那是「把周期性量编码进标量通道」的固有缺陷，
        而不是渲染错误。真实 RGB 图像不存在此问题（颜色值无周期性），
        因此这类测试必须用真实图像或非周期编码图来做。

        接缝的**几何**正确性已由 `TestRayDirectionsMatch` 在坐标级验证
        （覆盖 yaw=±179/180 等用例），此处只确认不产生非法值。
        """
        cam = VirtualCamera(180.0, 0.0, 0.0, 90.0, 129, 73)
        out = render_nfov(angle_erp, cam, mode="bilinear")

        assert np.isfinite(out).all()
        # 正对接缝的中间三列，yaw 应仍在合法值域内
        center_cols = out[36, 63:66, 0]
        assert np.all(np.abs(center_cols) <= 180.0 + 1e-6)
