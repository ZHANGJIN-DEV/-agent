"""渲染层：把相机轨迹应用到 ERP 全景，输出 NFOV 成片。"""

from .nfov import NfovRenderer, RenderConfig

__all__ = ["NfovRenderer", "RenderConfig"]
