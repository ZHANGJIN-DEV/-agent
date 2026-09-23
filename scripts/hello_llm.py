"""连通性自检：验证 LLM / VLM 提供商的配置是否正确。

W0 的验收项之一 —— 在写任何 Agent 逻辑之前，先确认「能调通模型」，
否则后面调试编排代码时无法区分是逻辑错还是配置错。

用法:
    python scripts/hello_llm.py                  # 检测 .env 中配置的默认提供商
    python scripts/hello_llm.py --all            # 检测所有已配置密钥的平台
    python scripts/hello_llm.py --vision         # 额外验证 VLM 能否看图
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许在 pip install -e 之前直接从源码目录运行
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pano_agent.agent.providers import ChatMessage, ProviderError, create_provider  # noqa: E402
from pano_agent.agent.providers.base import BaseProvider  # noqa: E402
from pano_agent.config import get_settings  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def _ok(msg: str) -> str:
    return f"{GREEN}✓{RESET} {msg}"


def _fail(msg: str) -> str:
    return f"{RED}✗{RESET} {msg}"


def _warn(msg: str) -> str:
    return f"{YELLOW}!{RESET} {msg}"


def test_llm(provider: BaseProvider) -> bool:
    """发一轮对话，验证连通性与 JSON mode。"""
    print(f"\n  [LLM] {provider.name}")
    print(f"  {DIM}模型: {getattr(provider, '_default_model', 'n/a')}{RESET}")

    # --- 基础对话 ---
    try:
        result = provider.chat(
            [ChatMessage.user("Reply with exactly: OK")],
            max_tokens=16,
            temperature=0.0,
        )
    except ProviderError as exc:
        print(f"    {_fail(f'调用失败: {exc}')}")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"    {_fail(f'{type(exc).__name__}: {exc}')}")
        return False

    print(f"    {_ok(f'对话成功 → {result.text.strip()[:60]!r}')}")
    print(f"    {DIM}实际模型: {result.model} | tokens: {result.usage.total_tokens}{RESET}")

    # --- JSON mode（Agent 规划器的可靠性基础）---
    try:
        js = provider.chat(
            [ChatMessage.user('Return JSON: {"status": "ok", "value": 42}')],
            max_tokens=64,
            temperature=0.0,
            json_mode=True,
        )
        import json

        parsed = json.loads(js.text)
        print(f"    {_ok(f'JSON mode 正常 → {parsed}')}")
    except Exception as exc:  # noqa: BLE001
        print(f"    {_warn(f'JSON mode 异常（非致命）: {type(exc).__name__}: {exc}')}")

    return True


def test_vlm(provider: BaseProvider) -> bool:
    """发一张合成图，验证视觉输入能否被正确处理。"""
    print(f"\n  [VLM] {provider.name}")

    if not provider.supports_vision:
        print(f"    {_warn('该提供商不支持视觉输入，跳过')}")
        return False

    image_path = _make_test_image()

    try:
        from pano_agent.agent.providers import ImagePart, TextPart

        result = provider.chat(
            [
                ChatMessage.user([
                    TextPart("这张图里有什么形状？只回答形状名称。"),
                    ImagePart(path=image_path),
                ])
            ],
            max_tokens=64,
            temperature=0.0,
        )
    except ProviderError as exc:
        print(f"    {_fail(f'调用失败: {exc}')}")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"    {_fail(f'{type(exc).__name__}: {exc}')}")
        return False

    print(f"    {_ok(f'看图成功 → {result.text.strip()[:80]!r}')}")
    print(f"    {DIM}tokens: {result.usage.total_tokens}{RESET}")
    return True


def _make_test_image() -> Path:
    """生成一张纯合成的测试图（红底蓝方块），避免依赖任何真实数据。"""
    out_dir = Path(__file__).resolve().parents[1] / ".cache"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "hello_vlm_test.png"
    if out.exists():
        return out

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        # 没有 PIL 时写一个最小的合法 PNG（1x1 红色像素）
        import base64

        png_1px_red = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
            b"\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        out.write_bytes(png_1px_red)
        return out

    img = Image.new("RGB", (256, 256), (220, 40, 40))
    draw = ImageDraw.Draw(img)
    draw.rectangle([64, 64, 192, 192], fill=(40, 80, 220))
    img.save(out)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM / VLM 连通性自检")
    parser.add_argument("--all", action="store_true", help="检测所有已配置密钥的平台")
    parser.add_argument("--vision", action="store_true", help="额外验证 VLM 看图能力")
    parser.add_argument("--llm", default=None, help="指定 LLM 提供商")
    parser.add_argument("--vlm", default=None, help="指定 VLM 提供商")
    args = parser.parse_args(argv)

    settings = get_settings()

    print("\n" + "=" * 62)
    print("  PanoDirector 连通性自检")
    print("=" * 62)

    # 密钥概览
    print("\n  已配置的密钥:")
    for label, has in [
        ("DEEPSEEK_API_KEY", bool(settings.deepseek_api_key.get_secret_value())),
        ("DASHSCOPE_API_KEY (通义)", bool(settings.dashscope_api_key.get_secret_value())),
        ("ZHIPU_API_KEY (智谱)", bool(settings.zhipu_api_key.get_secret_value())),
    ]:
        print(f"    {_ok(label) if has else _warn(f'{label} 未配置')}")

    results: list[tuple[str, bool]] = []

    llm_targets = ["deepseek", "qwen", "zhipu"] if args.all else [args.llm or settings.llm_provider]
    for name in dict.fromkeys(t for t in llm_targets if t):
        if not settings.has_llm_credentials(name):
            print(f"\n  [LLM] {name}\n    {_warn('无密钥，跳过')}")
            continue
        try:
            results.append((f"LLM/{name}", test_llm(create_provider("llm", name, settings))))
        except Exception as exc:  # noqa: BLE001
            print(f"\n  [LLM] {name}\n    {_fail(f'构造失败: {exc}')}")
            results.append((f"LLM/{name}", False))

    vlm_targets = ["qwen", "zhipu"] if args.all else [args.vlm or settings.vlm_provider]
    for name in dict.fromkeys(t for t in vlm_targets if t):
        if name == "mock":
            print(f"\n  [VLM] mock\n    {_warn('MockProvider 离线模式（在 .env 中设置 VLM key 后可测真实调用）')}")
            continue
        if not settings.has_vlm_credentials(name):
            print(f"\n  [VLM] {name}\n    {_warn('无密钥，跳过')}")
            continue
        try:
            ok = test_vlm(create_provider("vlm", name, settings)) if args.vision else True
            results.append((f"VLM/{name}", ok))
        except Exception as exc:  # noqa: BLE001
            print(f"\n  [VLM] {name}\n    {_fail(f'构造失败: {exc}')}")
            results.append((f"VLM/{name}", False))

    # 汇总
    print("\n" + "-" * 62)
    if results:
        for label, ok in results:
            print(f"  {_ok(label) if ok else _fail(label)}")
        passed = sum(1 for _, ok in results if ok)
        print("-" * 62)
        print(f"  通过 {passed}/{len(results)}")
    else:
        print("  没有可测试的提供商（全部无密钥或已跳过）")

    print("\n  下一步:")
    print("    DeepSeek 无视觉模型，VLM 层需要通义或智谱的 key →")
    print("      通义: https://dashscope.console.aliyun.com/")
    print("      智谱: https://open.bigmodel.cn/  （glm-4v-flash 免费）")
    print("    填入 .env 的 DASHSCOPE_API_KEY 或 ZHIPU_API_KEY 后重新运行本脚本\n")

    return 0 if all(ok for _, ok in results) or not results else 1


if __name__ == "__main__":
    sys.exit(main())
