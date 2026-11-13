"""闸门测试：质量指标要能区分「有内容的图」和「噪声/纯色/垃圾」。"""

from __future__ import annotations

import io
from dataclasses import replace

import numpy as np
from neko_image_embed.engine.config import load_settings
from neko_image_embed.engine.guard import (
    inspect_image,
    is_blocked_target,
    measure,
    path_roots_refusal,
    resolve_persona_scope,
)
from PIL import Image, ImageFilter

LIMITS = load_settings({}).limits


def _noise(size=(400, 300), seed: int = 7) -> Image.Image:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(array, "RGB")


def _jpeg(image: Image.Image, quality: int = 90) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def test_detailed_image_passes() -> None:
    result = inspect_image(_jpeg(_noise()), LIMITS)
    assert result.ok, result.detail
    assert result.metrics is not None
    assert result.metrics.unique_colors > 100
    assert result.metrics.sharpness > LIMITS.min_sharpness


def test_solid_colour_is_rejected() -> None:
    flat = Image.new("RGB", (400, 300), (120, 120, 120))
    result = inspect_image(_jpeg(flat), LIMITS)
    assert not result.ok
    assert result.code == "low_entropy"


def test_blurred_image_is_rejected_when_threshold_is_high() -> None:
    blurred = _noise().filter(ImageFilter.GaussianBlur(radius=8))
    strict = replace(LIMITS, min_sharpness=10_000.0)
    result = inspect_image(_jpeg(blurred), strict)
    assert not result.ok
    assert result.code == "blurry"


def test_tiny_image_rejected() -> None:
    result = inspect_image(_jpeg(Image.new("RGB", (8, 8), (0, 0, 0))), LIMITS)
    assert not result.ok
    assert result.code == "too_small"


def test_garbage_bytes_rejected() -> None:
    result = inspect_image(b"not an image at all", LIMITS)
    assert not result.ok
    assert result.code == "not_image"


def test_oversized_bytes_rejected_before_decoding() -> None:
    strict = replace(LIMITS, max_source_bytes=1024)
    result = inspect_image(b"x" * 2048, strict)
    assert not result.ok
    assert result.code == "too_large"


def test_animation_is_rejected_by_default() -> None:
    rng = np.random.default_rng(4)
    frames = [
        Image.fromarray(rng.integers(0, 256, (96, 96, 3), dtype=np.uint8), "RGB")
        for _ in range(3)
    ]
    buffer = io.BytesIO()
    frames[0].save(buffer, format="GIF", save_all=True, append_images=frames[1:], duration=100)
    result = inspect_image(buffer.getvalue(), LIMITS, allow_animated=False)
    assert not result.ok
    assert result.code == "animated_not_allowed"

    allowed = inspect_image(buffer.getvalue(), LIMITS, allow_animated=True)
    assert allowed.ok, allowed.detail
    assert allowed.flattened is True


def test_metrics_are_stable_for_same_pixels() -> None:
    image = _noise(seed=3)
    first = measure(image)
    second = measure(image)
    assert first.dhash == second.dhash
    assert first.pixel_sha == second.pixel_sha

def test_internal_url_targets_are_blocked() -> None:
    blocked = ("127.0.0.1", "::1", "10.0.0.7", "192.168.1.1", "169.254.169.254",
               "localhost", "db.internal", "[fd00::1]", "",
               "127.1", "2130706433", "0x7f000001", "100.64.0.1",
               "[::ffff:127.0.0.1]", "[::ffff:100.100.100.200]")
    for host in blocked:
        assert is_blocked_target(host) is True, host
    for host in ("93.184.216.34", "example.com", "img.example.test",
                 "2001:4860:4860::8888", "cafe.babe"):
        assert is_blocked_target(host) is False, host


def test_path_roots_gate(tmp_path) -> None:
    inside = tmp_path / "pics" / "a.jpg"
    inside.parent.mkdir(parents=True, exist_ok=True)
    inside.write_bytes(b"x")
    outside = tmp_path / "secret.jpg"
    outside.write_bytes(b"x")
    assert path_roots_refusal(inside, ()) == ""
    assert path_roots_refusal(inside, (str(inside.parent),)) == ""
    assert path_roots_refusal(outside, (str(inside.parent),)) != ""


def test_persona_scope_gate() -> None:
    assert resolve_persona_scope("", "YUI", True) == ("YUI", False)
    assert resolve_persona_scope("YUI", "YUI", False) == ("YUI", False)
    assert resolve_persona_scope("阿香", "YUI", True) == ("阿香", False)
    assert resolve_persona_scope("阿香", "YUI", False) == ("YUI", True)
    # 当前角色未知时开关关着，也不许自己点档口
    assert resolve_persona_scope("阿香", "", False) == ("", True)


def test_new_gates_default_to_old_behaviour() -> None:
    settings = load_settings({})
    assert settings.allow_cross_persona is True
    assert settings.path_roots == ()
    custom = load_settings({"image_embed": {
        "allow_cross_persona": False,
        "path_roots": ["D:/pics", "  "],
    }})
    assert custom.allow_cross_persona is False
    assert custom.path_roots == ("D:/pics",)
