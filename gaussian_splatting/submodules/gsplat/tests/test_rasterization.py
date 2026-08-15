"""Tests for the functions in the CUDA extension.

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

from contextlib import nullcontext
from typing import Optional, Tuple

import pytest
import torch

device = torch.device("cuda:0")


class _ProfilerRecorder:
    def __init__(self):
        self.names = []
        self.values = {}

    def record(self, name):
        self.names.append(name)
        return nullcontext()

    def record_value(self, name, value):
        self.values[name] = self.values.get(name, 0.0) + float(value)


def _make_shared_template_inputs(batch_size=3, gaussians_per_instance=256):
    torch.manual_seed(7)

    B, G = batch_size, gaussians_per_instance
    width, height = 80, 64
    means = torch.empty((B, G, 3), device=device)
    means[..., 0:2] = torch.rand((B, G, 2), device=device) * 0.5 - 0.25
    means[..., 2] = 2.0 + torch.rand((B, G), device=device) * 0.25

    quats = torch.randn((B, G, 4), device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    scales = torch.rand((G, 3), device=device) * 0.015 + 0.01
    opacities = torch.full((G,), 0.8, device=device)
    colors = torch.randn((G, 16, 3), device=device) * 0.2

    viewmats = torch.eye(4, device=device).unsqueeze(0)
    focal = 70.0
    Ks = torch.tensor(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        device=device,
    ).unsqueeze(0)
    backgrounds = torch.zeros((1, 3), device=device)
    return {
        "means": means.reshape(B * G, 3),
        "quats": quats.reshape(B * G, 4),
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
        "gaussians_per_instance": G,
        "viewmats": viewmats,
        "Ks": Ks,
        "backgrounds": backgrounds,
        "width": width,
        "height": height,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("per_view_color", [True, False])
@pytest.mark.parametrize("sh_degree", [None, 3])
@pytest.mark.parametrize("render_mode", ["RGB", "RGB+D", "D"])
@pytest.mark.parametrize("packed", [True, False])
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
def test_rasterization(
    per_view_color: bool,
    sh_degree: Optional[int],
    render_mode: str,
    packed: bool,
    batch_dims: Tuple[int, ...],
):
    from gsplat.rendering import _rasterization, rasterization

    torch.manual_seed(42)

    C, N = 3, 10_000
    means = torch.rand(batch_dims + (N, 3), device=device)
    quats = torch.randn(batch_dims + (N, 4), device=device)
    scales = torch.rand(batch_dims + (N, 3), device=device)
    opacities = torch.rand(batch_dims + (N,), device=device)
    if per_view_color:
        if sh_degree is None:
            colors = torch.rand(batch_dims + (C, N, 3), device=device)
        else:
            colors = torch.rand(
                batch_dims + (C, N, (sh_degree + 1) ** 2, 3), device=device
            )
    else:
        if sh_degree is None:
            colors = torch.rand(batch_dims + (N, 3), device=device)
        else:
            colors = torch.rand(
                batch_dims + (N, (sh_degree + 1) ** 2, 3), device=device
            )

    width, height = 300, 200
    focal = 300.0
    Ks = torch.tensor(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        device=device,
    ).expand(batch_dims + (C, -1, -1))
    viewmats = torch.eye(4, device=device).expand(batch_dims + (C, -1, -1))

    renders, alphas, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=sh_degree,
        render_mode=render_mode,
        packed=packed,
    )

    if render_mode == "D":
        assert renders.shape == batch_dims + (C, height, width, 1)
    elif render_mode == "RGB":
        assert renders.shape == batch_dims + (C, height, width, 3)
    elif render_mode == "RGB+D":
        assert renders.shape == batch_dims + (C, height, width, 4)

    _renders, _alphas, _meta = _rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=sh_degree,
        render_mode=render_mode,
    )
    torch.testing.assert_close(renders, _renders, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(alphas, _alphas, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_profiled_isect_tiles_matches_unprofiled_and_records_stats():
    from gsplat.cuda._wrapper import isect_tiles

    torch.manual_seed(11)
    n_visible = 512
    width, height, tile_size = 80, 64, 16
    tile_width = (width + tile_size - 1) // tile_size
    tile_height = (height + tile_size - 1) // tile_size
    means2d = torch.empty((n_visible, 2), device=device)
    means2d[:, 0] = torch.rand((n_visible,), device=device) * width
    means2d[:, 1] = torch.rand((n_visible,), device=device) * height
    radii = torch.randint(1, 8, (n_visible, 2), device=device, dtype=torch.int32)
    depths = torch.rand((n_visible,), device=device) + 1.0
    image_ids = torch.zeros((n_visible,), device=device, dtype=torch.long)
    gaussian_ids = torch.arange(n_visible, device=device, dtype=torch.long)

    expected = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=True,
        n_images=1,
        image_ids=image_ids,
        gaussian_ids=gaussian_ids,
    )
    profiler = _ProfilerRecorder()
    profiled = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=True,
        n_images=1,
        image_ids=image_ids,
        gaussian_ids=gaussian_ids,
        profiler=profiler,
    )

    for actual, reference in zip(profiled, expected):
        assert torch.equal(actual, reference)

    timing_names = [
        "isect_tiles_count_kernel_ms",
        "isect_tiles_cumsum_ms",
        "isect_tiles_emit_kernel_ms",
        "isect_tiles_sort_ms",
        "isect_tiles_cuda_total_ms",
    ]
    for name in timing_names:
        assert name in profiler.values
        assert profiler.values[name] >= 0.0

    tiles_per_gauss, isect_ids, _ = profiled
    assert profiler.values["isect_visible_gaussians"] == float(tiles_per_gauss.numel())
    assert profiler.values["isect_total_tile_intersections"] == float(isect_ids.numel())
    assert profiler.values["isect_avg_tiles_per_gaussian"] == pytest.approx(
        float(isect_ids.numel()) / float(tiles_per_gauss.numel())
    )
    assert profiler.values["isect_max_tiles_per_gaussian"] == float(
        tiles_per_gauss.max().item()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_ellipse_tile_filter_reduces_corner_aabb_tiles():
    from gsplat.cuda._wrapper import isect_tiles

    width, height, tile_size = 64, 64, 16
    tile_width = (width + tile_size - 1) // tile_size
    tile_height = (height + tile_size - 1) // tile_size
    means2d = torch.tensor([[32.0, 32.0]], device=device)
    radii = torch.tensor([[32, 32]], device=device, dtype=torch.int32)
    depths = torch.tensor([1.0], device=device)
    image_ids = torch.zeros((1,), device=device, dtype=torch.long)
    gaussian_ids = torch.zeros((1,), device=device, dtype=torch.long)
    conics = torch.tensor([[0.1, 0.0, 0.1]], device=device)
    opacities = torch.ones((1,), device=device)

    rect_tiles, rect_ids, _ = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=True,
        n_images=1,
        image_ids=image_ids,
        gaussian_ids=gaussian_ids,
    )
    filtered_tiles, filtered_ids, _ = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=True,
        n_images=1,
        image_ids=image_ids,
        gaussian_ids=gaussian_ids,
        conics=conics,
        opacities=opacities,
        image_width=width,
        image_height=height,
        ellipse_tile_filter=True,
    )

    n_tiles = tile_width * tile_height
    tile_n_bits = int(torch.floor(torch.log2(torch.tensor(float(n_tiles)))).item()) + 1
    tile_mask = (1 << tile_n_bits) - 1
    rect_tile_ids = set(((rect_ids.cpu() >> 32) & tile_mask).tolist())
    filtered_tile_ids = set(((filtered_ids.cpu() >> 32) & tile_mask).tolist())

    assert int(filtered_tiles.sum().item()) == filtered_ids.numel()
    assert int(rect_tiles.sum().item()) == rect_ids.numel()
    assert filtered_ids.numel() < rect_ids.numel()
    assert filtered_tile_ids.issubset(rect_tile_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("rasterize_mode", ["classic", "antialiased"])
def test_shared_template_sh_rgb_fast_path_matches_fallback(rasterize_mode: str):
    from gsplat.rendering import rasterization_shared_template

    kwargs = _make_shared_template_inputs()
    fast_profiler = _ProfilerRecorder()
    fallback_profiler = _ProfilerRecorder()

    fast_renders, fast_alphas, fast_meta = rasterization_shared_template(
        **kwargs,
        sh_degree=3,
        render_mode="RGB+ED",
        rasterize_mode=rasterize_mode,
        batch_image_mode=True,
        profiler=fast_profiler,
    )
    fallback_renders, fallback_alphas, fallback_meta = rasterization_shared_template(
        **kwargs,
        sh_degree=3,
        render_mode="RGB+ED",
        rasterize_mode=rasterize_mode,
        batch_image_mode=True,
        fuse_shared_template_sh=False,
        profiler=fallback_profiler,
    )

    torch.testing.assert_close(fast_renders, fallback_renders, rtol=1e-4, atol=2e-4)
    torch.testing.assert_close(fast_alphas, fallback_alphas, rtol=1e-4, atol=2e-4)
    torch.testing.assert_close(
        fast_meta["depths"], fallback_meta["depths"], rtol=1e-4, atol=1e-4
    )
    assert torch.equal(fast_meta["gaussian_ids"], fallback_meta["gaussian_ids"])
    assert "shared_template_gather_ms" not in fast_profiler.names
    assert "spherical_harmonics_ms" not in fast_profiler.names
    assert "shared_template_gather_ms" in fallback_profiler.names
    assert "spherical_harmonics_ms" in fallback_profiler.names


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("rasterize_mode", ["classic", "antialiased"])
def test_shared_template_ellipse_tile_filter_matches_rectangle(rasterize_mode: str):
    from gsplat.rendering import rasterization_shared_template

    kwargs = _make_shared_template_inputs(batch_size=2, gaussians_per_instance=512)
    rect_renders, rect_alphas, rect_meta = rasterization_shared_template(
        **kwargs,
        sh_degree=3,
        render_mode="RGB+ED",
        rasterize_mode=rasterize_mode,
        batch_image_mode=True,
        ellipse_tile_filter=False,
    )
    filtered_renders, filtered_alphas, filtered_meta = rasterization_shared_template(
        **kwargs,
        sh_degree=3,
        render_mode="RGB+ED",
        rasterize_mode=rasterize_mode,
        batch_image_mode=True,
        ellipse_tile_filter=True,
    )

    torch.testing.assert_close(
        filtered_renders, rect_renders, rtol=1e-4, atol=3e-4
    )
    torch.testing.assert_close(filtered_alphas, rect_alphas, rtol=1e-4, atol=3e-4)
    assert filtered_meta["isect_ids"].numel() <= rect_meta["isect_ids"].numel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_shared_template_sh_rgb_fast_path_falls_back_for_per_view_sh():
    from gsplat.rendering import rasterization_shared_template

    kwargs = _make_shared_template_inputs()
    kwargs["colors"] = kwargs["colors"].unsqueeze(0)
    profiler = _ProfilerRecorder()

    rasterization_shared_template(
        **kwargs,
        sh_degree=3,
        render_mode="RGB+ED",
        rasterize_mode="classic",
        batch_image_mode=True,
        profiler=profiler,
    )

    assert "shared_template_gather_ms" in profiler.names
    assert "spherical_harmonics_ms" in profiler.names
