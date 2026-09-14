"""Check renderer storage ownership and compositing with strided RGB/alpha."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import gaussian_splatting.gaussian_renderer as renderer
from qqtt.engine.trainer_warp import InvPhyTrainerWarp


def reference(rgba, overlay):
    images = rgba.permute(0, 2, 3, 1).detach().clamp(0, 1)
    mask = (images[..., :3] != 1.0).any(dim=3) & (images[..., 3] > 100 / 255)
    alpha = torch.where(mask[..., None], images[..., 3:4], 0.0)
    return overlay.unsqueeze(0) * (1.0 - alpha) + images[..., :3] * alpha * 255.0, mask


def inputs(batch, device):
    generator = torch.Generator().manual_seed(142)
    colors = torch.rand((batch, 1, 7, 34, 3), generator=generator).to(device)
    alphas = torch.rand((batch, 1, 7, 34, 2), generator=generator).to(device)
    # Exercise row/column strides and nonzero storage offsets for every input.
    colors = colors[:, :, 1:6, ::2]
    alphas = alphas[:, :, 1:6, ::2, :1]
    colors[:, 0, 0, :8] = torch.tensor([
        [1, 1, 1], [0.2, 0.3, 0.4], [0.2, 0.3, 0.4],
        [0.2, 0.3, 0.4], [-1, 2, -0.0], [float("nan"), 0, 0],
        [float("inf"), float("-inf"), 0.5], [0.2, 0.3, 0.4],
    ], device=device)
    alphas[:, 0, 0, :8, 0] = torch.tensor(
        [1, 100 / 255, 0.5, 0, 2, 1, 0.75, float("nan")], device=device
    )
    overlay = torch.rand((7, 34, 3), generator=generator).to(device)[1:6, ::2] * 255
    # Preserve the cropped overlay strides after scaling.
    overlay_storage = torch.zeros((7, 34, 3), device=device)
    overlay_storage[1:6, ::2] = overlay
    return colors, alphas, overlay_storage[1:6, ::2]


class StridedRenderCompositingTests(unittest.TestCase):
    def setUp(self):
        self.trainer = object.__new__(InvPhyTrainerWarp)

    def test_renderer_keeps_original_rgb_and_alpha_storage(self):
        colors, alphas, _ = inputs(3, "cpu")
        for shared_template in (False, True):
            for separate_alpha in (False, True):
                with self.subTest(shared_template=shared_template, separate_alpha=separate_alpha):
                    pc = SimpleNamespace(
                        uses_shared_template_rendering=shared_template,
                        uses_batch_image_rendering=True,
                        uses_separate_render_alpha=separate_alpha,
                        get_xyz=torch.zeros((3, 3) if shared_template else (3, 1, 3)),
                        get_rotation=torch.zeros((3, 4)), get_scaling=torch.ones((3, 3)),
                        get_template_scaling=torch.ones((1, 3)), get_opacity=torch.ones((3, 1)),
                        get_template_opacity=torch.ones((1, 1)), gaussians_per_instance=1,
                    )
                    camera = SimpleNamespace(
                        K=torch.eye(3), world_view_transform=torch.eye(4),
                        image_height=5, image_width=17,
                    )
                    with patch.object(renderer, "rasterization", return_value=(colors, alphas, {})), \
                         patch.object(renderer, "rasterization_shared_template", return_value=(colors, alphas, {})):
                        result = renderer.render_gsplat(
                            camera, pc, None, torch.zeros(3), override_color=torch.zeros((1, 3))
                        )
                    torch.testing.assert_close(result["render"][:, :3].permute(0, 2, 3, 1), colors[:, 0], equal_nan=True)
                    self.assertIsNone(result["depth"])
                    if separate_alpha:
                        self.assertEqual(result["render"].shape, (3, 3, 5, 17))
                        self.assertEqual(result["render"].data_ptr(), colors.data_ptr())
                        self.assertEqual(result["alpha"].data_ptr(), alphas.data_ptr())
                        self.assertFalse(result["render"].is_contiguous())
                        torch.testing.assert_close(result["alpha"], alphas[:, 0, ..., 0], equal_nan=True)
                    else:
                        self.assertEqual(result["render"].shape, (3, 4, 5, 17))
                        self.assertTrue(result["render"].is_contiguous())
                        self.assertIsNone(result["alpha"])
                        torch.testing.assert_close(result["render"][:, 3], alphas[:, 0, ..., 0], equal_nan=True)

    def check_compositing(self, device):
        for batch in (0, 1, 3):
            with self.subTest(batch=batch):
                colors, alphas, overlay = inputs(batch, device)
                packed = renderer._format_gsplat_batch_images_output(colors, alphas, {}, 1)[0]
                rgb = renderer._format_gsplat_batch_images_output(
                    colors, alphas, {}, 1, separate_alpha=True
                )[0]
                alpha = alphas[:, 0, ..., 0]
                expected = reference(packed.cpu(), overlay.cpu())
                if device == "cuda":
                    import warp as wp
                    if batch:
                        for view in (rgb, alpha, overlay):
                            wrapped = wp.from_torch(view)
                            self.assertEqual(wrapped.ptr, view.data_ptr())
                            self.assertEqual(wrapped.strides, tuple(s * view.element_size() for s in view.stride()))
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        actual = self.trainer._composite_batch_images_without_shadows(rgb, overlay, batch_alpha=alpha)
                    stream.synchronize()
                else:
                    actual = self.trainer._composite_batch_images_without_shadows(rgb, overlay, batch_alpha=alpha)
                legacy = self.trainer._composite_batch_images_without_shadows(packed, overlay)
                for candidate in (actual, legacy):
                    torch.testing.assert_close(candidate[0].cpu(), expected[0], atol=0, rtol=0, equal_nan=True)
                    self.assertTrue(torch.equal(candidate[1].cpu(), expected[1]))
                if batch:
                    self.assertEqual(actual[1][0, 0, :8].tolist(), [False, False, True, False, True, True, True, False])

    def test_cpu_compositing(self):
        self.check_compositing("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
    def test_cuda_compositing_and_caller_stream(self):
        self.check_compositing("cuda")

    def test_separate_alpha_shape_validation(self):
        for alpha in (torch.ones(1, 2), torch.ones(1, 1, 2, 2)):
            with self.assertRaisesRegex(ValueError, "separate render alpha"):
                self.trainer._composite_batch_images_without_shadows(
                    torch.zeros(1, 3, 2, 2), torch.zeros(2, 2, 3), batch_alpha=alpha
                )

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
    def test_alpha_dtype_falls_back_to_torch(self):
        colors, alphas, overlay = inputs(1, "cuda")
        rgb = colors[:, 0].permute(0, 3, 1, 2)
        alpha = alphas[:, 0, ..., 0].double()
        actual = self.trainer._composite_batch_images_without_shadows(rgb, overlay, batch_alpha=alpha)
        expected = self.trainer._composite_batch_images_without_shadows(rgb.cpu(), overlay.cpu(), batch_alpha=alpha.cpu())
        self.assertEqual(actual[0].dtype, torch.float64)
        torch.testing.assert_close(actual[0].cpu(), expected[0], atol=0, rtol=0, equal_nan=True)
        self.assertTrue(torch.equal(actual[1].cpu(), expected[1]))


if __name__ == "__main__":
    unittest.main()
