"""Single-pass FP32 compositing over Boba's shared background image."""

import torch
import warp as wp


wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})
wp.init()
ALPHA_THRESHOLD = wp.constant(100.0 / 255.0)


@wp.func
def _clamp_color(value: float):
    # Match torch.clamp, including NaNs and conversion of -0.0 to +0.0.
    result = value
    if value <= 0.0:
        result = 0.0
    elif value > 1.0:
        result = 1.0
    return result


@wp.kernel
def _composite_pixels(
    rgba: wp.array4d(dtype=float),
    overlay: wp.array3d(dtype=float),
    frames: wp.array4d(dtype=float),
    image_mask: wp.array3d(dtype=wp.bool),
):
    instance, y, x = wp.tid()
    red = _clamp_color(rgba[instance, 0, y, x])
    green = _clamp_color(rgba[instance, 1, y, x])
    blue = _clamp_color(rgba[instance, 2, y, x])
    opacity = _clamp_color(rgba[instance, 3, y, x])
    visible = (red != 1.0 or green != 1.0 or blue != 1.0) and opacity > ALPHA_THRESHOLD
    alpha = opacity if visible else 0.0
    transmittance = 1.0 - alpha
    # Keep the reference operation order and rounding, including color * 0
    # for masked pixels, so nonfinite inputs retain their original behavior.
    frames[instance, y, x, 0] = overlay[y, x, 0] * transmittance + (red * alpha) * 255.0
    frames[instance, y, x, 1] = overlay[y, x, 1] * transmittance + (green * alpha) * 255.0
    frames[instance, y, x, 2] = overlay[y, x, 2] * transmittance + (blue * alpha) * 255.0
    image_mask[instance, y, x] = visible


def composite(batch_rendering, overlay):
    """Return the original RGB values and mask on the caller's CUDA stream.

    Inputs retain their strides. The background is one shared HxWx3 image,
    while frames and masks are per instance. RGB values use the [0,255] scale;
    final display clamping remains the caller's responsibility.
    """
    batch, _, height, width = batch_rendering.shape
    frames = torch.empty((batch, height, width, 3), device=batch_rendering.device,
                         dtype=torch.float32)
    mask = torch.empty((batch, height, width), device=batch_rendering.device,
                       dtype=torch.bool)
    if frames.numel() == 0:
        return frames, mask
    wp.launch(
        _composite_pixels,
        dim=(batch, height, width),
        inputs=[wp.from_torch(batch_rendering.detach()), wp.from_torch(overlay)],
        outputs=[wp.from_torch(frames), wp.from_torch(mask)],
        stream=wp.stream_from_torch(torch.cuda.current_stream(batch_rendering.device)),
    )
    return frames, mask
