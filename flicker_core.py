"""Temporal flicker removal for video frame sequences.

Uses the industry-standard timelapse deflicker approach:
1. Compute per-frame luminance/statistics.
2. Temporally smooth the statistics curve.
3. Apply gain/offset correction to match each frame to the smooth target.
4. Multi-pass iteration for convergence (like LRTimelapse visual deflicker).

Enhanced with adaptive trend detection to balance flicker removal
vs. preserving intentional brightness changes.
"""
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Border masking — exclude black bars from statistics
# ---------------------------------------------------------------------------

def _compute_content_mask(images: torch.Tensor, threshold: float = 0.02) -> torch.Tensor:
    """Detect content vs. black border regions for stabilized/cropped footage.

    Computes a spatial mask [H, W] where True = content pixel, False = border.
    A pixel is considered border if its mean brightness across ALL frames is
    below the threshold. This catches letterbox, pillarbox, and irregular
    stabilization crops.

    The mask is only used for statistics computation (frame means, CDFs).
    Corrections are always applied to the full frame.

    Args:
        images: [B, H, W, C] sRGB tensor in [0, 1].
        threshold: Brightness below this (across all frames) = border. Default
            0.02 catches near-black borders without masking dark content.

    Returns:
        [H, W] boolean mask. True = content pixel.
    """
    # Mean brightness per pixel across all frames and channels
    temporal_mean = images.mean(dim=(0, -1))  # [H, W]
    mask = temporal_mean >= threshold

    # Safety: if mask excludes >95% of pixels, it's probably a very dark
    # scene, not actual borders — fall back to using everything.
    if mask.sum() < mask.numel() * 0.05:
        return torch.ones_like(mask, dtype=torch.bool)

    return mask


# ---------------------------------------------------------------------------
# Temporal smoothing utilities
# ---------------------------------------------------------------------------

def _temporal_gaussian_kernel(window_size: int) -> torch.Tensor:
    """Create a 1D Gaussian kernel for temporal smoothing."""
    sigma = window_size / 4.0
    half = window_size // 2
    x = torch.arange(-half, half + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / max(sigma, 0.5)) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


def temporal_smooth(values: torch.Tensor, window_size: int) -> torch.Tensor:
    """Gaussian temporal smoothing of a 1D signal [num_frames].

    Uses reflect padding to avoid edge artifacts.
    """
    if window_size <= 1 or len(values) <= 1:
        return values.clone()

    window_size = min(window_size, len(values))
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 3:
        return values.clone()

    kernel = _temporal_gaussian_kernel(window_size).to(values.device)
    half = len(kernel) // 2

    padded = F.pad(
        values.unsqueeze(0).unsqueeze(0),
        (half, half),
        mode="reflect",
    )
    smoothed = F.conv1d(padded, kernel.unsqueeze(0).unsqueeze(0))
    return smoothed.squeeze(0).squeeze(0)


def temporal_median_smooth(values: torch.Tensor, window_size: int) -> torch.Tensor:
    """Median temporal smoothing — robust to outliers."""
    if window_size <= 1 or len(values) <= 1:
        return values.clone()

    n = len(values)
    half = window_size // 2
    result = torch.empty_like(values)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result[i] = values[lo:hi].median()

    return temporal_smooth(result, max(3, window_size // 3))


# ---------------------------------------------------------------------------
# Step removal — instant correction of latent space shifts
# ---------------------------------------------------------------------------

def _remove_steps(
    ch_data: torch.Tensor,
    content_mask: torch.Tensor | None = None,
    threshold_mult: float = 5.0,
    strength: float = 1.0,
) -> torch.Tensor:
    """Remove step discontinuities (latent space shifts) from a channel.

    Detects sharp frame-to-frame brightness jumps that stand out from the
    normal motion noise, then applies cumulative gain correction to undo
    them instantly. Unlike temporal smoothing, this preserves the natural
    brightness trend — it only removes the discrete steps.

    Algorithm:
    1. Compute per-frame means (masked).
    2. Compute frame-to-frame diffs.
    3. Detect outlier diffs (> threshold_mult × median abs diff).
    4. Accumulate detected steps as a running correction.
    5. Apply as per-frame gain: target / current.

    Args:
        ch_data: [N, H, W] single channel data.
        content_mask: [H, W] bool mask. True = content pixel for stats.
        threshold_mult: Sensitivity — how many times the median abs diff
            counts as a step. Lower = more sensitive. Default 5.0.
        strength: 0.0 = no correction, 1.0 = full step removal.

    Returns:
        Corrected [N, H, W] tensor.
    """
    N = ch_data.shape[0]
    if N < 3:
        return ch_data.clone()

    flat = ch_data.reshape(N, -1)

    # Compute per-frame means (using content mask if provided)
    if content_mask is not None:
        mask_flat = content_mask.reshape(-1)
        if mask_flat.any():
            frame_means = flat[:, mask_flat].mean(dim=1)
        else:
            frame_means = flat.mean(dim=1)
    else:
        frame_means = flat.mean(dim=1)

    # Frame-to-frame diffs
    diffs = frame_means[1:] - frame_means[:-1]

    # Threshold: outlier diffs relative to the typical noise.
    # Use a brightness-relative floor so detection works even when frames
    # are nearly identical (e.g., static scene with only step changes).
    median_abs_diff = diffs.abs().median()
    floor = frame_means.mean().item() * 0.005  # 0.5% of mean brightness
    threshold = max(median_abs_diff.item() * threshold_mult, floor)

    if threshold < 1e-6:
        return ch_data.clone()

    # Accumulate step corrections (vectorized)
    step_corrections = torch.where(diffs.abs() > threshold, -diffs, torch.zeros_like(diffs))
    cumulative = torch.zeros(N, device=ch_data.device)
    cumulative[1:] = step_corrections.cumsum(dim=0)

    # No steps detected
    if cumulative.abs().max() < 1e-6:
        return ch_data.clone()

    # Blend with strength
    cumulative = cumulative * strength

    # Apply as gain correction (preserves black levels)
    target_means = frame_means + cumulative
    safe_means = frame_means.clamp(min=1e-2)
    gains = (target_means / safe_means).clamp(0.25, 4.0).view(-1, 1, 1)

    return ch_data * gains


# ---------------------------------------------------------------------------
# Adaptive trend detection
# ---------------------------------------------------------------------------

def _masked_frame_means(
    images: torch.Tensor, content_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Compute per-frame mean brightness using only content pixels.

    Args:
        images: [B, H, W, C] or [B, H, W].
        content_mask: [H, W] bool mask, or None for all pixels.

    Returns:
        [B] tensor of per-frame means.
    """
    if content_mask is None:
        if images.dim() == 4:
            return images.mean(dim=(1, 2, 3))
        return images.mean(dim=(1, 2))

    if images.dim() == 4:
        # [B, H, W, C] -> mask [H, W] -> expand to [H, W, C]
        mask_expanded = content_mask.unsqueeze(-1).expand_as(images[0])
        flat = images[:, mask_expanded].reshape(images.shape[0], -1)
    else:
        # [B, H, W]
        flat = images[:, content_mask]
    return flat.mean(dim=1)


def _detect_trend(frame_means: torch.Tensor) -> bool:
    """Detect whether there's a significant linear trend in the frame means.

    Returns True if the trend magnitude significantly exceeds the noise level,
    indicating intentional brightness changes that should be preserved.
    """
    N = len(frame_means)
    if N < 5:
        return False

    t = torch.arange(N, dtype=torch.float32, device=frame_means.device)
    t_centered = t - t.mean()
    m_centered = frame_means - frame_means.mean()

    # Linear regression slope
    slope = (m_centered * t_centered).sum() / (t_centered ** 2).sum()

    # Trend magnitude: total brightness change across the sequence
    trend_magnitude = abs(slope.item()) * N

    # Compare against flicker magnitude (std of means)
    flicker_magnitude = frame_means.std().item()

    # Trend is "significant" if total change > 2x the noise level
    return trend_magnitude > flicker_magnitude * 2.0


# ---------------------------------------------------------------------------
# Core correction: global_mean + wide_trend + iteration
# ---------------------------------------------------------------------------

def _correct_channel(
    ch_data: torch.Tensor,
    window_size: int,
    strength: float,
    smooth_fn,
    has_trend: bool = False,
    content_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Correct a single channel's temporal flicker using gain-based correction.

    Uses multiplicative (gain) correction instead of affine (mean+std) to
    preserve black levels: pixel * gain keeps 0 at 0, like an exposure dial.

    Algorithm:
    - No trend: flatten means to global constant via gain.
    - Trend detected: fit polynomial or wide-window smooth, apply via gain.

    Args:
        ch_data: [N, H, W] single channel.
        window_size: user's temporal smoothing window.
        strength: correction strength (0-1).
        smooth_fn: temporal smoothing function.
        has_trend: if True, preserve slow brightness changes.
        content_mask: [H, W] bool mask. True = content pixel for stats.

    Returns:
        Corrected [N, H, W] (unclamped — caller handles clamping).
    """
    num_frames = ch_data.shape[0]
    flat = ch_data.reshape(num_frames, -1)

    # Use only content pixels for statistics, but correct all pixels
    if content_mask is not None:
        mask_flat = content_mask.reshape(-1)
        frame_means = flat[:, mask_flat].mean(dim=1)  # [N]
    else:
        frame_means = flat.mean(dim=1)  # [N]
    global_mean = frame_means.mean()

    if has_trend:
        # Trend detected: fit a low-order polynomial to capture the smooth
        # underlying brightness curve — like a compositor drawing a spline
        # in Nuke. The polynomial gives a perfectly smooth target that
        # removes ALL per-frame noise while preserving the overall shape.
        #
        # If the polynomial fits poorly (e.g. step function at a chunk
        # boundary), fall back to iterative wide-window smoothing.
        degree = max(2, min(5, num_frames // 20))
        t = torch.arange(num_frames, dtype=torch.float32, device=ch_data.device)
        t_norm = t / max(num_frames - 1, 1)
        V = torch.stack([t_norm ** d for d in range(degree + 1)], dim=1)
        coeffs = torch.linalg.solve(V.T @ V, V.T @ frame_means)
        poly_target = V @ coeffs

        # Check fit quality: if max residual > 3x the typical noise,
        # the polynomial doesn't capture the signal well (step function).
        poly_residual = (frame_means - poly_target).abs()
        noise_estimate = frame_means.std() * 0.5
        if poly_residual.max() > noise_estimate * 3:
            # Poor polynomial fit — use iterative wide-window smoothing.
            wide_w = min(window_size * 2, num_frames)
            if wide_w % 2 == 0:
                wide_w = max(3, wide_w - 1)
            corrected_means = frame_means.clone()
            for _ in range(3):
                trend = smooth_fn(corrected_means, wide_w)
                tc = trend - trend.mean()
                corrected_means = global_mean + tc
            target_means = corrected_means
        else:
            target_means = poly_target
    else:
        # No significant trend: flatten to constant → near-100% removal.
        target_means = torch.full_like(frame_means, global_mean)

    # Blend with strength
    final_means = frame_means + (target_means - frame_means) * strength

    # --- Apply gain-based correction per frame ---
    # gain = target / current — keeps 0 at 0 (like exposure compensation)
    safe_means = frame_means.clamp(min=1e-2)
    gains = (final_means / safe_means).clamp(0.25, 4.0).view(-1, 1, 1)

    corrected = ch_data * gains

    return corrected


def _correct_channel_grid(
    ch_data: torch.Tensor,
    window_size: int,
    strength: float,
    smooth_fn,
    has_trend: bool,
    grid_size: int,
    content_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Correct a single channel using per-cell gain on a spatial grid.

    Computes per-cell means, then applies multiplicative gain correction
    per cell with bilinear-interpolated seams.

    Args:
        ch_data: [N, H, W] single channel.
        window_size, strength, smooth_fn, has_trend: same as _correct_channel.
        grid_size: number of cells per axis.
        content_mask: [H, W] bool mask. True = content pixel for stats.

    Returns:
        Corrected [N, H, W].
    """
    N, H, W = ch_data.shape
    device = ch_data.device

    # Fast path: no mask → use adaptive_avg_pool2d (fused GPU kernel)
    if content_mask is None:
        data_4d = ch_data.unsqueeze(1)
        cell_means_pooled = F.adaptive_avg_pool2d(data_4d, grid_size).squeeze(1)  # [N, G, G]

    cell_h = H / grid_size
    cell_w = W / grid_size

    # For each cell: compute gain = target_mean / current_mean
    gains = torch.ones(N, grid_size, grid_size, device=device)

    for gy in range(grid_size):
        for gx in range(grid_size):
            if content_mask is None:
                # Fast path: use pooled means
                cm = cell_means_pooled[:, gy, gx]  # [N]
            else:
                y0 = round(gy * cell_h)
                y1 = round((gy + 1) * cell_h)
                x0 = round(gx * cell_w)
                x1 = round((gx + 1) * cell_w)

                cell_data = ch_data[:, y0:y1, x0:x1]
                cell_flat = cell_data.reshape(N, -1)
                cell_mask = content_mask[y0:y1, x0:x1].reshape(-1)
                if cell_mask.any():
                    cm = cell_flat[:, cell_mask].mean(dim=1)
                else:
                    # Entire cell is border — skip correction
                    continue

            global_mean = cm.mean()

            if has_trend:
                target_means = smooth_fn(cm, window_size)
            else:
                target_means = torch.full_like(cm, global_mean)

            final_means = cm + (target_means - cm) * strength
            safe_means = cm.clamp(min=1e-2)

            gains[:, gy, gx] = (final_means / safe_means).clamp(0.25, 4.0)

    # Upsample gain map from [N, G, G] to [N, H, W] with bilinear interpolation
    gains_up = F.interpolate(
        gains.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False,
    ).squeeze(1)  # [N, H, W]

    return ch_data * gains_up


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _pixel_temporal_smooth(
    images: torch.Tensor,
    window_size: int,
    blend_strength: float,
) -> torch.Tensor:
    """Per-pixel temporal smoothing (inspired by SuperBeasts PixelDeflicker).

    For each pixel, averages its value across neighboring frames within a
    sliding window. This removes spatially-varying flicker that frame-mean
    correction cannot address. The result is blended with the input using
    blend_strength to preserve detail.

    Uses Gaussian-weighted averaging (not box filter) to preserve temporal
    sharpness while removing high-frequency pixel-level noise.

    Args:
        images: [B, H, W, C] tensor.
        window_size: temporal window for averaging.
        blend_strength: 0=no pixel smoothing, 1=full replacement.

    Returns:
        Blended [B, H, W, C] tensor.
    """
    if blend_strength <= 0 or window_size <= 1:
        return images

    num_frames = images.shape[0]
    window_size = min(window_size, num_frames)
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 3:
        return images

    # Gaussian weights for the temporal window
    sigma = window_size / 4.0
    half = window_size // 2
    t = torch.arange(-half, half + 1, dtype=torch.float32, device=images.device)
    weights = torch.exp(-0.5 * (t / max(sigma, 0.5)) ** 2)
    weights = weights / weights.sum()  # [K]

    # Weighted temporal average per pixel using conv1d
    # Reshape: [B, H, W, C] -> [H*W*C, 1, B] for conv1d over time axis
    B, H, W, C = images.shape
    flat = images.permute(1, 2, 3, 0).reshape(-1, 1, B)  # [H*W*C, 1, B]

    kernel = weights.flip(0).view(1, 1, -1)  # [1, 1, K]
    padded = F.pad(flat, (half, half), mode="reflect")
    smoothed = F.conv1d(padded, kernel)  # [H*W*C, 1, B]

    smoothed = smoothed.reshape(H, W, C, B).permute(3, 0, 1, 2)  # [B, H, W, C]

    # Blend: original * (1-strength) + smoothed * strength
    result = images * (1.0 - blend_strength) + smoothed * blend_strength
    return result


@torch.no_grad()
def deflicker_frames(
    images: torch.Tensor,
    window_size: int = 15,
    strength: float = 1.0,
    channels: str = "L",
    use_median: bool = False,
    pixel_smoothing: float = 0.0,
    grid_size: int = 1,
    drift_mode: str = "auto",
    content_mask: torch.Tensor | None = None,
    mode: str = "temporal_smoothing",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove temporal brightness/color flicker from a frame sequence.

    Supports two correction modes:
    - temporal_smoothing: Gaussian/median smoothing of per-frame statistics.
      Best for random per-frame flicker (AI video noise).
    - step_removal: Instant correction of sharp brightness steps caused by
      latent space shifts. Preserves natural trends, removes only discontinuities.
    - both: Step removal first, then temporal smoothing.

    Args:
        images: [B, H, W, 3] sRGB tensor in [0, 1].
        window_size: Temporal smoothing window in frames. Controls the
            separation between "flicker" (removed) and "trend" (preserved).
            Only used in temporal_smoothing/both modes.
        strength: 0.0 = no correction, 1.0 = full correction.
        channels: "L" = brightness only (uniform delta across RGB),
                  "LAB" = per-channel correction (fixes color flicker too).
        use_median: Use median pre-filter (robust to extreme outlier frames).
        pixel_smoothing: Per-pixel temporal smoothing strength (0=off, 1=full).
        grid_size: Spatial grid for correction. 1 = global, >1 = per-cell.
        drift_mode: "auto" = detect trend automatically,
                    "flicker_only" = remove all changes including slow drift,
                    "preserve_trend" = always keep slow brightness changes.
        content_mask: [H, W] bool mask from _compute_content_mask(). If None,
            computed automatically.
        mode: "temporal_smoothing" = classic window-based correction,
              "step_removal" = instant step discontinuity correction,
              "both" = step removal then temporal smoothing.

    Returns:
        (corrected_images, debug_heatmap) — both [B, H, W, 3].
    """
    num_frames, H, W, C = images.shape
    device = images.device

    if num_frames < 2 or strength <= 0:
        return images, torch.zeros(num_frames, H, W, 3, device=device)

    smooth_fn = temporal_median_smooth if use_median else temporal_smooth

    # Auto-detect black borders (stabilized/cropped footage)
    if content_mask is None:
        content_mask = _compute_content_mask(images)

    do_steps = mode in ("step_removal", "both")
    do_temporal = mode in ("temporal_smoothing", "both")

    corrected = images

    # --- Phase 0: Step removal (latent space shift correction) ---
    if do_steps:
        if channels == "L":
            brightness = corrected.mean(dim=-1)  # [N, H, W]
            corrected_brightness = _remove_steps(
                brightness, content_mask, strength=strength,
            )
            gain_map = (corrected_brightness / brightness.clamp(min=1e-4)).unsqueeze(-1)
            gain_map = gain_map.clamp(0.25, 4.0)
            corrected = (corrected * gain_map).clamp(0.0, 1.0)
        else:
            corrected = corrected.clone()
            for ch in range(3):
                corrected[..., ch] = _remove_steps(
                    corrected[..., ch], content_mask, strength=strength,
                ).clamp(0.0, 1.0)

    # --- Phase 1: Per-frame statistics correction (temporal smoothing) ---
    if do_temporal:
        # Determine trend handling based on drift_mode
        if drift_mode == "flicker_only":
            has_trend = False
        elif drift_mode == "preserve_trend":
            has_trend = True
        else:
            brightness_means = _masked_frame_means(corrected, content_mask)
            has_trend = _detect_trend(brightness_means)

        correct_fn = (
            lambda ch, ws, st, sf, ht: _correct_channel(ch, ws, st, sf, ht, content_mask)
        ) if grid_size <= 1 else (
            lambda ch, ws, st, sf, ht: _correct_channel_grid(
                ch, ws, st, sf, ht, grid_size, content_mask,
            )
        )

        if channels == "L":
            brightness = corrected.mean(dim=-1)
            corrected_brightness = correct_fn(
                brightness, window_size, strength, smooth_fn, has_trend,
            )
            gain_map = (corrected_brightness / brightness.clamp(min=1e-4)).unsqueeze(-1)
            gain_map = gain_map.clamp(0.25, 4.0)
            corrected = (corrected * gain_map).clamp(0.0, 1.0)
        else:
            tmp = corrected.clone()
            for ch in range(3):
                tmp[..., ch] = correct_fn(
                    corrected[..., ch], window_size, strength, smooth_fn, has_trend,
                )
            corrected = tmp.clamp(0.0, 1.0)

    # --- Phase 2: Per-pixel temporal smoothing (optional) ---
    if pixel_smoothing > 0 and do_temporal:
        corrected = _pixel_temporal_smooth(
            corrected, window_size, pixel_smoothing * strength,
        )
        corrected = corrected.clamp(0.0, 1.0)

    heatmap = _generate_correction_heatmap(corrected, images)
    return corrected, heatmap


def _generate_correction_heatmap(
    corrected: torch.Tensor, original: torch.Tensor,
) -> torch.Tensor:
    """Blue-black-red heatmap from brightness difference."""
    diff = corrected.mean(dim=-1) - original.mean(dim=-1)
    B, H, W = diff.shape
    device = diff.device
    max_abs = diff.abs().max()
    if max_abs < 1e-6:
        return torch.zeros(B, H, W, 3, device=device)

    normalized = diff / max_abs
    heatmap = torch.zeros(B, H, W, 3, device=device)
    heatmap[..., 0] = normalized.clamp(min=0)
    heatmap[..., 2] = (-normalized).clamp(min=0)
    return heatmap
