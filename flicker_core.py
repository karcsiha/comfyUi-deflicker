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
# Adaptive trend detection
# ---------------------------------------------------------------------------

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

    Returns:
        Corrected [N, H, W] (unclamped — caller handles clamping).
    """
    num_frames = ch_data.shape[0]
    flat = ch_data.reshape(num_frames, -1)

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
) -> torch.Tensor:
    """Correct a single channel using per-cell gain on a spatial grid.

    Computes per-cell means using avg_pool2d, then applies multiplicative
    gain correction per cell with bilinear-interpolated seams.

    Args:
        ch_data: [N, H, W] single channel.
        window_size, strength, smooth_fn, has_trend: same as _correct_channel.
        grid_size: number of cells per axis.

    Returns:
        Corrected [N, H, W].
    """
    N, H, W = ch_data.shape
    device = ch_data.device

    # Compute per-cell means using adaptive avg pool
    # [N, H, W] -> [N, 1, H, W] -> pool to [N, 1, grid, grid]
    data_4d = ch_data.unsqueeze(1)
    cell_means = F.adaptive_avg_pool2d(data_4d, grid_size).squeeze(1)  # [N, G, G]

    # For each cell: compute gain = target_mean / current_mean
    gains = torch.ones(N, grid_size, grid_size, device=device)

    for gy in range(grid_size):
        for gx in range(grid_size):
            cm = cell_means[:, gy, gx]  # [N]
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


def deflicker_frames(
    images: torch.Tensor,
    window_size: int = 15,
    strength: float = 1.0,
    channels: str = "L",
    use_median: bool = False,
    pixel_smoothing: float = 0.0,
    grid_size: int = 1,
    drift_mode: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove temporal brightness/color flicker from a frame sequence.

    Two-phase correction:
    1. Per-frame statistics correction: equalizes per-frame mean brightness
       using adaptive trend detection + iterative smoothing.
    2. Per-pixel temporal smoothing (optional): averages each pixel across
       neighboring frames to remove spatially-varying flicker.

    Args:
        images: [B, H, W, 3] sRGB tensor in [0, 1].
        window_size: Temporal smoothing window in frames. Controls the
            separation between "flicker" (removed) and "trend" (preserved).
        strength: 0.0 = no correction, 1.0 = full correction.
        channels: "L" = brightness only (uniform delta across RGB),
                  "LAB" = per-channel correction (fixes color flicker too).
        use_median: Use median pre-filter (robust to extreme outlier frames).
        pixel_smoothing: Per-pixel temporal smoothing strength (0=off, 1=full).
        grid_size: Spatial grid for correction. 1 = global, >1 = per-cell.
        drift_mode: "auto" = detect trend automatically,
                    "flicker_only" = remove all changes including slow drift,
                    "preserve_trend" = always keep slow brightness changes.

    Returns:
        (corrected_images, debug_heatmap) — both [B, H, W, 3].
    """
    num_frames, H, W, C = images.shape
    device = images.device

    if num_frames < 2 or strength <= 0:
        return images.clone(), torch.zeros(num_frames, H, W, 3, device=device)

    smooth_fn = temporal_median_smooth if use_median else temporal_smooth

    # Determine trend handling based on drift_mode
    if drift_mode == "flicker_only":
        has_trend = False
    elif drift_mode == "preserve_trend":
        has_trend = True
    else:
        # Auto: detect trend from brightness
        brightness_means = images.mean(dim=(2, 3)).mean(dim=1)
        has_trend = _detect_trend(brightness_means)

    # --- Phase 1: Per-frame statistics correction ---
    correct_fn = _correct_channel if grid_size <= 1 else (
        lambda ch, ws, st, sf, ht: _correct_channel_grid(ch, ws, st, sf, ht, grid_size)
    )

    if channels == "L":
        brightness = images.mean(dim=-1)
        corrected_brightness = correct_fn(
            brightness, window_size, strength, smooth_fn, has_trend,
        )
        # Apply as multiplicative gain map to preserve black levels
        gain_map = (corrected_brightness / brightness.clamp(min=1e-4)).unsqueeze(-1)
        gain_map = gain_map.clamp(0.25, 4.0)
        corrected = (images * gain_map).clamp(0.0, 1.0)
    else:
        corrected = images.clone()
        for ch in range(3):
            corrected[..., ch] = correct_fn(
                images[..., ch], window_size, strength, smooth_fn, has_trend,
            )
        corrected = corrected.clamp(0.0, 1.0)

    # --- Phase 2: Per-pixel temporal smoothing (optional) ---
    if pixel_smoothing > 0:
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
    pos_mask = normalized > 0
    heatmap[..., 0][pos_mask] = normalized[pos_mask]
    neg_mask = normalized < 0
    heatmap[..., 2][neg_mask] = -normalized[neg_mask]
    return heatmap
