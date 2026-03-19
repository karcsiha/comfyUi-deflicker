import torch
import torch.nn.functional as F
from typing import List


# ---------------------------------------------------------------------------
# sRGB <-> LAB color space conversion (pure torch, GPU-compatible)
# ---------------------------------------------------------------------------

def _srgb_to_linear(srgb: torch.Tensor) -> torch.Tensor:
    """Inverse sRGB companding: sRGB [0,1] -> linear RGB [0,1]."""
    return torch.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )


def _linear_to_srgb(linear: torch.Tensor) -> torch.Tensor:
    """sRGB companding: linear RGB [0,1] -> sRGB [0,1]."""
    return torch.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * linear.clamp(min=1e-10) ** (1.0 / 2.4) - 0.055,
    )


# sRGB D65 matrix: linear RGB -> XYZ
_RGB_TO_XYZ = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])

# Inverse: XYZ -> linear RGB
_XYZ_TO_RGB = torch.tensor([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
])

# D65 whitepoint
_D65 = torch.tensor([0.95047, 1.0, 1.08883])

# CIE constants
_EPSILON = (6.0 / 29.0) ** 3  # 0.008856
_KAPPA = (29.0 / 6.0) ** 2 / 3  # 7.787037


def srgb_to_lab(images: torch.Tensor) -> torch.Tensor:
    """Convert sRGB [0,1] images to LAB. Input/output: [B,H,W,3]."""
    device = images.device
    rgb_to_xyz = _RGB_TO_XYZ.to(device)
    d65 = _D65.to(device)

    # sRGB -> linear RGB
    linear = _srgb_to_linear(images)

    # linear RGB -> XYZ via matrix multiply
    xyz = torch.einsum("...c,cd->...d", linear, rgb_to_xyz.T)

    # Normalize by D65 whitepoint
    xyz_norm = xyz / d65

    # XYZ -> LAB f(t) function
    f = torch.where(
        xyz_norm > _EPSILON,
        xyz_norm.clamp(min=1e-10) ** (1.0 / 3.0),
        _KAPPA * xyz_norm + 16.0 / 116.0,
    )

    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])

    return torch.stack([L, a, b], dim=-1)


def lab_to_srgb(lab: torch.Tensor) -> torch.Tensor:
    """Convert LAB to sRGB [0,1]. Input/output: [B,H,W,3]. Output clamped to [0,1]."""
    device = lab.device
    xyz_to_rgb = _XYZ_TO_RGB.to(device)
    d65 = _D65.to(device)

    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]

    # LAB -> f values
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    # Inverse f(t): f -> t
    def inv_f(f_val):
        return torch.where(
            f_val > 6.0 / 29.0,
            f_val ** 3,
            (f_val - 16.0 / 116.0) / _KAPPA,
        )

    xyz = torch.stack([inv_f(fx), inv_f(fy), inv_f(fz)], dim=-1)

    # Denormalize by D65
    xyz = xyz * d65

    # XYZ -> linear RGB
    linear = torch.einsum("...c,cd->...d", xyz, xyz_to_rgb.T)

    # linear RGB -> sRGB
    srgb = _linear_to_srgb(linear)

    return srgb.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Boundary detection and brightness correction
# ---------------------------------------------------------------------------



def _compute_auto_threshold(frame_diffs: torch.Tensor, sensitivity: float) -> float:
    """Compute automatic detection threshold from frame-to-frame L differences.

    Uses a robust outlier detection approach: any diff that is sensitivity times
    larger than the typical (non-outlier) variation is considered a boundary.
    We use the 75th percentile as the baseline "normal" variation.
    """
    if len(frame_diffs) < 2:
        return 0.5  # fallback for very short sequences

    sorted_diffs = frame_diffs.sort().values
    p75_idx = int(len(sorted_diffs) * 0.75)
    p75 = sorted_diffs[p75_idx].item()

    # Threshold: sensitivity times the 75th percentile, with a minimum floor
    threshold = max(p75 * sensitivity, 0.3)
    return threshold


def detect_boundaries_auto(L: torch.Tensor, detection_threshold: float = 0.0,
                           sensitivity: float = 3.0) -> List[int]:
    """Auto-detect boundaries by finding frames with large luminance jumps.

    Args:
        L: Luminance tensor [B, H, W] (range [0, 100]).
        detection_threshold: If > 0, use this fixed threshold. If 0, compute
            automatically from the sequence statistics.
        sensitivity: For auto threshold: multiplier for std deviation above median.
            Lower = more sensitive. Default 3.0 (detects clear outlier jumps).

    Returns list of frame indices where a new chunk starts (the frame AFTER the jump).
    """
    num_frames = L.shape[0]
    if num_frames < 2:
        return []

    # Compute per-frame mean luminance
    frame_means = L.reshape(num_frames, -1).mean(dim=1)  # [B]

    # Compute frame-to-frame absolute differences
    frame_diffs = (frame_means[1:] - frame_means[:-1]).abs()  # [B-1]

    # Determine threshold
    if detection_threshold > 0:
        thresh = detection_threshold
    else:
        thresh = _compute_auto_threshold(frame_diffs, sensitivity)

    # Find jumps exceeding threshold — vectorized
    jump_mask = frame_diffs >= thresh
    boundaries = (jump_mask.nonzero(as_tuple=False).squeeze(-1) + 1).tolist()

    return boundaries



# ---------------------------------------------------------------------------
# CDF-based histogram matching (industry-standard deflicker)
# ---------------------------------------------------------------------------

NUM_BINS = 2048  # histogram resolution for full-frame CDF
NUM_BINS_GRID = 512  # reduced resolution for grid cells


def _compute_cdf(frame: torch.Tensor, vmin: float = 0.0, vmax: float = 100.0,
                 num_bins: int = NUM_BINS) -> torch.Tensor:
    """Compute the cumulative distribution function of values.

    Args:
        frame: [H, W] tensor.
        vmin, vmax: value range for histogram bins.
        num_bins: histogram resolution.

    Returns:
        CDF tensor of shape [num_bins], values in [0, 1].
    """
    flat = frame.flatten().float()
    hist = torch.histc(flat, bins=num_bins, min=vmin, max=vmax)
    cdf = hist.cumsum(dim=0)
    cdf = cdf / cdf[-1].clamp(min=1e-10)
    return cdf


def _cdf_mean(cdf: torch.Tensor, vmin: float = 0.0, vmax: float = 100.0) -> float:
    """Compute the mean value from a CDF (expected value of the distribution)."""
    n = cdf.shape[0]
    pdf = torch.zeros_like(cdf)
    pdf[0] = cdf[0]
    pdf[1:] = cdf[1:] - cdf[:-1]
    bin_centers = torch.linspace(vmin, vmax, n, device=cdf.device)
    return (pdf * bin_centers).sum().item()


def _histogram_match(frame: torch.Tensor, source_cdf: torch.Tensor,
                     target_cdf: torch.Tensor,
                     vmin: float = 0.0, vmax: float = 100.0) -> torch.Tensor:
    """Transform frame so its distribution matches the target CDF.

    Automatically adapts to the CDF bin count (works with both NUM_BINS
    and NUM_BINS_GRID).

    Args:
        frame: [H, W] tensor.
        source_cdf, target_cdf: [N] CDFs (same length).
        vmin, vmax: value range matching the CDFs.

    Returns:
        [H, W] tensor with matched values.
    """
    device = frame.device
    shape = frame.shape
    flat = frame.flatten().float()
    vrange = vmax - vmin
    n = source_cdf.shape[0]

    # Map pixel values to bin indices
    normalized = ((flat - vmin) / vrange).clamp(0, 1)
    bin_indices = (normalized * (n - 1)).long().clamp(0, n - 1)
    unique_bins = bin_indices.unique().shape[0]

    if unique_bins < 5:
        # Fallback: simple mean-shift
        source_mean = _cdf_mean(source_cdf, vmin, vmax)
        target_mean = _cdf_mean(target_cdf, vmin, vmax)
        return frame + (target_mean - source_mean)

    # Look up each pixel's percentile in the source CDF
    percentiles = source_cdf[bin_indices]

    # Invert the target CDF
    matched_bins = torch.searchsorted(target_cdf, percentiles.contiguous())
    matched_bins = matched_bins.clamp(0, n - 1)

    # Convert bin indices back to values
    matched = matched_bins.float() / (n - 1) * vrange + vmin

    return matched.reshape(shape)


def _blend_cdfs(cdf_a: torch.Tensor, cdf_b: torch.Tensor,
                weight_b: float) -> torch.Tensor:
    """Linearly interpolate between two CDFs.

    weight_b=0.0 → pure cdf_a, weight_b=1.0 → pure cdf_b.
    """
    return cdf_a * (1.0 - weight_b) + cdf_b * weight_b



def _generate_heatmap(correction_map: torch.Tensor) -> torch.Tensor:
    """Generate a diverging blue-black-red heatmap from correction values.

    Input: [B, H, W] correction values.
    Output: [B, H, W, 3] RGB heatmap.
    """
    device = correction_map.device
    B, H, W = correction_map.shape

    max_abs = correction_map.abs().max()
    if max_abs < 1e-6:
        return torch.zeros(B, H, W, 3, device=device)

    # Normalize and boost contrast with sqrt (gamma=0.5) so subtle
    # corrections are more visible in the heatmap
    normalized = correction_map / max_abs  # range [-1, 1]
    boosted = normalized.abs().sqrt() * normalized.sign()

    heatmap = torch.zeros(B, H, W, 3, device=device)
    # Positive (brightening) -> red
    pos_mask = boosted > 0
    heatmap[..., 0][pos_mask] = boosted[pos_mask]  # R
    # Negative (darkening) -> blue
    neg_mask = boosted < 0
    heatmap[..., 2][neg_mask] = -boosted[neg_mask]  # B

    return heatmap


def _gaussian_blur_2d(tensor: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """Apply 2D Gaussian blur to a [H, W] tensor."""
    device = tensor.device
    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size, device=device, dtype=torch.float32) - kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Separable 2D: blur rows, then columns
    t = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    pad = kernel_size // 2

    # Horizontal
    k_h = kernel_1d.view(1, 1, 1, -1)
    t = F.pad(t, (pad, pad, 0, 0), mode="reflect")
    t = F.conv2d(t, k_h)

    # Vertical
    k_v = kernel_1d.view(1, 1, -1, 1)
    t = F.pad(t, (0, 0, pad, pad), mode="reflect")
    t = F.conv2d(t, k_v)

    return t.squeeze(0).squeeze(0)


def _apply_cdf_correction_grid(
    frame_lab: torch.Tensor,
    before_lab: torch.Tensor,
    after_lab: torch.Tensor,
    t: float,
    strength: float,
    grid_size: int,
    ch_ranges: list,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply CDF histogram matching per grid cell with bilinear interpolation.

    Splits the frame into grid_size x grid_size cells, does CDF matching
    per cell, then blends cell boundaries smoothly.

    Args:
        frame_lab: Current frame [H, W, 3] LAB.
        before_lab: Stable frame before boundary [H, W, 3] LAB.
        after_lab: Stable frame after boundary [H, W, 3] LAB.
        t: Interpolation weight (0=before, 1=after).
        strength: Correction strength.
        grid_size: Number of cells per axis.
        ch_ranges: List of (vmin, vmax) per channel.

    Returns:
        (corrected [H, W, 3], L_correction [H, W])
    """
    H, W = frame_lab.shape[:2]
    device = frame_lab.device
    corrected = frame_lab.clone()
    L_correction = torch.zeros(H, W, device=device)

    cell_h = H / grid_size
    cell_w = W / grid_size

    # Compute per-cell corrected values
    cell_results = {}  # (gy, gx) -> corrected [cell_H, cell_W, 3]

    for gy in range(grid_size):
        for gx in range(grid_size):
            y0 = round(gy * cell_h)
            y1 = round((gy + 1) * cell_h)
            x0 = round(gx * cell_w)
            x1 = round((gx + 1) * cell_w)

            cell_frame = frame_lab[y0:y1, x0:x1, :]
            cell_before = before_lab[y0:y1, x0:x1, :]
            cell_after = after_lab[y0:y1, x0:x1, :]

            cell_corrected = cell_frame.clone()
            nb = NUM_BINS_GRID  # fewer bins for grid cells = faster
            for ch in range(3):
                vmin, vmax = ch_ranges[ch]
                frame_cdf = _compute_cdf(cell_frame[..., ch], vmin, vmax, nb)
                before_cdf = _compute_cdf(cell_before[..., ch], vmin, vmax, nb)
                after_cdf = _compute_cdf(cell_after[..., ch], vmin, vmax, nb)
                target_cdf = _blend_cdfs(before_cdf, after_cdf, t)
                blended_target = _blend_cdfs(frame_cdf, target_cdf, strength)
                cell_corrected[..., ch] = _histogram_match(
                    cell_frame[..., ch], frame_cdf, blended_target, vmin, vmax,
                ).clamp(vmin, vmax)

            cell_results[(gy, gx)] = cell_corrected

    # Assemble: if grid_size == 1, just copy. Otherwise bilinear blend.
    if grid_size == 1:
        corrected = cell_results[(0, 0)]
        corrected = F.pad(corrected[:H, :W, :], (0, 0, 0, 0))  # ensure size
        L_correction = corrected[..., 0] - frame_lab[..., 0]
    else:
        # Simple assembly — place cells and blur the seams
        for gy in range(grid_size):
            for gx in range(grid_size):
                y0 = round(gy * cell_h)
                y1 = round((gy + 1) * cell_h)
                x0 = round(gx * cell_w)
                x1 = round((gx + 1) * cell_w)
                corrected[y0:y1, x0:x1, :] = cell_results[(gy, gx)]

        L_correction = corrected[..., 0] - frame_lab[..., 0]

        # Smooth seams: blur the correction map and reapply
        blur_size = max(3, round(min(cell_h, cell_w) * 0.4))
        if blur_size % 2 == 0:
            blur_size += 1
        for ch in range(3):
            ch_diff = corrected[..., ch] - frame_lab[..., ch]
            ch_diff_smooth = _gaussian_blur_2d(ch_diff, blur_size, blur_size / 4.0)
            vmin, vmax = ch_ranges[ch]
            corrected[..., ch] = (frame_lab[..., ch] + ch_diff_smooth).clamp(vmin, vmax)

        L_correction = corrected[..., 0] - frame_lab[..., 0]

    return corrected, L_correction


def auto_brightness_equalize(
    images: torch.Tensor,
    blend_radius: int = 5,
    strength: float = 1.0,
    detection_threshold: float = 0.0,
    sensitivity: float = 3.0,
    ref_frames: int = 0,
    grid_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Main entry point. Returns (corrected_images, debug_heatmap).

    Two-phase correction:
    1. Global drift correction: normalizes the entire shot to match
       the first ref_frames frames (reference zone).
    2. Local boundary correction: auto-detects brightness jumps and
       applies CDF histogram matching to smooth them.

    Args:
        blend_radius: Max frames to blend around each detected boundary.
        strength: Correction strength (0 = off, 1 = full).
        detection_threshold: Fixed L-channel threshold. 0 = automatic.
        sensitivity: For auto threshold: multiplier for 75th percentile.
            Lower = more sensitive.
        ref_frames: Number of initial frames to use as brightness reference.
            Set to 0 to skip drift correction.
        grid_size: Spatial grid for correction. 1 = global, >1 = per-cell.
    """
    num_frames, H, W, C = images.shape
    device = images.device

    # Convert all images to LAB
    lab = srgb_to_lab(images)
    L = lab[..., 0]  # [B, H, W]

    # Channel ranges: L=[0,100], a=[-128,128], b=[-128,128]
    ch_ranges = [(0.0, 100.0), (-128.0, 128.0), (-128.0, 128.0)]

    # ===================================================================
    # Phase 1: Global drift correction
    # Use first ref_percent of frames as reference, normalize entire shot
    # to match that reference's histogram distribution.
    # ===================================================================
    corrected_lab = lab.clone()
    correction_map = torch.zeros(num_frames, H, W, device=device)

    ref_end = ref_frames if ref_frames > 0 and num_frames > ref_frames else 0

    if ref_end > 0:
        # Compute per-frame means for each LAB channel
        all_means = []  # [3, num_frames]
        for ch in range(3):
            ch_data = lab[..., ch]  # [num_frames, H, W]
            frame_means = ch_data.reshape(num_frames, -1).mean(dim=1)  # [num_frames]
            all_means.append(frame_means)

        # Compute temporally smoothed target means
        # The target is what the brightness SHOULD be if there were no flicker/drift.
        # We use the reference zone mean, smoothly extended across the shot.
        for ch in range(3):
            vmin, vmax = ch_ranges[ch]
            frame_means = all_means[ch]  # [num_frames]
            ref_mean = frame_means[:ref_end].mean()

            # Target: reference mean for all frames — vectorized
            offsets = (ref_mean - frame_means) * strength  # [num_frames]
            corrected_lab[..., ch] = (lab[..., ch] + offsets.view(-1, 1, 1)).clamp(vmin, vmax)

            if ch == 0:
                correction_map += offsets.view(-1, 1, 1).expand_as(correction_map)

    # ===================================================================
    # Phase 2: Local boundary correction (chunk seam removal)
    # Detect brightness jumps and smooth them with CDF matching
    # ===================================================================
    # Re-extract L from drift-corrected data for boundary detection
    L_corrected = corrected_lab[..., 0]

    boundaries = detect_boundaries_auto(L_corrected, detection_threshold, sensitivity)

    if boundaries:
        # Group nearby boundaries into transition zones
        # Each zone has: stable_before -> turbulent region -> stable_after
        groups = []
        current_group = [boundaries[0]]
        for b in boundaries[1:]:
            if b - current_group[-1] <= blend_radius * 2:
                current_group.append(b)
            else:
                groups.append(current_group)
                current_group = [b]
        groups.append(current_group)

        phase2_lab = corrected_lab.clone()

        for group in groups:
            zone_start = group[0]     # first boundary in group
            zone_end = group[-1]      # last boundary in group

            # Stable reference frames: outside the turbulent zone
            stable_before_idx = max(0, zone_start - blend_radius - 1)
            stable_after_idx = min(num_frames - 1, zone_end + blend_radius)

            # Compute reference CDFs from stable frames on both sides
            before_cdfs = []
            after_cdfs = []
            for ch in range(3):
                vmin, vmax = ch_ranges[ch]
                before_cdfs.append(_compute_cdf(corrected_lab[stable_before_idx, ..., ch], vmin, vmax))
                after_cdfs.append(_compute_cdf(corrected_lab[stable_after_idx, ..., ch], vmin, vmax))

            # Blend zone: from stable_before to stable_after
            zone_len = stable_after_idx - stable_before_idx
            if zone_len <= 0:
                continue

            for i in range(stable_before_idx, stable_after_idx + 1):
                # Interpolation weight: 0.0 at stable_before, 1.0 at stable_after
                t = (i - stable_before_idx) / max(zone_len, 1)

                if grid_size > 1:
                    corrected_frame, L_corr = _apply_cdf_correction_grid(
                        corrected_lab[i], corrected_lab[stable_before_idx],
                        corrected_lab[stable_after_idx], t, strength,
                        grid_size, ch_ranges,
                    )
                    phase2_lab[i] = corrected_frame
                    correction_map[i] += L_corr
                else:
                    for ch in range(3):
                        vmin, vmax = ch_ranges[ch]
                        frame_ch = corrected_lab[i, ..., ch]
                        frame_cdf = _compute_cdf(frame_ch, vmin, vmax)

                        # Target: smooth blend from before CDF to after CDF
                        target_cdf = _blend_cdfs(before_cdfs[ch], after_cdfs[ch], t)

                        # Apply with strength
                        blended_target = _blend_cdfs(frame_cdf, target_cdf, strength)
                        matched = _histogram_match(frame_ch, frame_cdf, blended_target, vmin, vmax)
                        phase2_lab[i, ..., ch] = matched.clamp(vmin, vmax)

                        if ch == 0:
                            correction_map[i] += matched - frame_ch

        corrected_lab = phase2_lab

    # Convert back to sRGB
    corrected_images = lab_to_srgb(corrected_lab)

    # Generate debug heatmap
    heatmap = _generate_heatmap(correction_map)

    return corrected_images, heatmap
