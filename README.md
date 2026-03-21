# Deflicker Frames

ComfyUI custom node for removing brightness flicker and chunk boundary artifacts in AI-generated video sequences (WAN, VACE, FramePack, etc.).

## Installation

### Option 1: Git clone (recommended)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/karcsiha/comfyUi-deflicker.git
```

Restart ComfyUI. The node will appear under the **deflicker** category.

### Option 2: Manual download

1. Download this repository as ZIP
2. Extract to `ComfyUI/custom_nodes/ComfyUI-deflicker/`
3. Restart ComfyUI

### Requirements

- PyTorch (already included with ComfyUI)
- No additional dependencies

## Usage

```
[Load Video] → [Deflicker Frames] → [Save Video]
```

Find the node under **Add Node → deflicker → Deflicker Frames**.

The node outputs two images:
- **images** — corrected frame sequence
- **debug_heatmap** — visualization of corrections (red = brightened, blue = darkened, black = no change). Connect to a Preview node to inspect.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | temporal_smoothing | **temporal_smoothing**: Gaussian window-based correction for random per-frame flicker. **step_removal**: instant correction of sharp brightness steps caused by latent space shifts — preserves natural trends, only removes discontinuities. **both**: step removal first, then temporal smoothing. |
| `window_size` | 15 | Temporal smoothing window in frames. Larger = more aggressive flicker removal. Use 11–15 for mild flicker, 21–31 for heavy flicker. Must be odd. Only used in temporal_smoothing/both modes. |
| `strength` | 1.2 | Correction strength. 0 = off, 1 = full correction, >1 = overcorrect (useful for stubborn flicker). **Caution:** values much above 1.5 may introduce artifacts. |
| `channels` | L | `L`: brightness only — preserves original color relationships. `LAB`: corrects each channel independently — removes both brightness and color temperature flicker. |
| `drift_mode` | auto | `auto`: automatically detects slow brightness trends and preserves them. `flicker_only`: removes all brightness changes including slow drift — use this if slow brightness drift is not being corrected. `preserve_trend`: always keeps slow brightness changes. |
| `use_median` | off | Median pre-filter before Gaussian smoothing. Turn on if you have extreme outlier frames (single very bright/dark frames). |
| `pixel_smoothing` | 0.0 | Per-pixel temporal smoothing. `0.0` = off. `0.3–0.5` = recommended for AI video with spatially varying flicker. **Warning:** can cause ghosting on fast-moving objects. Start low and increase carefully. |
| `grid_size` | 1 | Spatial grid for correction. `1` = global (whole frame corrected uniformly). Higher values (e.g. `6`) split the frame into zones for spatially varying flicker where different parts of the frame flicker differently. |
| `equalize` | on | Auto brightness equalize — detects and smooths chunk boundary brightness jumps after deflicker. |
| `eq_blend_radius` | 5 | Equalize: number of frames to blend around each detected boundary. |
| `eq_sensitivity` | 1.5 | Equalize: boundary detection sensitivity. Lower = detects smaller jumps (more sensitive). Range 1.0–6.0. |

## How it works

The node runs up to four correction phases depending on mode:

**Phase 0 — Step removal (when `mode` = step_removal or both)**

Detects sharp brightness discontinuities (latent space shifts) by finding frame-to-frame brightness changes that significantly exceed the normal noise level. Applies instant cumulative gain correction to undo each step. Unlike temporal smoothing, this preserves the natural brightness trend — it only removes the discrete jumps. Works well for stabilized footage generated in chunks.

**Phase 1 — Per-frame statistics correction (when `mode` = temporal_smoothing or both)**

Computes per-frame mean brightness and normalizes it across the sequence using Gaussian temporal smoothing. Automatically detects whether the sequence has an intentional brightness trend (e.g. a gradual fade) and preserves it while removing per-frame noise. When `grid_size > 1`, this runs independently per spatial zone.

**Phase 2 — Per-pixel temporal smoothing (when `pixel_smoothing > 0`)**

Applies a Gaussian-weighted temporal average per pixel across neighboring frames. Helps with flicker that varies spatially within each frame. Can soften fast motion at high values.

**Phase 3 — Auto brightness equalize (when `equalize` is on)**

Detects brightness discontinuities at chunk boundaries (common in multi-chunk AI video generation). Converts to LAB color space, finds frames with abnormal brightness jumps, and applies CDF histogram matching to smoothly blend across boundaries. When `grid_size > 1`, matching runs per spatial zone.

**Auto border masking**

Automatically detects black borders from stabilized/cropped footage (letterbox, pillarbox, irregular crops). Border pixels are excluded from all statistics computation so they don't distort the correction. Corrections are still applied to the full frame. No configuration needed — fully automatic.

## Tips

- **Start with defaults** — they work well for most AI-generated video
- **Use `step_removal` mode** for sharp latent space brightness shifts that temporal smoothing can't fix. Use `both` if you have both step shifts and random flicker.
- **Increase `window_size`** (e.g. 25–31) if flicker is still visible. For long sequences (200+ frames), values up to 101–201 can be useful.
- **Increase `strength`** above 1.0 if correction is not strong enough
- **Set `drift_mode` to `flicker_only`** if slow brightness drift is not being corrected — this removes all brightness changes, not just fast flicker
- **Turn on `pixel_smoothing`** (0.3–0.5) if different parts of the frame flicker differently
- **Increase `grid_size`** (e.g. 4–6) only if the flicker is not uniform across the frame (e.g. one side flickers more than the other). The default global correction works well for most cases.
- **Lower `eq_sensitivity`** (e.g. 1.0) to catch subtle chunk boundaries
- **Stabilized footage** with black borders is handled automatically — no need to crop first

## Technology

- Pure PyTorch — all operations are GPU-compatible tensor ops
- No external dependencies beyond torch
- Automatic black border detection for stabilized/cropped footage
- Step discontinuity detection via statistical outlier analysis
- LAB color space for perceptual boundary correction
- CDF histogram matching with 2048-bin resolution
- Adaptive trend detection via linear regression
- Polynomial least-squares target curve fitting
- Vectorized spatial grid correction via adaptive pooling + bilinear interpolation
