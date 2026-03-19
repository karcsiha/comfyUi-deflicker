import torch

from .brightness_core import auto_brightness_equalize
from .flicker_core import deflicker_frames


class DeflickerFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "window_size": ("INT", {
                    "default": 15, "min": 3, "max": 99, "step": 2,
                    "tooltip": "Temporal smoothing window (frames). Larger = more aggressive.",
                }),
                "strength": ("FLOAT", {
                    "default": 1.2, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Correction strength. 0 = off, 1 = full, >1 = overcorrect.",
                }),
                "channels": (["LAB", "L"], {
                    "default": "LAB",
                    "tooltip": "LAB: brightness + color. L: brightness only.",
                }),
                "use_median": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Median pre-filter for extreme outlier frames.",
                }),
                "pixel_smoothing": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Per-pixel temporal smoothing. 0=off, 0.3-0.5=AI video.",
                }),
                "grid_size": ("INT", {
                    "default": 1, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Spatial grid for correction. 1 = global, 6 = 6x6 zones for spatially varying flicker.",
                }),
                "equalize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Auto brightness equalize: detect and smooth chunk boundary jumps after deflicker.",
                }),
                "eq_blend_radius": ("INT", {
                    "default": 5, "min": 1, "max": 20, "step": 1,
                    "tooltip": "Equalize: max frames to blend around each boundary.",
                }),
                "eq_sensitivity": ("FLOAT", {
                    "default": 1.5, "min": 1.0, "max": 6.0, "step": 0.5,
                    "tooltip": "Equalize: detection sensitivity. Lower = more sensitive.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "debug_heatmap")
    FUNCTION = "deflicker"
    CATEGORY = "deflicker"

    def deflicker(self, images, window_size, strength, channels, use_median,
                  pixel_smoothing, grid_size, equalize, eq_blend_radius, eq_sensitivity):
        # Phase 1: Temporal deflicker
        corrected, heatmap = deflicker_frames(
            images=images, window_size=window_size, strength=strength,
            channels=channels, use_median=use_median,
            pixel_smoothing=pixel_smoothing, grid_size=grid_size,
        )

        # Phase 2: Auto brightness equalize (boundary smoothing)
        if equalize:
            corrected, eq_heatmap = auto_brightness_equalize(
                images=corrected, blend_radius=eq_blend_radius,
                strength=strength, sensitivity=eq_sensitivity,
                grid_size=grid_size,
            )

        return (corrected, heatmap)
