try:
    from .nodes import DeflickerFrames

    NODE_CLASS_MAPPINGS = {
        "DeflickerFrames": DeflickerFrames,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "DeflickerFrames": "Deflicker Frames",
    }

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
except ImportError:
    # Running outside ComfyUI (e.g. pytest) — relative imports unavailable
    pass
