"""PixelMemory backend package."""

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass
