"""parktuber - fine-tuning TubeR (CVPR 2022) on custom CCTV footage.

Importing any submodule puts the official TubeR code (third_party/tubelet-transformer) on sys.path.
"""
from . import upstream  # noqa: F401  (side effect: sys.path setup)
